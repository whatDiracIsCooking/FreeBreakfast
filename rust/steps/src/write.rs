//! The run's products and its record on disk: [`Run::write`], on the host
//! after execute mode, from the `Output`s' host copies, so no step's `run`
//! allocates or waits on the disk for them.
//!
//! | File | Written | What it holds |
//! |---|---|---|
//! | `budget.txt` | always | the budget table, [`Run::table`]: `--budget-only`'s text |
//! | `timing.json` | always | the run's times: `profile` (`debug` or `release`), `setup`, `budget` and `execute`, and per step of the last execute its [`StepTime`], each time integer nanoseconds, each lock its rank's name (`Manager(Eri)`, `Gpu(0)`); `steps` is empty for a dry run |
//! | `overlap.npy` | for `calculation = "overlap"`, after execute | S, `n_ao × n_ao` float64 |
//! | `d.npy` | for `calculation = "guess"`, after execute | the guess's density, `n_ao × n_ao` float64 |
//! | `scf.json` | with `[scf]`, after execute | `energy`, `residual`, `iterations`, `converged`, `n_alpha`, `n_beta` and `epsilon`, ascending, each double in its shortest round-trip form |
//! | `c.npy`, `d.npy` | with `[scf]`, after execute | C and D, `n_ao × n_ao` float64 in Fortran order ([`fb_io_npy::write_npy`]), the host copies' bytes as they are |
//! | `localization.json` | with `[localization]`, after execute | `mode`, `exponent`, `functional_before`, `functional_after` (`L` of the SCF's orbitals and of the localized ones) and `sweeps`, each summed over the blocks |
//! | `c_local.npy` | with `[localization]`, after execute | `C_local`, `n_ao × n_alpha` float64 in Fortran order, each block's columns ascending in `⟨i\|F\|i⟩` |
//!
//! A step's products are one private function of its `Output` here, called
//! from [`Run::write`] when the config has the step's table and execute
//! mode ran.
//!
//! [`Run::table`]: crate::Run::table
//! [`StepTime`]: fb_core_flow::StepTime

use std::path::Path;

use fb_core_error_handling::{Error, Status};
use fb_core_flow::Bus;
use serde_json::{Value, json};

use crate::{
    Calculation, Dims, GuessStep, LocalizationOutput, LocalizationStep, LocalizationTable,
    OverlapStep, Run, ScfOutput, ScfStep,
};

impl Run {
    /// Write the run's record into `dir`, and, after an execute mode, each
    /// step's products: the files of the module's table. `dir` is made if
    /// absent; a file already there is replaced.
    ///
    /// A dry run writes the record alone:
    ///
    /// ```
    /// use fb_core_budget::Backing;
    /// use fb_steps::{Config, run};
    ///
    /// let data = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data"));
    /// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n";
    /// let run = run(Config::parse(text, data).unwrap(), Backing::Dry).unwrap();
    /// let dir = std::env::temp_dir().join(format!("fb-steps-write-doc-{}", std::process::id()));
    /// run.write(&dir).unwrap();
    /// assert_eq!(std::fs::read_to_string(dir.join("budget.txt")).unwrap(), run.table);
    /// assert!(!dir.join("scf.json").exists());
    /// std::fs::remove_dir_all(&dir).unwrap();
    /// ```
    ///
    /// # Errors
    /// `Io`, naming the directory or the file that cannot be written.
    pub fn write(&self, dir: &Path) -> Status {
        std::fs::create_dir_all(dir).map_err(|e| {
            Error::io(format!(
                "cannot make the output directory '{}': {e}",
                dir.display()
            ))
        })?;
        write_text(&dir.join("budget.txt"), &self.table)?;
        write_json(&dir.join("timing.json"), &self.timing())?;
        if self.steps.is_empty() {
            return Ok(());
        }
        let shared = self.plan.bus().shared();
        let n = shared.dims.n_ao;
        match shared.config.calculation {
            Some(Calculation::Overlap) => {
                if let Some(s) = self.plan.output(OverlapStep).s_host() {
                    fb_io_npy::write_npy(&dir.join("overlap.npy"), &[n, n], s)?;
                }
            }
            Some(Calculation::Guess) => {
                if let Some(d) = self.plan.output(GuessStep).d_host() {
                    fb_io_npy::write_npy(&dir.join("d.npy"), &[n, n], d)?;
                }
            }
            _ => {}
        }
        if shared.config.scf.is_some() {
            write_scf(&self.plan.output(ScfStep), &shared.dims, dir)?;
        }
        if let Some(table) = &shared.config.localization {
            write_localization(
                &self.plan.output(LocalizationStep),
                table,
                &shared.dims,
                dir,
            )?;
        }
        Ok(())
    }

    /// `timing.json`'s object.
    fn timing(&self) -> Value {
        let ns = |d: std::time::Duration| d.as_nanos();
        let steps: Vec<Value> = self
            .steps
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "run": ns(s.run),
                    "sync": ns(s.sync),
                    "began": ns(s.began),
                    "ended": ns(s.ended),
                    "locks": s.locks.iter().map(|l| format!("{l:?}")).collect::<Vec<_>>(),
                    "waited": s.waited,
                    "held": ns(s.held),
                    "released": ns(s.released),
                })
            })
            .collect();
        json!({
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "setup": ns(self.setup),
            "budget": ns(self.budget),
            "execute": ns(self.execute),
            "steps": steps,
        })
    }
}

/// The SCF's products: `scf.json`, `c.npy` and `d.npy`.
fn write_scf(scf: &ScfOutput, dims: &Dims, dir: &Path) -> Status {
    let n = dims.n_ao;
    write_json(
        &dir.join("scf.json"),
        &json!({
            "energy": scf.energy(),
            "residual": scf.residual(),
            "iterations": scf.iterations(),
            "converged": scf.converged(),
            "n_alpha": dims.n_alpha,
            "n_beta": dims.n_beta,
            "epsilon": scf.epsilon(),
        }),
    )?;
    fb_io_npy::write_npy(&dir.join("c.npy"), &[n, n], scf.c())?;
    fb_io_npy::write_npy(&dir.join("d.npy"), &[n, n], scf.d())
}

/// The localization's products: `localization.json` and `c_local.npy`.
fn write_localization(
    out: &LocalizationOutput,
    table: &LocalizationTable,
    dims: &Dims,
    dir: &Path,
) -> Status {
    write_json(
        &dir.join("localization.json"),
        &json!({
            "mode": "pipek_mezey",
            "exponent": table.exponent,
            "functional_before": out.functional_before(),
            "functional_after": out.functional_after(),
            "sweeps": out.sweeps(),
        }),
    )?;
    fb_io_npy::write_npy(
        &dir.join("c_local.npy"),
        &[dims.n_ao, out.n_cols()],
        out.c(),
    )
}

/// `value`, pretty-printed, to `path`; serde_json writes each double in its
/// shortest round-trip form.
fn write_json(path: &Path, value: &Value) -> Status {
    write_text(path, &format!("{value:#}\n"))
}

/// `text` to `path`, a failure an `Io` error naming it.
fn write_text(path: &Path, text: &str) -> Status {
    std::fs::write(path, text)
        .map_err(|e| Error::io(format!("cannot write '{}': {e}", path.display())))
}
