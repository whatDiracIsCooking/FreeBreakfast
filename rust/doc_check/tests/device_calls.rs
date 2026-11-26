//! Two rules of the register (`docs/rules.md`) on `fb-core-device`'s
//! v0.0.4 runtime (`docs/gpu_layer.md`, "The v0.0.4 runtime") that a scan
//! of its two sources holds, each failing with the file and line of every
//! departure:
//!
//! - **A device call makes its stream's device current**: in the inherent
//!   `impl` of `Stream` and `Event` (`core/device/src/stream.rs`) and of
//!   `GpuResources` (`core/device/src/gpu.rs`), every `unsafe` block, the
//!   crate's one way to the runtime and the kernels, sits in the scope of
//!   a named `ScopedDevice` guard (`let current = ScopedDevice::new(…)`,
//!   `ScopedDevice::try_new(…)` or the resources' `self.current()`) bound
//!   earlier in its function, unless the block only calls an `unsafe fn` of
//!   the same `impl`, which is held to the same rule. So every public
//!   method either holds a guard over its device call or reaches the device
//!   only through a method that does. `let _ = ScopedDevice::new(…)` drops
//!   the guard at once and is no guard.
//! - **A solver call repeats its size query**: in `impl GpuResources`,
//!   every method that calls a dense-solver binding
//!   (`wwr_solver_dn_<routine>`, not a create, a set-stream, a destroy or a
//!   `_buffer_size` query) asks the method that makes that routine's
//!   `_buffer_size` query again, with arguments that are the method's own
//!   parameters, checks its work buffer against the answer with
//!   `check_work` before its first `unsafe` block, and uses the answer
//!   after the check; the method and `check_work` are `#[track_caller]`,
//!   so the panic is at the caller's line, and the query method's answer
//!   is at least one element (`.max(1)`).
//!
//! Each is a text scan, so each sees only the spelling it looks for; the
//! register's rows say what that misses. Tier 3: it reads files and
//! nothing else.

#![forbid(unsafe_code)]

use fb_doc_check::{attributes, code_only, is_ident, line_of, rust_root, word_at, words};

/// The sources and the types whose inherent `impl` the guard rule reads.
const GUARDED: [(&str, &[&str]); 2] = [
    ("core/device/src/stream.rs", &["Stream", "Event"]),
    ("core/device/src/gpu.rs", &["GpuResources"]),
];

/// The source and type the solver rule reads.
const SOLVER: (&str, &str) = ("core/device/src/gpu.rs", "GpuResources");

/// The solver bindings that are no call on a matrix.
const SOLVER_HOUSEKEEPING: [&str; 3] = ["create", "set_stream", "destroy"];

/// One method of an inherent `impl`, in a source's `code_only` text.
struct Method<'c> {
    name: &'c str,
    /// The byte of its `fn`.
    at: usize,
    /// Its body, from its `{` to one past its `}`.
    open: usize,
    close: usize,
    /// Its parameter list, between the parentheses.
    params: &'c str,
    unsafe_fn: bool,
    track_caller: bool,
}

/// The byte just past the `}` closing the `{` at `open`.
fn closing(b: &[u8], open: usize) -> usize {
    let mut depth = 0i32;
    for (j, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return j + 1;
                }
            }
            _ => {}
        }
    }
    b.len()
}

/// The byte of the `)` closing the call or list whose contents start at
/// `from`, just past its `(`.
fn args_end(code: &str, from: usize) -> usize {
    let mut depth = 0i32;
    for (j, c) in code.bytes().enumerate().skip(from) {
        match c {
            b'(' => depth += 1,
            b')' if depth == 0 => return j,
            b')' => depth -= 1,
            _ => {}
        }
    }
    code.len()
}

/// The methods of every `impl <ty> {` block of `code`, a trait's `impl …
/// for <ty>` not among them.
fn methods<'c>(code: &'c str, ty: &str) -> Vec<Method<'c>> {
    let b = code.as_bytes();
    let ws = words(code);
    let attrs = attributes(code);
    let mut out = Vec::new();
    for (k, &(_, w)) in ws.iter().enumerate() {
        if w != "impl" || ws.get(k + 1).map(|&(_, t)| t) != Some(ty) {
            continue;
        }
        let after = ws[k + 1].0 + ty.len();
        if !code[after..].trim_start().starts_with('{') {
            continue;
        }
        let open = after + code[after..].find('{').unwrap_or(0);
        let close = closing(b, open);
        // The `fn`s at the block's own depth, not a closure's or a nested
        // item's.
        let mut depth = 0i32;
        let mut prev_end = open + 1;
        let mut j = open;
        while j < close {
            match b[j] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                c if is_ident(c) && (j == 0 || !is_ident(b[j - 1])) => {
                    let word = word_at(b, j);
                    if word == "fn" && depth == 1 {
                        let name_at =
                            j + 2 + (code[j + 2..].len() - code[j + 2..].trim_start().len());
                        let name = word_at(b, name_at);
                        let paren = name_at + code[name_at..].find('(').unwrap_or(0);
                        let params_end = args_end(code, paren + 1);
                        let Some(body) =
                            (params_end..close).find(|&i| b[i] == b'{' || b[i] == b';')
                        else {
                            break;
                        };
                        if b[body] == b';' {
                            j = body + 1;
                            continue;
                        }
                        let end = closing(b, body);
                        let head = code[..j].rfind('\n').map_or(0, |p| p + 1);
                        out.push(Method {
                            name,
                            at: j,
                            open: body,
                            close: end,
                            params: &code[paren + 1..params_end],
                            unsafe_fn: code[head..j].split_whitespace().any(|t| t == "unsafe"),
                            track_caller: attrs.iter().any(|a| {
                                a.name == "track_caller" && a.start >= prev_end && a.end <= j
                            }),
                        });
                        prev_end = end;
                        j = end;
                        continue;
                    }
                    j += word.len();
                    continue;
                }
                _ => {}
            }
            j += 1;
        }
    }
    out
}

/// Each `unsafe {` block of `code[from..to]`: the byte of its `unsafe`,
/// and its contents, trimmed.
fn unsafe_blocks(code: &str, from: usize, to: usize) -> Vec<(usize, &str)> {
    let b = code.as_bytes();
    words(&code[from..to])
        .into_iter()
        .filter(|&(_, w)| w == "unsafe")
        .filter_map(|(i, _)| {
            let at = from + i;
            let rest = &code[at + 6..to];
            rest.trim_start().starts_with('{').then(|| {
                let open = at + 6 + rest.find('{').unwrap_or(0);
                let close = closing(b, open);
                (at, code[open + 1..close - 1].trim())
            })
        })
        .collect()
}

/// The `{` of the innermost block of `code` around byte `at`, searching
/// back no further than `floor`.
fn enclosing(code: &str, at: usize, floor: usize) -> usize {
    let b = code.as_bytes();
    let mut depth = 0i32;
    for j in (floor..at).rev() {
        match b[j] {
            b'}' => depth += 1,
            b'{' if depth == 0 => return j,
            b'{' => depth -= 1,
            _ => {}
        }
    }
    floor
}

/// Each named `ScopedDevice` guard bound in `m`'s body: where its `let`
/// is, and the byte past the block it lives to.
fn guards(code: &str, m: &Method<'_>) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    words(&code[m.open..m.close])
        .into_iter()
        .filter(|&(_, w)| w == "let")
        .filter_map(|(i, _)| {
            let at = m.open + i;
            let rest = code[at + 3..m.close].trim_start();
            let rest = rest.strip_prefix("mut ").map_or(rest, str::trim_start);
            let name = word_at(rest.as_bytes(), 0);
            if name.is_empty() || name == "_" {
                return None;
            }
            let value = rest[name.len()..]
                .trim_start()
                .strip_prefix('=')?
                .trim_start();
            let guard = [
                "ScopedDevice::new(",
                "ScopedDevice::try_new(",
                "self.current()",
            ]
            .iter()
            .any(|g| value.starts_with(g));
            guard.then(|| (at, closing(b, enclosing(code, at, m.open))))
        })
        .collect()
}

/// Every `unsafe` block of `source`'s inherent `impl`s of `types` that no
/// named `ScopedDevice` guard covers, each a message with its line; and
/// how many blocks were read.
fn unguarded(path: &str, source: &str, types: &[&str]) -> (Vec<String>, usize) {
    let code = code_only(source);
    let mut out = Vec::new();
    let mut read = 0;
    for ty in types {
        let ms = methods(&code, ty);
        for m in &ms {
            let gs = guards(&code, m);
            for (at, inner) in unsafe_blocks(&code, m.open, m.close) {
                read += 1;
                let callee = inner
                    .strip_prefix("self.")
                    .map(|r| word_at(r.as_bytes(), 0))
                    .filter(|c| inner[5 + c.len()..].trim_start().starts_with('('));
                if callee.is_some_and(|c| ms.iter().any(|o| o.name == c && o.unsafe_fn)) {
                    continue;
                }
                if !gs.iter().any(|&(g, end)| g < at && at < end) {
                    out.push(format!(
                        "{path}:{}: {ty}::{} reaches the device in an `unsafe` block no \
                         named `ScopedDevice` guard covers: bind `let _current = \
                         ScopedDevice::new(…)` (or `self.current()`) before it, in a scope \
                         that holds it",
                        line_of(&code, at),
                        m.name
                    ));
                }
            }
        }
    }
    (out, read)
}

/// The parameter names of a parameter list, `self` left out.
fn param_names(params: &str) -> Vec<&str> {
    params
        .split(',')
        .filter_map(|p| p.split_once(':').map(|(n, _)| n.trim()))
        .map(|n| n.strip_prefix("mut ").unwrap_or(n).trim())
        .collect()
}

/// Every departure of `source`'s `impl <ty>` from the solver rule, each a
/// message with its line; and how many solver calls were read.
fn solver_departures(path: &str, source: &str, ty: &str) -> (Vec<String>, usize) {
    let code = code_only(source);
    let ms = methods(&code, ty);
    let mut out = Vec::new();
    let mut read = 0;
    let mut fail = |at: usize, name: &str, what: String| {
        out.push(format!(
            "{path}:{}: {ty}::{name} {what}",
            line_of(&code, at)
        ));
    };
    let check_work = words(&code)
        .into_iter()
        .position(|(i, w)| w == "fn" && code[i + 2..].trim_start().starts_with("check_work"));
    let check_work_traced = check_work.is_some_and(|k| {
        let at = words(&code)[k].0;
        let head = code[..at].rfind('}').unwrap_or(0);
        attributes(&code)
            .iter()
            .any(|a| a.name == "track_caller" && a.start >= head && a.end <= at)
    });
    for m in &ms {
        let body = &code[m.open..m.close];
        for (i, w) in words(body) {
            let Some(routine) = w.strip_prefix("wwr_solver_dn_") else {
                continue;
            };
            if routine.ends_with("_buffer_size") || SOLVER_HOUSEKEEPING.contains(&routine) {
                continue;
            }
            read += 1;
            let call = m.open + i;
            let query = format!("{w}_buffer_size");
            let Some(q) = ms.iter().find(|q| {
                words(&code[q.open..q.close])
                    .iter()
                    .any(|&(_, x)| x == query)
            }) else {
                fail(
                    call,
                    m.name,
                    format!("calls {w} and no method makes its size query {query}"),
                );
                continue;
            };
            if !code[q.open..q.close].contains(".max(1)") {
                fail(
                    q.at,
                    q.name,
                    format!("answers {query} without `.max(1)`: a query of 0 gets one element"),
                );
            }
            if !m.track_caller {
                fail(
                    m.at,
                    m.name,
                    "is not #[track_caller]: a short work buffer must panic at the caller's line"
                        .into(),
                );
            }
            if !check_work_traced {
                fail(
                    m.at,
                    m.name,
                    "checks its work with a `check_work` that is missing or not #[track_caller]"
                        .into(),
                );
            }
            let ask = format!("self.{}(", q.name);
            let Some(asked) = body.find(&ask).map(|a| m.open + a) else {
                fail(
                    call,
                    m.name,
                    format!("calls {w} without asking {}() again", q.name),
                );
                continue;
            };
            let args_at = asked + ask.len();
            let args = &code[args_at..args_end(&code, args_at)];
            let own = param_names(m.params);
            if let Some(stray) = args.split(',').map(str::trim).find(|a| !own.contains(a)) {
                fail(
                    asked,
                    m.name,
                    format!(
                        "asks {}() at `{stray}`, not one of its own arguments",
                        q.name
                    ),
                );
            }
            // The answer's name: `let <need> = self.<query>(…)`.
            let line = code[..asked].rfind('\n').map_or(0, |p| p + 1);
            let need = code[line..asked]
                .trim()
                .strip_prefix("let ")
                .map(|r| word_at(r.trim_start().as_bytes(), 0))
                .unwrap_or("");
            let first_unsafe = unsafe_blocks(&code, m.open, m.close)
                .first()
                .map_or(m.close, |&(u, _)| u);
            let checked = body.find("check_work(").map(|c| m.open + c).filter(|&c| {
                let from = c + "check_work(".len();
                let args = &code[from..args_end(&code, from)];
                c > asked && !need.is_empty() && args.split(',').any(|a| a.trim() == need)
            });
            match checked {
                None => fail(
                    asked,
                    m.name,
                    format!("does not check its work buffer against {}()'s answer with check_work", q.name),
                ),
                Some(c) if c > first_unsafe => fail(
                    c,
                    m.name,
                    "checks its work buffer after an `unsafe` block: check before anything is enqueued".into(),
                ),
                Some(c) => {
                    let after = &code[args_end(&code, c + "check_work(".len())..m.close];
                    if !words(after).iter().any(|&(_, x)| x == need) {
                        fail(c, m.name, format!("never uses the query's answer `{need}` after the check"));
                    }
                }
            }
        }
    }
    (out, read)
}

fn source(path: &str) -> String {
    std::fs::read_to_string(rust_root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn every_device_call_makes_its_streams_device_current() {
    let mut failures = Vec::new();
    for (path, types) in GUARDED {
        let (bad, read) = unguarded(path, &source(path), types);
        assert!(
            read > 0,
            "{path}: no `unsafe` block found in the impls of {types:?}"
        );
        failures.extend(bad);
    }
    assert!(
        failures.is_empty(),
        "{} device call(s) with no ScopedDevice guard (docs/gpu_layer.md, \"The v0.0.4 \
         runtime\"):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn every_solver_call_repeats_its_size_query() {
    let (path, ty) = SOLVER;
    let (failures, read) = solver_departures(path, &source(path), ty);
    assert!(read > 0, "{path}: no solver call found in impl {ty}");
    assert!(
        failures.is_empty(),
        "{} solver call(s) departing from the size-query rule (docs/gpu_layer.md, \"The \
         v0.0.4 runtime\"):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A `Stream` and an `Event` whose methods each break, or keep, the guard
/// rule once; `ok_*` keep it.
const GUARD_FIXTURE: &str = r#"
impl Stream {
    pub fn ok_held(&self) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: a comment's `unsafe { x() }` is not code.
        runtime_status(unsafe { wwr_sys::wwr_stream_synchronize(self.raw) })
    }
    pub fn ok_delegated(&self) -> Status {
        // SAFETY: the callee holds the guard.
        unsafe { self.copy(1) }
    }
    pub fn ok_no_device(&self) -> i32 {
        self.physical
    }
    pub fn none(&self) -> Status {
        runtime_status(unsafe { wwr_sys::wwr_stream_synchronize(self.raw) })
    }
    pub fn dropped_at_once(&self) -> Status {
        let _ = ScopedDevice::new(self.physical);
        runtime_status(unsafe { wwr_sys::wwr_stream_synchronize(self.raw) })
    }
    pub fn out_of_scope(&self) -> Status {
        {
            let _current = ScopedDevice::new(self.physical);
        }
        runtime_status(unsafe { wwr_sys::wwr_stream_synchronize(self.raw) })
    }
    pub fn too_late(&self) -> Status {
        let status = unsafe { wwr_sys::wwr_stream_synchronize(self.raw) };
        let _current = ScopedDevice::new(self.physical);
        runtime_status(status)
    }
    unsafe fn copy(&self, bytes: usize) -> Status {
        let _current = ScopedDevice::try_new(self.physical);
        runtime_status(unsafe { wwr_sys::wwr_memcpy_async(bytes) })
    }
    unsafe fn unguarded_callee(&self) -> Status {
        runtime_status(unsafe { wwr_sys::wwr_memcpy_async(0) })
    }
    pub fn to_unguarded(&self) -> Status {
        unsafe { self.unguarded_callee() }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = unsafe { wwr_sys::wwr_stream_destroy(self.raw) };
    }
}

impl Event {
    pub fn ok_loop(&self) -> Status {
        for _ in 0..2 {
            let current = ScopedDevice::new(self.physical);
            unsafe { fb_kernel(current) };
        }
        Ok(())
    }
    pub fn kernel(&self) -> Status {
        unsafe { fb_recipes_sys::fb_inverse_sqrt(self.raw) };
        Ok(())
    }
}
"#;

#[test]
fn a_device_call_without_a_guard_in_scope_is_found() {
    let (bad, read) = unguarded("f.rs", GUARD_FIXTURE, &["Stream", "Event"]);
    let named: Vec<&str> = bad
        .iter()
        .map(|m| {
            m.split(": ")
                .nth(1)
                .and_then(|s| s.split(' ').next())
                .unwrap_or("")
        })
        .collect();
    assert_eq!(
        named,
        [
            "Stream::none",
            "Stream::dropped_at_once",
            "Stream::out_of_scope",
            "Stream::too_late",
            "Stream::unguarded_callee",
            "Event::kernel",
        ],
        "{bad:#?}"
    );
    // The `Drop` impl is a trait's, not read; every block of the two
    // inherent impls is.
    assert_eq!(read, 11);
    assert!(bad[0].starts_with("f.rs:16:"), "{bad:#?}");
}

/// A `GpuResources` with one solver call that keeps the rule, `syevd`,
/// and one per departure, each over the routine named in the call.
const SOLVER_FIXTURE: &str = r#"
#[track_caller]
fn check_work(routine: &str, n: usize, len: usize, need: usize) {
    assert!(len >= need, "short");
}

impl GpuResources {
    pub fn syevd_lwork(&self, n: usize) -> Result<usize> {
        let mut lwork = 0;
        solver_status("q", unsafe { wwr_sys::wwr_solver_dn_dsyevd_buffer_size(n, &mut lwork) })?;
        Ok(usize::try_from(lwork).unwrap_or(0).max(1))
    }
    #[track_caller]
    pub fn syevd(&mut self, n: usize, work: &mut impl DeviceSliceMut<f64>) -> Status {
        let need = self.syevd_lwork(n)?;
        check_work("syevd", n, work.len(), need);
        let _current = self.current();
        let lwork = i32::try_from(need).unwrap_or(i32::MAX);
        solver_status("syevd", unsafe { wwr_sys::wwr_solver_dn_dsyevd(n, work, lwork) })
    }
    pub fn gesvd_lwork(&self, m: usize) -> Result<usize> {
        let mut lwork = 0;
        solver_status("q", unsafe { wwr_sys::wwr_solver_dn_dgesvd_buffer_size(m, &mut lwork) })?;
        Ok(usize::try_from(lwork).unwrap_or(0))
    }
    #[track_caller]
    pub fn gesvd(&mut self, m: usize, work: &mut impl DeviceSliceMut<f64>) -> Status {
        let need = self.gesvd_lwork(64)?;
        let _current = self.current();
        solver_status("gesvd", unsafe { wwr_sys::wwr_solver_dn_dgesvd(m, work, need) })?;
        check_work("gesvd", m, work.len(), need);
        Ok(())
    }
    pub fn getrf(&mut self, n: usize, work: &mut impl DeviceSliceMut<f64>) -> Status {
        let _current = self.current();
        solver_status("getrf", unsafe { wwr_sys::wwr_solver_dn_dgetrf(n, work, work.len()) })
    }
    pub fn potri_lwork(&self, n: usize) -> Result<usize> {
        let mut lwork = 0;
        solver_status("q", unsafe { wwr_sys::wwr_solver_dn_dpotri_buffer_size(n, &mut lwork) })?;
        Ok(usize::try_from(lwork).unwrap_or(0).max(1))
    }
    #[track_caller]
    pub fn potri(&mut self, n: usize, work: &mut impl DeviceSliceMut<f64>) -> Status {
        let need = self.potri_lwork(n)?;
        check_work("potri", n, work.len(), need);
        let _current = self.current();
        solver_status("potri", unsafe { wwr_sys::wwr_solver_dn_dpotri(n, work, work.len()) })
    }
}
"#;

#[test]
fn a_solver_call_that_skips_its_size_query_is_found() {
    let (bad, read) = solver_departures("f.rs", SOLVER_FIXTURE, "GpuResources");
    assert_eq!(read, 4, "{bad:#?}");
    let has = |s: &str| bad.iter().any(|m| m.contains(s));
    assert!(!bad.iter().any(|m| m.contains("::syevd")), "{bad:#?}");
    assert!(has("GpuResources::gesvd_lwork answers"), "{bad:#?}");
    assert!(
        has("GpuResources::gesvd asks gesvd_lwork() at `64`"),
        "{bad:#?}"
    );
    assert!(
        has("GpuResources::gesvd checks its work buffer after"),
        "{bad:#?}"
    );
    assert!(
        has("GpuResources::getrf calls wwr_solver_dn_dgetrf and no method"),
        "{bad:#?}"
    );
    assert!(
        has("GpuResources::potri never uses the query's answer"),
        "{bad:#?}"
    );
    assert_eq!(bad.len(), 5, "{bad:#?}");

    // Untraced, the same calls fail at the caller's line no longer.
    let untraced = SOLVER_FIXTURE.replace("#[track_caller]", "");
    let (bad, _) = solver_departures("f.rs", &untraced, "GpuResources");
    assert!(
        bad.iter()
            .any(|m| m.contains("::syevd is not #[track_caller]")),
        "{bad:#?}"
    );
    assert!(
        bad.iter()
            .any(|m| m.contains("`check_work` that is missing")),
        "{bad:#?}"
    );
}
