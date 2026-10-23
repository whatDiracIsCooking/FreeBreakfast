# recipes/cg_unitary

`free_breakfast.recipes.cg_unitary`: **Riemannian conjugate gradient** for
optimization under a unitary constraint, after Abrudan, Eriksson & Koivunen,
*Conjugate gradient algorithm for optimization under unitary matrix
constraint*, Signal Processing 89 (2009) 1704–1714. Minimize or maximize a
real `J(W)` subject to `W ∈ U(n)` — `O(n)` for a real `T` — along geodesics
of the group, with the step from one of the paper's two almost-periodicity
line searches. It is what orbital localization (Pipek–Mezey, Boys) runs on.

## Module

`free_breakfast.recipes.cg_unitary`, also re-exported by
`free_breakfast.recipes`.

| Name | Role |
|---|---|
| `cg_unitary<T>(res, n, w, cost, direction, options)` | The solve: `w` in as `W_0`, out as the optimum; returns `Result<CgInfo>` |
| `UnitaryCostFunction<F, T>` | What `cost` must provide: `F::kOrder`, `euclidean_gradient`, `value` |
| `CgDirection` | `Minimize` or `Maximize` |
| `LineSearchMethod` | `Polynomial` (Table 1) or `Dft` (Table 2) |
| `CgOptions<R>` | Tuning; the defaults are the paper's (§5.1) but for `dft_samples_per_period` |
| `CgInfo` | Steps, resets, failed searches, final `<G, G>`, last step |

`T` is `float`, `double`, `wwrFloatComplex` or `wwrDoubleComplex`. `w` is
`n × n` with packed leading dimension `n`, and unitary on entry: the identity
for the paper's runs.

```cpp
struct Brockett {                       // J(W) = tr(W^H R W N), §5.1
  static constexpr int32_t kOrder = 2;  // quadratic in W
  Status euclidean_gradient(DeviceResources &res, int32_t n, const T *w, T *psi) const;
  Status value(DeviceResources &res, int32_t n, const T *w, const T *psi, T *out) const;
};
const CgInfo info = FBF_TRY(cg_unitary<T>(res, n, w, cost, CgDirection::Maximize));
```

**Not converging is an error**, as for `davidson_solve`: `MaxIterationsExceeded`
when the budget runs out, `Divergence` when no step is found even along the
gradient. `w` holds the last iterate either way.

## The idea

A constrained problem on `C^{n×n}` becomes an unconstrained one on the group
the constraint defines, and the group structure makes the pieces cheap:

- **Geodesics are exponentials**: `Γ(t) = exp(tH) W` for `H ∈ u(n)` (eq. 3),
  from `recipes/expm`'s `skew_eig` / `skew_expm`. No retraction, no
  reprojection.
- **The gradient is a skew-Hermitian part**: `G = Ψ W^H − W Ψ^H` (eq. 2), from
  the cost's Euclidean gradient `Ψ = ∂J/∂W*`, one `gemm` and one `geam`.
- **Parallel transport is a right multiplication** (eq. 5), and with the
  approximate Polak–Ribière factor of eq. (10) it drops out entirely. That is
  what makes conjugacy affordable here.

## One step (Table 3)

```
2. Psi = dJ/dW*(W);  G = Psi W^H - W Psi^H            eq. (2)
   H := G every n^2 steps, n^2 being dim U(n)
3. stop when <G, G> = 0.5 ||G||_F^2 < tol              eq. (1)
4. mu by geodesic line search                         Tables 1 or 2
5. W <- exp(sign mu H) W, sign = -1 to minimize       eq. (11), Remark 1
6. gamma = <G' - G, G'> / <G, G>;  H <- G' + gamma H   eqs. (10), (7)
7. H := G' if <H, G'> <= 0                             Remark 2
```

A search that finds nothing along a conjugate `H` is retried along `G`;
nothing along `G` either is `Divergence`. `reset_period = 1` is Riemannian
steepest descent, which is how the tests measure what conjugacy buys.

## The line searches

Along a geodesic, `exp(μH)` has eigenvalues `e^{iω_k μ}`, so a smooth `J`
restricted to it is almost periodic in `μ`, and so is its derivative. The
highest frequency present is `q |ω_max|`, `q` being the cost's `kOrder`, so in

```
T_mu = 2 pi / (q |omega_max|)                          eq. (15)
```

the derivative completes at most one cycle of its fastest component and
crosses zero at most twice (§3.1). That is why both searches fit the
*derivative* and look for its zeros. Each sample is one level-1 reduction,
`dJ/dμ = 2 sign Re tr((H W_μ)^H Ψ(W_μ))`, taken in device pointer mode so it
stays on the device.

| | Table 1 (`Polynomial`) | Table 2 (`Dft`) |
|---|---|---|
| Fits | degree-`P` polynomial over `[0, T_mu]` | Hann-windowed DFT over `N_T` periods |
| Samples | `P + 1`, one free | `N_DFT ≤ 31`, one free |
| Roots of | the interpolant, in `s = μ/h` | `z^{(N−1)/2} D(z)`, on the unit circle |
| Returns | the first zero crossing | the best of the minima, by sampled `J` |

Sample 0 is the current point, with the gradient the solver already has.

**One `heevd` per search.** `skew_eig(H)` gives `iH = U diag(w) U^H`, and every
sample is `exp(t_i H) = U diag(e^{−i t_i w}) U^H` evaluated directly: a phase
kernel and a `gemm`, with no matrix powers, so no error accumulates across
samples. The same decomposition is the step itself in step 5. It also gives
`|ω_max| = max |w|` exactly, at either end of the ascending spectrum.

**The roots come from `recipes/polynomial_roots`**, then a selector in
`select_root.cu` keeps the ones that are real zero crossings. Each root maps to
a point `x` on a real line — `Re z` for Table 1, `arg z` for Table 2 — with an
off-line part `y` — `Im z`, or `|z| − 1`. It is kept when `|y| ≤ δ` and the
real function (the polynomial, or `D(θ)`) changes sign across `[x − δ, x + δ]`,
`δ = sqrt(ε)` times the line's scale. The first test drops complex pairs; the
second drops a double root, where the derivative touches zero without
crossing. Aberth returns such a root as a pair split by `O(sqrt(ε))`, and a
fixed threshold alone would misjudge it one way or the other. Polynomials whose
roots did not settle (`iters = −1`, e.g. a zero leading coefficient) give no
step.

**Table 2 picks by direction, not by position.** The paper's step 12 takes the
1st, 3rd, 5th, … crossing, which assumes none is ever missed. Near the optimum
the first crossing sits at `θ ≈ 1e−5`, and its computed argument can land just
below 0, wrap to `≈ 2π` and sort last. Every pick after it is then the opposite
kind, and the step lands on a minimum of the `J` being maximized. So each
crossing is classified by the derivative across it instead: rising at a
minimum, falling at a maximum. The window is periodic, so its last sample
interval is mapped to just before 0.

**Table 2 defers to Table 1 inside its first sample interval.** As `G` shrinks,
`H` does too, so `T_mu` grows as `1/|G|` while the first extremum stays `O(1)`
away. It then falls inside the DFT's first sample interval, where the Hann
window has all but zeroed the samples and the reconstruction joins the
window's two ends. A DFT step that is not past one sample spacing goes to
Table 1 instead, whose unwindowed interpolant through the sample at 0 resolves
it. In practice the DFT supplies the first few large steps of a solve and
Table 1 the rest. Step 13's best sample also has to beat the first by more than
`sqrt(ε)` of the costs' scale: below that, the samples differ only by rounding,
and the "best" one can point many periods away.

**What stays on the device.** The samples, the interpolation, the DFT, the root
solve, the selection and the step choice all run on device values, in
single-block kernels. A search synchronizes three times: inside `heevd`, to
read `|ω_max|` (the sample points are host scalars), and to read the step back
for the solver's control flow.

## Changes from calaman's port

calaman (`calaman.cg_unitary`) is the source. What differs:

1. **Exponentials by eigendecomposition**, not Padé: `|ω_max|` is exact, which
   drops calaman's power iteration (`skew_spectral_radius`, its warm-start
   vector) and the `use_spectral_radius` / 1-norm surrogate option. Power
   iteration approaches `|ω_max|` from below, so a loose estimate overstated
   `T_mu`; the 1-norm overstates `|ω_max|` and shortened the steps. The exact
   value does neither.
2. **Roots from Aberth**, not a grid scan. calaman bracketed sign changes on a
   1024- (Table 1) or 2048-point (Table 2) grid in one thread, because it had
   no shared root finder. Table 2 is now the paper's own root solve, and close
   crossing pairs a grid cell would merge are resolved. Finding every root is
   what exposed the two Table 2 changes above. The grid could not see a
   crossing within `2π/2048` of 0, so calaman fell back to Table 1 there by
   accident, before the parity or the window edge could matter.
3. **Table 1 is fitted in `s = μ/h`**, so its coefficients are `O(1)` whatever
   the frequency, and the root's units are the sample spacing.
4. **No workspace argument.** The iterate, gradients, decomposition and samples
   are the call's own pool buffers, because each is live across a cost callback
   that may take `res.device_workspace()`. The steps' scratch, dead before the
   next callback, is the workspace's. calaman's `cg_unitary_bufferSize` and its
   carved layout are gone.
5. **No `reorthogonalize_period`.** The step is an exact exponential of a
   skew-Hermitian matrix, unitary to roundoff, and the paper reports the drift
   at machine precision (§5.1).
6. **Non-convergence is an error**, not an `ok()` status with a reason.
7. **Brockett lives in the test**, not the module: it is the paper's example,
   not something a caller here needs.

Kept from calaman, and still deviations from the paper: `K = 5` rather than 3,
because the 31-sample cap trades `K` against `N_T`; and the DFT search's
fallback to Table 1 when no minimum survives, rather than stopping. New
deviations, from the paper and from calaman alike, are the three Table 2 rules
above: classification by direction, the fallback inside the first sample
interval, and the rounding floor on step 13.

## Tests

`test/gtest/cg_unitary_gpu.cpp`, all four element types, on the Brockett
criterion with `R = Q diag(λ) Q^H` built from a dense unitary `Q`. The optimum
is known in closed form, `W^H R W = diag(λ)` sorted ascending for a maximum
and descending for a minimum, so no reference eigensolver is needed. Both
searches, both directions, steepest descent against CG, a cost that grows and
poisons the device workspace on every call, `U(1)` / `O(1)`, and the error
returns.

## Files

| File | Role |
|------|------|
| `cg_unitary.cppm` | Module interface: the solver template, the line search, the cost concept and options |
| `cg_unitary.cpp` | `polynomial_step` and `dft_step`, every device launch, and their four instantiations |
| `line_search.cu` | Table 1's interpolation, Table 2's windowed DFT and step choice |
| `select_root.cu` | The crossings among `polynomial_roots`' roots, for each table |
| `cg_unitary_bridge.h` | The launchers' declarations, shared by the implementation unit's GMF and the `.cu` files |
| `cg_unitary_limits.h` | `kCgMaxSamples`, the sample cap, for the interface's GMF and the kernels; no runtime header, which in an interface's GMF would collide with `wwr.runtime_api` in every importer |
| `CMakeLists.txt` | The device library and the module |
