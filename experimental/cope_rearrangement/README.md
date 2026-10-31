# `cope_rearrangement/`

A PySCF study of the chair Cope rearrangement of 1,5-hexadiene. The question
is whether it goes through one concerted C2h TS once dynamic correlation is
included. CASSCF(6,6) alone overstabilises the cyclohexane-1,4-diyl; NEVPT2
is expected to remove that artifact. The deciding test is the sign of the
a_u curvature at the C2h stationary point.

Nothing here is built: each step is Python generators and their output. They
run under `uv run --group tools`, which carries PySCF and geomeTRIC.

| Step | What | State |
|---|---|---|
| [`step_1/`](step_1/README.md) | C2h chair guess, symmetry-labelled CAS(6,6) (2a_g + 2b_u + a_u + b_g), relaxed C2h DF-CASSCF(6,6)/cc-pVDZ scan of R16 = R34, 1.50–2.60 Å | done: diyl minimum at 1.65 Å, "aromatic" minimum at 2.30 Å |
| step 2 | PC-NEVPT2 and the a_u force constant along the scan | not started |
| step 3 | 2D (R16, R34) surface under C2 | not started |
| step 4 | PT2-level TS, Hessian, IRC | not started |
| step 5 | CCSD(T) and cc-pVTZ cross-checks | not started |
| step 6 | comparison to experiment | not started |
