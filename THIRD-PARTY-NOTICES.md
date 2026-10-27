# Third-party notices

free-breakfast itself is MIT-licensed; see [`LICENSE`](LICENSE). This file
records the third-party material the tree **carries or derives from**, and the
attribution each one's licence asks for. It is not a dependency list: the
build's own libraries are in the last section, and the reference codebases read
during development are not here at all, because nothing was taken from them.

Everything below is one of three kinds:

- **Data compiled into the binary** — `data/basis/*.json`, which
  `src/io/basis/CMakeLists.txt` wraps verbatim in a raw string literal in a
  generated module implementation unit. A built binary therefore *contains*
  these documents, which is why the binary-form clauses matter to anyone
  redistributing one.
- **Code derived from a third-party implementation** — one module.
- **Acknowledgements with no obligation attached**, kept so this file and the
  source comments agree.

## Basis Set Exchange — BSD 3-Clause

Copyright (c) 2020 The Molecular Sciences Software Institute, Virginia Tech

Four of the five documents in `data/basis/` are Basis Set Exchange JSON,
downloaded by `tools/basis/bse_download.py` from
<https://www.basissetexchange.org/> and compiled into the binary:

| File | BSE name | Revision |
| --- | --- | --- |
| `sto-3g.json` | STO-3G | 2018-06-19 |
| `6-31g.json` | 6-31G | 2018-06-19 |
| `cc-pVDZ.json` | cc-pVDZ | 2018-10-03 |
| `cc-pVTZ.json` | cc-pVTZ | 2018-10-03 |

Each document keeps its own BSE header, including the per-element
`references` and the revision metadata, so the provenance of every shell
travels in the file itself. No exponent or coefficient was edited. One
modification was made: in `6-31g.json` the `gto_cartesian` d shells of K–Kr
are relabelled `gto_spherical` by `tools/basis/cartesian_to_spherical.py`,
which also updates the header's `function_types` and appends a note saying so
to its `revision_description`. `data/basis/README.md` records the exact
commands and the element ranges.

Full licence text: [Appendix B](#appendix-b-bsd-3-clause-licence). Clause 1
(source form) and clause 2 (binary form) are both engaged — the data is in the
repository *and* in any binary built from it.

The underlying basis-set parameters are published numerical data from the
primary literature, cited per element inside each document; the BSD 3-Clause
notice above is for MolSSI's compilation and distribution of them.

## PySCF — Apache License 2.0

Copyright 2014-2024 The PySCF Developers.

PySCF (<https://pyscf.org>) is used two ways that need attribution, and a third
that does not.

### 1. `data/basis/minao.json` (data)

Converted from PySCF 2.14.0's `pyscf/gto/basis/minao.py` by
`tools/minao_from_pyscf.py`, and compiled into the binary like the BSE files.
MINAO is not on the Basis Set Exchange, so PySCF is the carrier this tree reads
it from. The table itself is Molpro's, created by Gerald Knizia as the first
n_occ contracted functions of cc-pVTZ (J. Chem. Theory Comput. **9**, 4834
(2013), doi:10.1021/ct400687b).

**Changes made:** reshaped from a Python literal into the same BSE-schema JSON
the reader already understands; each value written as its shortest round-trip
decimal, so the parser recovers exactly the double PySCF holds; element range
limited to Z 1-18 and 20-36. The file's own `revision_description` names the
source module and version, and each element's `reference_description` credits
Knizia's paper and names PySCF as the carrier. `data/basis/README.md` has the
regeneration command and the reasoning.

`data/sad/MINAO/*.npy` is computed **over** this basis by this project's own
generator (`tools/sad`, `free-breakfast-sad-stamps`); those stamps are
free-breakfast's output, not PySCF content.

### 2. `src/response/hessian/` (code — a derivative work)

`free_breakfast.response.hessian` (`hessian.cpp`, `hessian.cppm`) implements the
exact CASSCF orbital-orbital Hessian-vector product and its diagonal, and it is
a derivative of `pyscf/mcscf/mc1step.py`'s `gen_g_hop` — specifically its
`h_op`, `h_diag` and `update_jk_in_ah`. The decomposition into intermediates,
and the order they are formed in, follow PySCF's.

This is stated throughout the source and history rather than only here:
`hessian.cppm:6`, `src/response/hessian/README.md`, `src/response/README.md`,
the commit that added the module, and issue #250 / PR #272, which specify the
deliverable by reference to `gen_g_hop`. The module was added as
`free_breakfast.mcscf.hessian` in `src/mcscf/hessian/` and moved to
`src/response/hessian/` by issue #372, unchanged.

**Changes made:**

- Ported from Python/NumPy to C++23 named modules over GPU BLAS, targeting
  either CUDA or HIP; the product and the diagonal are device-resident.
- Scaled by two throughout, matching this tree's gradient convention
  (`H = 2 (X2 - X2^T)`), so it is twice `h_op` and twice `h_diag`.
- J and K of the transition densities are built **integral-direct** on the GPU
  through this project's own `coulomb` and `exchange` over `U C`, not through
  PySCF's `get_jk`; every density is scaled under this tree's `|D| <= 2` prune
  cap, with the cap factor read back per build.
- The 1-RDM is carried as its occupied block only.
- `hdm2` is built once in `create` by strided-batched gemms over this tree's
  `ppaa` and `papa`, so the contraction is a single gemv per product.
- The diagonal is assembled on the host in O(pairs) from device diagonals read
  back once, including one J and one K per core orbital.
- Screening makes the product only approximately linear; the resulting
  asymmetry is measured per source and bounded absolutely at
  `100 tau max|H| ||x|| ||y||`, recorded in the module README.

Full licence text: [Appendix A](#appendix-a-apache-license-20). The attribution
notices from PySCF's `NOTICE`, which Apache-2.0 section 4(d) asks be carried,
are reproduced in [Appendix C](#appendix-c-pyscf-notice).

### 3. Reference data for the test suite (no obligation)

`test/gtest/data/` holds numerical reference values produced by **running**
PySCF — the `tools/emit/*_reference.py` scripts import it and record its
output. Those scripts vendor no PySCF code, and the output of running a program
is not covered by the program's licence, so nothing is owed for this. It is
listed because the dependency is real and worth knowing about: PySCF is an
optional tooling dependency (`[dependency-groups] tools` in `pyproject.toml`),
not a build or runtime one.

## GPU4PySCF — Apache License 2.0 (acknowledgement)

Copyright 2021-2024 The PySCF Developers. All Rights Reserved. (GPU4PySCF
ships no `NOTICE`; this is the copyright line its own sources carry.)

`src/recipes/orthogonalize/orthogonalize.cpp` sets its `cond(S)` warning
threshold to `1e10`, the value GPU4PySCF's `check_linear_dependency` warns at,
and the comment there says so. A single numeric threshold is not a
copyrightable work and no attribution is required; this entry exists so that
the source comment and this file do not disagree. No GPU4PySCF code is present
in the tree.

## Build and link dependencies

None of these is vendored in this repository — they are fetched or compiled by
the build and the container image — so this repository redistributes none of
them. Their notices are listed for anyone redistributing a **built binary**,
which links them.

| Component | Version | Licence | How it arrives |
| --- | --- | --- | --- |
| spdlog | v1.15.3 | MIT (c) 2016 Gabi Melman | prebuilt into `/opt` by `docker/Dockerfile.base`, `find_package` |
| Taskflow | v4.0.0 | MIT (c) 2018-2025 Dr. Tsung-Wei Huang | same |
| toml++ | v3.4.0 | MIT (c) Mark Gillard | same |
| nlohmann/json | v3.11.3 | MIT (c) 2013-2022 Niels Lohmann | `FetchContent` (`deps/CMakeLists.txt`) |
| GoogleTest | v1.17.0 | BSD 3-Clause (c) 2008 Google Inc. | `FetchContent`, declared by WarpWraps' own `deps/` |

`deps/WarpWraps` and `deps/FelizJueves` are submodules by this project's author
and are MIT-licensed; they are not third-party for the purpose of this file.

---

## Appendix A: Apache License 2.0

Applies to the PySCF material in this file.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Appendix B: BSD 3-Clause licence

Applies to the Basis Set Exchange material in this file.

```
Copyright (c) 2020 The Molecular Sciences Software Institute, Virginia Tech

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Appendix C: PySCF NOTICE

Reproduced from PySCF's `NOTICE`, as Apache-2.0 section 4(d) asks.

```
Copyright 2014-2024 The PySCF Developers.
PySCF (www.pyscf.org) was developed by:

Qiming Sun
Sandeep Sharma (University of Colorado Boulder)
Alexander Sokolov (The Ohio State University)
George H. Booth (King's College London)
Wirawan Purwanto (William & Mary)
Sheng Guo (Caltech)
Sebastian Wouters (Ghent University)
Zhendong Li (Beijing Normal University)
Jun Yang (Hong Kong University)
Carlos Jimenez-Hoyos (Wesleyan University)
Junzi Liu (John Hopkins; University of Science and Technology Beijing)
Garnet Chan (Princeton University; Caltech)
Timothy Berkelbach (Princeton University; University of Chicago; CCQ)
James D. McClain (Princeton University; Caltech)
Nick Blunt (LBNL)
Elvira Sayfutyarova (Princeton University; Yale University)
Jan Hermann (MPI)
Peter Koval (Donostia International Physics Center)
Marc Barbry (CFM Materials Physics Center; Simbeyond)
Susi Lehtola (LBNL; University of Helsinki; MolSSI)
Paul J. Robinson (UCLA)
Narbe Mardirossian (Caltech)
Lucas Wagner (University of Illinois)
Mark J. Williamson (Cambridge University)
Boxiao Zheng (Princeton University)
Bastien Mussard (University of Colorado Boulder)
Kevin Koh (University of Notre dame)
Janus Juul Eriksen (Johannes Gutenberg University Mainz)
Li Chen (University of Illinois)
Junhao Li (Cornell University)
Vijay Gopal Chilkuri (LPCQ)
Zhihao Cui (Caltech)
Nikolay A. Bogdanov (MPI FKF Stuttgart)
Artem Pulkin (Caltech)
James E. T. Smith (CCQ, Flatiron Institute)
Chong Sun (Caltech)
Jason Yu (Caltech)
Shining Sun (Caltech)
Alec White (Caltech)
Bryan Lau (Columbia University)
Matthew R. Hermes (University of Chicago)
Yang Gao (Caltech)
Xiao Wang (CCQ)
Henry Schurkus (Caltech)
Jia Chen (University of Florida)
Henrik Larsson (Caltech)
Shiv Upadhyay (University of Pittsburgh)
Samragni Banerjee (The Ohio State University)
Matthew Treinish (IBM)
Xing Zhang (Caltech)
Tianyu Zhu (Caltech)
Lee-Ping Wang (UC Davis)
Kevin Gasperich (University of Pittsburgh)
Jesse Gorzinski (IBM)
Zhi Wang (University of Chicago)
Maximilian Scheurer (Ruprecht-Karls University)
Peng Bao (Chinese Academy of Sciences)
Peter Pinski (HQS Quantum Simulations)
Oliver Backhouse (King's College London)
Matthew Chan (Tencent America LLC)
Verena A. Neufeld (Columbia University)
Max Nusspickel (King's College London)
Abdelrahman Ahmed (The Ohio State University)
Terrence Stahl (The Ohio State University)
Hung Q. Pham (Columbia University)
Eric Hermes
Hongzhou Ye
Huanchen Zhai
Kevin Focke
Matthias Degroote
Muhammad Firmansyah Kasim
Simon Ewing
Yixiao Chen
Zhenyu Zhu
Shirong Wang
Xubo Wang
Kyle Bystrom
Minye Zhang
Yusuke Kimura
Unathi Koketso Skosana
Matthew R Hennefarth (University of Chicago)
Will Wheeler
Basil Ibrahim
Junjie Yang
Pablo del Mazo Sevillano
Allan Chain
Xin Xing
Seunghoon Lee
Chia-Nan Yeh
Chun-Yu Chow
odidev (from puresoftware)
Nike Dattani
Egor Trushin
Wanja Schulze
Till Hanke
Kevin J. Sung
Jonathan Edward Moussa
Xiaojie Wu
Pavel Pokhilko
Frédéric Chapoton
Daniel King
Jiachen Li
Felipe S. S. Schneider
Aniruddha Seal
Peter Reinholdt
Christopher Hillenbrand
scohenjanes5
Michal Krompiec
Victor Yu
James Serna (The Ohio State University)
Brian Zhao
Juan Jose Aucar
Ivan Chernyshov
kvkarandashev
Jonas Greiner
Steffen Thomas
Ethan Vo
Raehyun Kim
Sebastian Ehlert
Thijs Vogels
Andrew Jenkins
Al Nejati
Oskar Weser
Yuanheng Wang (ByteDance Seed)
Weitang Li
Sander Cohen-Janes
Yu Jin
Dayou Zhang
Kori Smyser
Nathan Gillispie
Ning-Yuan Chen
Chaoqun Zhang
Anton Rikus
Jonathon Vandezande
Jakub Lang
Erwan Privat
Emerson Hsieh

---
* This list is not the contribution order. The order is roughly based on the
  time of the first commit from each contributor.
* Affiliations may not be the present affiliation of each contributor. If you
  wish to update your affiliation to any form you like, please write email to
  Xing Zhang (xzhang8@caltech.edu) or Qiming Sun (osirpt.sun@gmail.com)
```
