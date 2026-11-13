# 0018: Accept dependency licences outside the default families

Status: accepted, 2026-11-09. numpy and torch were accepted later the same day, once
the GCC runtime libraries in their Linux wheels had been read (see the last sections).

## Context

`scripts/check_licences.py` reads `pip-licenses --format=json` and accepts the MIT,
BSD, Apache, PSF, ISC, MPL-2.0, LGPL and Unlicense families. On 2026-11-09 it ran over
three environments synced from `uv.lock`: the runtime set (`uv sync --no-dev`, which
is what the container image installs), the runtime set plus the `compare` extra, and
the development set. Six packages failed: augurs 0.10.3, dd-unit-scaling 0.1.0,
numpy 2.5.4, regex 2026.9.29, text-unidecode 1.3 and torch 2.15.0.

What ships matters more than what is installed. The promcast wheel declares its
dependencies and contains none of them. The image runs
`uv sync --frozen --no-dev --no-editable` on Linux amd64 and arm64, so it
redistributes the Linux wheel of every runtime dependency, bundled libraries
included. Neither the wheel nor the image contains the `compare` extra or the
development group.

Each fact below was read on 2026-11-09 from the wheel locked in `uv.lock` (its
SHA-256 checked against the lock) and from upstream at the released version.

## augurs 0.10.3

- Runtime: a direct dependency, used by `core/baselines.py` for the ETS and MSTL
  baselines. Shipped in the image.
- Metadata: no `License` field, which `pip-licenses` reports as `UNKNOWN`; the wheel
  carries `LICENSE-APACHE` and `LICENSE-MIT`, and the sdist's `Cargo.toml` declares
  `license = "MIT OR Apache-2.0"`.
- Bundled: the extension statically links Rust crates. The wheel's CycloneDX SBOM
  (`augurs-0.10.3.dist-info/sboms/pyaugurs.cyclonedx.json`) lists 87 components on
  x86_64 and 86 on aarch64, each under MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause,
  `Apache-2.0 WITH LLVM-exception`, a choice that includes MIT or Apache-2.0, or
  `(MIT OR Apache-2.0) AND Unicode-3.0` (unicode-ident). None is copyleft.
- Sources: https://github.com/grafana/augurs/blob/pyaugurs-v0.10.3/LICENSE-MIT,
  https://github.com/grafana/augurs/blob/pyaugurs-v0.10.3/LICENSE-APACHE.
- Decision: accepted under MIT OR Apache-2.0.

## dd-unit-scaling 0.1.0

- Runtime: pulled in by toto-models 1.0.0 through toto-2 2.0.0. Shipped in the image.
- Metadata: no licence field, no classifier and no licence file in the wheel or the
  sdist. Upstream added PyPI licence metadata on 2026-09-08 (DataDog/toto commit
  `c079fd7e32424c5d33f9e267c5d32a5630242eb5`), after this release.
- Origin: PyPI received 0.1.0 on 2026-06-04 from the `pypi-publish-dd-unit-scaling`
  workflow in DataDog/toto. The sdist's `pyproject.toml` and its five modules are
  byte-identical to `dd_unit_scaling/` at commit
  `44ea4e88852228039564aa3e76fac26aafac0803` (2026-06-03). Every module opens with
  "Unless explicitly stated otherwise all files in this repository are licensed under
  the Apache-2.0 License" and "Copyright 2026 Datadog, Inc.", and the repository's
  `LICENSE` at that commit is Apache-2.0.
- Bundled: nothing. Its one dependency, unit-scaling, is Apache-2.0 and passes the check.
- Sources: https://github.com/DataDog/toto/blob/44ea4e88852228039564aa3e76fac26aafac0803/LICENSE,
  https://github.com/DataDog/toto/blob/44ea4e88852228039564aa3e76fac26aafac0803/NOTICE,
  https://pypi.org/project/dd-unit-scaling/0.1.0/.
- Decision: accepted under Apache-2.0. The wheel ships without the upstream `LICENSE`
  and `NOTICE`, which Apache-2.0 section 4 asks a redistributor to include, so the image
  carries both in `/opt/promcast/licences/dd-unit-scaling/`, fetched at the commit above.

## regex 2026.9.29

- Optional: pulled in by transformers 5.22.1 through chronos-forecasting 2.3.2, only
  with the `compare` extra. Not in the wheel or the image.
- Metadata: `License-Expression: Apache-2.0 AND CNRI-Python`. `LICENSE.txt` says the
  work derives from the `re` module of CPython 2.6 and 3.1 under CNRI's Python 1.6
  licence, and that all additions are Apache-2.0. CNRI-Python is a permissive,
  OSI-approved licence.
- Bundled: nothing beyond that derived code.
- Source: https://github.com/mrabarnett/mrab-regex/blob/2026.9.29/LICENSE.txt.
- Decision: accepted.

## text-unidecode 1.3

- Development only: pytest-playwright 0.10.0 requires python-slugify 9.1.3, which
  requires text-unidecode 1.3. The browser tests behind `make check-site` are its only
  use. Not in the wheel or the image, and absent from the `--no-dev` audit.
- Metadata: `License: Artistic License`, with classifiers for the Artistic License,
  the GPL and GPLv2 or later.
- Terms at 1.3: `LICENSE.txt` in the wheel, byte-identical to `LICENSE` at the upstream
  `1.3` tag, lets a recipient redistribute or modify the package under either "GPL or
  GPLv2+" or the Artistic License, whose full text follows: the Perl Artistic License
  1.0, including clause 8 on aggregation with a commercial distribution. The choice is
  the recipient's. Artistic-1.0 is not copyleft and does not restrict commercial use;
  the GPL options are copyleft.
- Source: https://github.com/kmike/text-unidecode/blob/1.3/LICENSE.
- Decision: accepted for development use only. Nothing promcast distributes contains
  it; if it ever became a runtime dependency, this decision would need revisiting
  before release.

## numpy 2.5.4

- Runtime: a direct dependency. Shipped in the image.
- Metadata: `License-Expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`,
  backed by 17 licence files for vendored sources (pocketfft, highway, libdivide,
  x86-simd-sort, SVML, the random generators and others). The check fails only on
  Zlib and CC0-1.0, both permissive.
- Bundled on Linux: `numpy.libs/` in both manylinux wheels holds
  `libscipy_openblas64_` (OpenBLAS, BSD-3-Clause, with LAPACK under
  BSD-3-Clause-Open-MPI) and `libgfortran.so.5`; the x86_64 wheel also holds
  `libquadmath.so.0`. The appendix to the wheel's `LICENSE.txt` declares libgfortran
  `GPL-3.0-or-later WITH GCC-exception-3.1` and carries the full texts of the GPL
  version 3 and of the GCC Runtime Library Exception version 3.1. It declares
  libquadmath LGPL-2.1-or-later and quotes its notice, but not the LGPL text. The
  macOS arm64 wheel bundles no library.
- Sources: https://github.com/numpy/numpy/blob/v2.5.4/LICENSE.txt,
  https://github.com/numpy/numpy-release/blob/main/tools/wheels/LICENSE_linux.txt.
- Decision: accepted; see "GCC runtime libraries" below.

## torch 2.15.0

- Runtime: a direct dependency, from the PyTorch CPU index on Linux, where it installs
  as torch 2.15.0+cpu; macOS installs torch 2.15.0 from PyPI. Shipped in the image.
- Metadata: `License-Expression: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND
  BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib`, with `LICENSE`
  and 60 licence files under `dist-info/licenses/third_party`, all permissive. The GPL
  appears in them only in the LLVM exception's GPLv2 compatibility clause (NVTX,
  llvm-openmp).
- Bundled on Linux: `torch/lib/` in both `+cpu` manylinux wheels holds `libgomp.so.1`
  and `libgfortran.so.5`. The aarch64 wheel also holds OpenBLAS 0.3.34
  (`libopenblas.so.0`, BSD-3-Clause) and Arm Compute Library v53.2.0
  (`libarm_compute.so`, `libarm_compute_graph.so`), whose repository licenses its code
  under MIT, Apache-2.0 and BSD-3-Clause. Each version is the string the library
  embeds. No licence file in the wheel covers any of these five libraries.
- Source: https://github.com/pytorch/pytorch/blob/v2.15.0/LICENSE.
- Decision: accepted; see "GCC runtime libraries" below.

## GCC runtime libraries

Read on 2026-11-09 from the four Linux wheels locked in `uv.lock` and from GCC's source.

| Library | Wheels | GCC version evidence | Licence |
|---|---|---|---|
| `libgfortran.so.5` | numpy and torch, x86_64 and aarch64 | symbol version `GFORTRAN_8` | GPL-3.0-or-later WITH GCC-exception-3.1 |
| `libgomp.so.1` | torch, x86_64 and aarch64 | aarch64 build tag `GCC: (GNU) 13.3.0`, symbol versions to `GOMP_5.1.1`; x86_64 to `GOMP_5.0` | GPL-3.0-or-later WITH GCC-exception-3.1 |
| `libquadmath.so.0` | numpy, x86_64 | symbol versions to `QUADMATH_1.1` | LGPL, version 2 or later in GCC's header, 2.1 or later in numpy's notice |

The licence column comes from the headers of `libgfortran/libgfortran.h` and
`libgomp/libgomp.h` at both the `releases/gcc-8.5.0` and `releases/gcc-13.3.0` tags,
which grant GPL version 3 or later plus "the GCC Runtime Library Exception, version
3.1", and from `libquadmath/quadmath.h` at `releases/gcc-13.3.0`. No bundled library
is under the GPL without that exception. The exception lets independent code that GCC
compiled and linked against these libraries be distributed under terms of its own
choosing; the libraries themselves stay under the GPL or LGPL.

The image redistributes these libraries only inside the unmodified wheels that
`uv sync --frozen` installs from the hashes in `uv.lock`, with each wheel's own
`dist-info` licence files. The image build fails if the numpy, torch or augurs licence
files are missing from the installed environment. Because torch carries no licence
text for its bundled libraries and numpy carries no LGPL text, the image also holds
GCC's `COPYING3` (GPL version 3), `COPYING.RUNTIME` (the exception, version 3.1) and
`COPYING.LIB` (LGPL version 2.1), and the OpenBLAS and Arm Compute Library licences, in
`/opt/promcast/licences/`. `licences/SOURCES` records each file's SHA-256 and the
commit URL it was fetched from, and `tests/audit/test_licence_texts.py` checks them.
GCC's source for these libraries is at https://gcc.gnu.org/git/gcc.git under the
release tags. Several bundled binaries carry Red Hat toolchain build tags (`8.5.0`,
`13.3.1`, `14.2.1`), so the exact patched GCC source behind each build was not
identified.

Sources:
https://www.gnu.org/licenses/gcc-exception-3.1.html,
https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/libgomp/libgomp.h,
https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/libgfortran/libgfortran.h,
https://github.com/gcc-mirror/gcc/blob/releases/gcc-8.5.0/libgfortran/libgfortran.h,
https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/libquadmath/quadmath.h.

Decision: accepted. GPL-3.0-or-later with the GCC Runtime Library Exception 3.1 permits
distributing promcast under Apache-2.0 beside these libraries, and LGPL is already an
allowed family.

## Consequences

`ACCEPTED` in `scripts/check_licences.py` names all six packages at the versions above
and points to this ADR. A new version of any of them fails the check until it is
reviewed again.

## Verification

- `uvx pip-licenses==5.5.5 --python <runtime venv> --format=json | uv run python
  scripts/check_licences.py` exits 0.
- `tests/audit/test_check_licences.py` checks that an accepted package passes only at
  its reviewed version and that every entry names an ADR recording it.
- `tests/audit/test_licence_texts.py` checks every file in `licences/` against its
  recorded SHA-256 and commit URL.
