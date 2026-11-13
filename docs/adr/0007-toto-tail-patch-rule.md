# 0007: Keep padding left of Toto's forced-observed final patch

Status: Accepted for the pinned models on CPU.

Date: 2026-11-05.

## Context

This ADR establishes the model contract before the forecaster and series code
depend on it. The installed packages are `toto-models==1.0.0`, `toto-2==2.0.0`,
`torch==2.15.0` and `numpy==2.5.4`. Tests ran with Python 3.12.16 on Darwin
27.2.0 arm64, CPU, float32 inputs and one torch intra-operation thread. The
fixture restores the original thread count. MPS and Linux execution are
unverified.

Both model files were read from `~/.cache/promcast/models/` and their SHA-256
digests checked against the pinned values before loading:

| Model | Revision | Weights SHA-256 |
| --- | --- | --- |
| toto-2.0-22m | `685e4ae3e2be8d8998025e53dd98e7fdcb296a89` | `9cd503d82df3aa71747862688f47a31c1d0a4b80f898df6e046189016eaa21dd` |
| toto-2.0-4m | `8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9` | `316660d5afb47943e531f39242e0b02ca0b8bb73be5709dfe07ca80dfce9805e` |

The cache contains weights without `config.json`. During this spike the
configuration was read from the exact
[22m revision](https://huggingface.co/Datadog/Toto-2.0-22m/resolve/685e4ae3e2be8d8998025e53dd98e7fdcb296a89/config.json)
and
[4m revision](https://huggingface.co/Datadog/Toto-2.0-4m/resolve/8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9/config.json).
The tests retain those configuration fields, write them into a temporary
directory, symlink the verified weights, and call
`Toto2Model.from_pretrained(local_dir, local_files_only=True, map_location="cpu")`.
The tests make no network request and do not modify the model cache. The local
loader returns the model in evaluation mode. A later runtime loader still needs
to supply configuration files; this spike does not change model acquisition.

| Model | Downloaded config.json SHA-256 | Canonical configuration SHA-256 |
| --- | --- | --- |
| toto-2.0-22m | `abeaf0fcd54aaac66757fde69ec3ddb4d3bfdcf96e0c8f767aefd03ab4c9e8d9` | `0d36fd02ef0df8005aa5511d63b2dbb8e1cacd7be4657ebef97db00761e35945` |
| toto-2.0-4m | `7a926d130e401ab0c5fdb3564f46c8d917bd05c7b3ae26b9c22d2da2ef01d2d8` | `cfce07f41b88938f96c1f664c5ed85f611a9f3d42ddd02e7095062e3fdd32969` |

Canonical configuration means SHA-256 of UTF-8 `json.dumps(config, sort_keys=True)`.
Both configurations staged by the tests were separately hashed and matched
the canonical digests of the downloaded revision files.
Context7 returned no matching Toto 2 API documentation. API details below were
checked against the installed
[toto-2 2.0.0 package](https://pypi.org/project/toto-2/2.0.0/) source.

## Measured contract

The test context is `10 + sin(t / 5) + t / 100`, with integer `t` starting at
zero, unless a case specifies constants or masked slots. Forecast inputs are a
dictionary with float32 `target` shaped `(B, 1, context)`, boolean `target_mask`
of the same shape, and int64 `series_ids` shaped `(B, 1)`, all zero. Every
forecast uses `decode_block_size=None` and horizon 37. Both configurations have
patch size 32 and one output patch.

| Check | toto-2.0-22m | toto-2.0-4m |
| --- | --- | --- |
| Shape, batches 1 and 3, context 128 | `(9, B, 1, 37)` | `(9, B, 1, 37)` |
| Quantile levels | 0.1 through 0.9, increments 0.1 | 0.1 through 0.9, increments 0.1 |
| Quantiles sorted on axis 0 | Yes, finite | Yes, finite |
| Same input twice | Bit-identical | Bit-identical |
| Constant contexts, values 0 and 7 | Finite, sorted | Finite, sorted |
| Last five zeros, mask False versus True: maximum absolute difference | 0.0 | 0.0 |
| First 100 of 128 slots masked: default versus scaler options, maximum absolute difference | 0.0 | 0.0 |
| Four observations in leading patch: default versus fallback alone, maximum absolute difference | 0.4284200668334961 | 1.6705760955810547 |

The scaler-options case sets `scaler_fallback_min_obs=8` and
`quantile_real_cap_k=1e4`. Both the default and protected forecasts are finite
and sorted. The final patch is forced observed, so this context's scaler sees
four zero slots plus the 28 supplied observations. Its completely masked
leading patches explain why fallback need not change this forecast.

An additional context observes slots 0 through 3 and 100 through 127, with the
other slots masked and zeroed. Fallback alone changes its output by the amount
above. This checks that the option affects a partially observed leading patch.

The real-space cap is checked against observed context bounds expanded by
`K * anchor_scale`, including the forced-observed final patch. With `K=0.01`,
both models actually clip output. Each result is bit-identical to explicitly
clamping the fallback-only forecast to those bounds; it remains finite and
sorted. The production setting `K=1e4` is also checked against its bounds.

For the short-history test, 64 slots contain the observed sine context at the
right and zero padding at the left. The mask marks only observations True.
Every masked slot is then replaced with each of `-1000000`, `17` and `1000000`.
Both calls use fallback 8 and cap 1e4. All outputs remain finite and sorted.

| Observed samples | Masked left slots | Maximum absolute difference, 22m | Maximum absolute difference, 4m |
| --- | --- | --- | --- |
| 32 | 32 | 0.0 | 0.0 |
| 33 | 31 | 0.0 | 0.0 |
| 40 | 24 | 0.0 | 0.0 |
| 63 | 1 | 0.0 | 0.0 |

Each table entry is the maximum over the three replacement values. All four
counts pass the strict difference `< 1e-6` rule for both model files. These
measurements cover finite padding sentinels; they do not establish behavior
for NaN or infinite values in masked slots.

## Decision and consequences

Keep the left padding to a multiple of 32, at least 64 slots, with
at least 32 real observations. Padding must be masked and remain entirely
outside the last 32-slot patch. Use fallback 8 and real-space cap 1e4. Apply
the tail rule before forecasting: the mask cannot hide missing values in the
final patch. The short-history horizon restriction still applies;
the 37-step spike horizon tests output behavior, not permission to extend it.

The source reason is `toto2/model.py`,
`Toto2Model._prepare_forecast_inputs` (lines 1092 to 1150 in the installed
2.0.0 package). It retains the context mask before the last patch, then
concatenates an all-True final context patch and an all-False prediction
region. `Toto2Model.forecast` (lines 1153 to 1336) sorts real-space quantiles
on axis 0, clamps non-finite predictions, and slices the output to `horizon`.
`backfill_short_patches` supplies statistics from the first observed points
to leading patches below the observation minimum. The return shape is already
truncated; callers should not assume a padded forecast horizon.

The no-padding fallback is not activated by these results. If any of the four
observed counts fails the `< 1e-6` comparison in a later supported environment
or pinned-model change, record its measured maximum here and replace
`left_pad_to_patch` everywhere with `left_trim_to_patch`: discard the oldest
slots down to a multiple of 32 and require at least 64 real observations.
Retain the finite and sorted checks. Never keep zero padding after a failed
comparison, relax the tolerance, or hide the failure with a skip. This spike
adds no core implementation.

## Verification

```sh
uv run pytest -m slow tests/core/test_toto_contract.py
make check
```

Both commands exited 0. The slow command ended with:

```text
38 passed, 1 warning in 11.59s
```

`make check` passed formatting, lint, strict typing and the existing scaffold
test, with the slow cases deselected:

```text
16 files already formatted
All checks passed!
0 errors, 0 warnings, 0 informations
1 passed, 38 deselected in 0.41s
```

The optional GluonTS JSON accelerator warning is upstream; no dependency was
added to suppress it.

To capture the measured differences as JUnit properties:

```sh
uv run pytest -m slow tests/core/test_toto_contract.py \
  --junitxml=/tmp/promcast-toto-contract.xml -o junit_family=legacy
```

In the restricted workspace, set `UV_CACHE_DIR="$PWD/.cache/uv"` and
`UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"` before invoking uv or make.
The model fixture also redirects incidental matplotlib and font caches to
temporary storage during the library import.
