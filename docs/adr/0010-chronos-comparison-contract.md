# 0010: Retain Chronos-2 small as a replay-only comparison

Status: Accepted for the pinned model on CPU.

Date: 2026-11-05.

## Context

The comparison-model smoke test establishes its quantile contract and audits the two
training datasets named on its
[model card](https://huggingface.co/autogluon/chronos-2-small/blob/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a/README.md).
`uv sync --extra compare` resolved and installed alongside the existing pins.
No dependency or lockfile change was required. Installed versions are
`chronos-forecasting==2.3.2`, `torch==2.15.0`, `numpy==2.5.4`,
`transformers==5.22.1`, `accelerate==1.15.0` and `pandas==2.3.3`.
The tests ran with Python 3.12.16 on Darwin 27.2.0 arm64, CPU, float32 inputs
and one torch intra-operation thread. The fixture restores the previous
thread count. MPS and Linux inference remain unverified.

Weights were read from `~/.cache/promcast/models/chronos-2-small/model.safetensors`
and SHA-256 checked before loading:

| Model | Revision | Weights SHA-256 |
| --- | --- | --- |
| chronos-2-small | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` | `492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd` |

The cache contains weights without `config.json`. The spike read the
[configuration at that revision](https://huggingface.co/autogluon/chronos-2-small/resolve/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a/config.json).
Its downloaded SHA-256 is
`f2780468adc8b16322aa0b6e28b26476c9401ecf64ab53292b6d2b0d3f423722`.
The canonical SHA-256 of UTF-8 `json.dumps(config, sort_keys=True)` is
`885ee12976bd37d5f853027102a7f5b893675cd22725e2455c3cba7e9f0e4d40`.
The test retains all configuration fields, asserts that canonical digest,
writes a temporary configuration, and symlinks the verified weights there.
It calls `chronos.BaseChronosPipeline.from_pretrained(local_dir,
device_map="cpu", local_files_only=True)` with `HF_HUB_OFFLINE=1` and receives
`Chronos2Pipeline` with its model in evaluation mode. The tests make no
network request and do not modify the model cache. The runtime loader will
still need to supply the configuration; model acquisition is outside this spike.

## Measured contract

Context7 documentation was checked before coding. The authoritative signature
below was read from the installed
[chronos-forecasting 2.3.2 package](https://pypi.org/project/chronos-forecasting/2.3.2/),
`chronos/chronos2/pipeline.py`, lines 763 through 820:

```python
def predict_quantiles(
    self,
    inputs: TensorOrArray
    | Sequence[TensorOrArray]
    | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]]
    | Sequence[PreparedInput],
    prediction_length: int | None = None,
    quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    **predict_kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
```

`TensorOrArray` denotes a torch tensor or NumPy array. The tested input is a
CPU float32 tensor shaped `(B, 1, 512)`, with `sin(t / 5)` for integer `t`
starting at zero. Batched series have offsets `0`, `1` and `2` to distinguish
them. Both calls use horizon 37 and explicit quantile levels 0.1 through 0.9,
under `torch.inference_mode()`. This crosses the 16-step output patch boundary.
No sampling parameter or seed is needed for this deterministic quantile head.

| Check | Batch 1 | Batch 3 |
| --- | --- | --- |
| Quantile return | List of 1 tensor `(1, 37, 9)` | List of 3 tensors `(1, 37, 9)` |
| Point return | List of 1 tensor `(1, 37)` | List of 3 tensors `(1, 37)` |
| Output device and dtype | CPU, float32 | CPU, float32 |
| Raw quantile and point values finite | Yes | Yes |
| Quantile and point results across two calls | Bit-identical | Bit-identical |
| Finite and monotone after `np.maximum.accumulate(..., axis=-1)` | Yes | Yes |
| Maximum correction from that guard | 0.0 | 0.0 |

Quantiles are on the last axis, in the requested level order. The configured
training levels also include 0.01, 0.05, 0.95 and 0.99. The requested nine
levels are selected directly without interpolation. The second return value
is named `mean` upstream but is actually the raw 0.5 quantile; the test checks
its equality to that column. It is not an arithmetic expectation. These
measurements establish neither calibration nor comparative forecasting quality.

## Training-data name audit

Read on 2026-11-05, using public metadata without downloading series. The
model card names `autogluon/chronos_datasets` and `Salesforce/GiftEvalPretrain`.
The following inventories are complete at their recorded revisions:

| Inventory | Revision | Entries |
| --- | --- | --- |
| Chronos configuration names | `eeecad0b82a8c237e212ce6f8d1abecb513e2cec` | 67 |
| GiftEvalPretrain dataset directories | `6830b624de7ed2b3d3e5b85bb6959d81dcc5d874` | 152 |

Chronos configuration names were taken from
[the pinned dataset metadata](https://huggingface.co/api/datasets/autogluon/chronos_datasets/revision/eeecad0b82a8c237e212ce6f8d1abecb513e2cec),
`cardData.configs[*].config_name`. They match the complete configuration set
returned by the [dataset splits endpoint](https://datasets-server.huggingface.co/splits?dataset=autogluon/chronos_datasets)
at the time of the audit:

```text
dominick, electricity_15min, ercot, exchange_rate, m4_daily, m4_hourly, m4_monthly,
m4_quarterly, m4_weekly, m4_yearly, m5, mexico_city_bikes, monash_australian_electricity,
monash_car_parts, monash_cif_2016, monash_covid_deaths, monash_electricity_hourly,
monash_electricity_weekly, monash_fred_md, monash_hospital, monash_kdd_cup_2018,
monash_london_smart_meters, monash_m1_monthly, monash_m1_quarterly, monash_m1_yearly,
monash_m3_monthly, monash_m3_quarterly, monash_m3_yearly, monash_nn5_weekly,
monash_pedestrian_counts, monash_rideshare, monash_saugeenday, monash_temperature_rain,
monash_tourism_monthly, monash_tourism_quarterly, monash_tourism_yearly, monash_traffic,
monash_weather, nn5, solar, solar_1h, taxi_1h, taxi_30min, training_corpus_kernel_synth_1m,
training_corpus_tsmixup_10m, uber_tlc_daily, uber_tlc_hourly, ushcn_daily, weatherbench_daily,
weatherbench_hourly_10m_u_component_of_wind, weatherbench_hourly_10m_v_component_of_wind,
weatherbench_hourly_2m_temperature, weatherbench_hourly_geopotential,
weatherbench_hourly_potential_vorticity, weatherbench_hourly_relative_humidity,
weatherbench_hourly_specific_humidity, weatherbench_hourly_temperature,
weatherbench_hourly_toa_incident_solar_radiation, weatherbench_hourly_total_cloud_cover,
weatherbench_hourly_total_precipitation, weatherbench_hourly_u_component_of_wind,
weatherbench_hourly_v_component_of_wind, weatherbench_hourly_vorticity, weatherbench_weekly,
wiki_daily_100k, wind_farms_daily, wind_farms_hourly
```

GiftEvalPretrain names were read from
[the pinned root tree](https://huggingface.co/api/datasets/Salesforce/GiftEvalPretrain/tree/6830b624de7ed2b3d3e5b85bb6959d81dcc5d874?recursive=false&limit=1000),
filtering `type == "directory"`. There was no next page. The names match all
unique directory prefixes in the pinned repository's file inventory:

```text
BEIJING_SUBWAY_30MIN, HZMETRO, LOS_LOOP, PEMS03, PEMS04, PEMS07, PEMS08, PEMS_BAY, Q-TRAFFIC,
SHMETRO, alibaba_cluster_trace_2018, australian_electricity_demand, azure_vm_traces_2017,
bdg-2_bear, bdg-2_fox, bdg-2_panther, bdg-2_rat, beijing_air_quality, bitcoin_with_missing,
borealis, borg_cluster_data_2011, buildings_900k, bull, cdc_fluview_ilinet,
cdc_fluview_who_nrevss, china_air_quality, cif_2016_12, cif_2016_6, cmip6_1850, cmip6_1855,
cmip6_1860, cmip6_1865, cmip6_1870, cmip6_1875, cmip6_1880, cmip6_1885, cmip6_1890, cmip6_1895,
cmip6_1900, cmip6_1905, cmip6_1910, cmip6_1915, cmip6_1920, cmip6_1925, cmip6_1930, cmip6_1935,
cmip6_1940, cmip6_1945, cmip6_1950, cmip6_1955, cmip6_1960, cmip6_1965, cmip6_1970, cmip6_1975,
cmip6_1980, cmip6_1985, cmip6_1990, cmip6_1995, cmip6_2000, cmip6_2005, cmip6_2010, cockatoo,
covid19_energy, covid_mobility, elecdemand, elf, era5_1989, era5_1990, era5_1991, era5_1992,
era5_1993, era5_1994, era5_1995, era5_1996, era5_1997, era5_1998, era5_1999, era5_2000,
era5_2001, era5_2002, era5_2003, era5_2004, era5_2005, era5_2006, era5_2007, era5_2008,
era5_2009, era5_2010, era5_2011, era5_2012, era5_2013, era5_2014, era5_2015, era5_2016,
era5_2017, era5_2018, extended_web_traffic_with_missing, favorita_sales, favorita_transactions,
fred_md, gfc12_load, gfc14_load, gfc17_load, godaddy, hog, ideal, kaggle_web_traffic_weekly,
kdd2022, largest_2017, largest_2018, largest_2019, largest_2020, largest_2021, lcl,
london_smart_meters_with_missing, m1_monthly, m1_quarterly, m1_yearly, m5, monash_m3_monthly,
monash_m3_other, monash_m3_quarterly, monash_m3_yearly, nn5_daily_with_missing, nn5_weekly,
oikolab_weather, pdb, pedestrian_counts, project_tycho, residential_load_power,
residential_pv_power, rideshare_with_missing, sceaux, smart, solar_power, spain, subseasonal,
subseasonal_precip, sunspot_with_missing, taxi_30min, tourism_monthly, tourism_quarterly,
tourism_yearly, traffic_hourly, traffic_weekly, uber_tlc_daily, uber_tlc_hourly,
vehicle_trips_with_missing, weather, wiki-rolling_nips, wind_farms_with_missing, wind_power
```

Neither inventory contains NAB, Numenta or BOOM, including a case-insensitive
substring check. No NAB or BOOM contamination hit was found, so no hit-specific
Chronos caveat is required for E2 or E3 on this audit. This is a dataset-name
check, not proof of unseen series: aliases, related sources, synthetic mixing
and overlap inside listed datasets were not traced. In particular, named
infrastructure trace datasets do not by themselves establish BOOM overlap.
Any later provenance evidence of overlap must add a caveat to the affected
corpus's Chronos rows.

## Decision and consequences

Retain `chronos-2-small` in the planned replay comparison behind the `compare`
extra. It remains outside the serving image and default forecaster. The repository now
implements `core.forecaster_chronos.Chronos2Forecaster`: it groups contexts by
length, chunks them at the configured batch size, preserves NaN missingness,
requests the nine shared levels explicitly, and transposes each list element
from `(1, horizon, 9)` to `(9, horizon)`. The monotonicity guard runs only on
finite results so `run_cycle` can reject invalid model output.

`shell.cli.load_forecaster` imports the adapter only when `chronos-2-small`
is requested. A missing `chronos` package raises `ConfigError` with
"install the compare extra", using the CLI's exit-1 error contract. Other
missing dependencies propagate rather than being misreported as a missing
extra. The current command surface is `validate`; subsequent replay command
handlers must use this selection entry point. The adapter is not deferred.
Loading requires both local `config.json` and `model.safetensors` and sets
`local_files_only=True`. Acquisition must still supply the pinned configuration.
The fallback is not activated: resolution and the contract both passed.

## Verification

In the restricted workspace, set `UV_CACHE_DIR="$PWD/.cache/uv"` and
`UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"` before invoking uv or make.

```sh
uv sync --extra compare
uv run --extra compare pytest -m slow tests/core/test_chronos_contract.py
make check
make check-slow
```

All commands exited 0. The targeted contract command ended with:

```text
2 passed in 2.85s
```

`make check` ended with:

```text
17 files already formatted
All checks passed!
0 errors, 0 warnings, 0 informations
1 passed, 40 deselected in 0.45s
```

`make check-slow` ended with:

```text
40 passed, 1 deselected, 1 warning in 13.00s
```

The warning is Toto's existing upstream GluonTS JSON accelerator warning.
The single deselected test is the fast scaffold case; no slow test was skipped.

To capture the measured correction as JUnit properties:

```sh
uv run --extra compare pytest -m slow tests/core/test_chronos_contract.py \
  --junitxml=/tmp/promcast-chronos-contract.xml -o junit_family=legacy
```
