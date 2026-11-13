# Configuration

`promcast.example.yaml` is the reference configuration. `promcast validate FILE`
checks a full document and `promcast validate FILE --fragment` checks a
recommendation fragment with targets and an optional required `model`. Both return 2 on configuration errors, printing an
`error:` line and a `next:` line without echoing rejected values.

## Keys

A full document has `tsdb:` and `targets:`; `model:`, `server:` and `cycle:`
fall back to the values below. Every mapping forbids unknown fields.

| Key | Type | Default | Rules |
| --- | --- | --- | --- |
| `tsdb.url` | URL | required | HTTP or HTTPS Prometheus-compatible endpoint; no userinfo |
| `tsdb.timeout_s` | number | `30` | greater than zero |
| `tsdb.headers` | mapping | `{}` | values are redacted secrets |
| `tsdb.bearer_token_file` | path | none | exclusive with `basic_auth` |
| `tsdb.basic_auth.username` | string | required | redacted secret, required with `basic_auth` |
| `tsdb.basic_auth.password_file` | path | required | required with `basic_auth` |
| `tsdb.tls.ca_file` | path | none | custom trust bundle |
| `model.name` | string | `toto-2.0-22m` | `toto-2.0-22m` or `toto-2.0-4m` |
| `model.batch_size` | integer | `16` | greater than zero |
| `model.cache_dir` | path | `~/.cache/promcast/models` | `PROMCAST_CACHE_DIR` wins when set |
| `server.listen` | string | `0.0.0.0:8787` | host and port |
| `cycle` | duration | `2m` | greater than zero |
| `targets[].name` | string | required | starts with a letter or `_`; letters, digits, `_` and `-` |
| `targets[].known_gauge` | boolean | `false` | bypass counter detection only for exporter-documented gauges; recipes set it to `true` |
| `targets[].query` | string | required | PromQL |
| `targets[].step` | duration | required | greater than zero |
| `targets[].context` | duration | required | 64 to 2,048 steps |
| `targets[].horizon` | duration | required | at least one step, at most 512 |
| `targets[].season` | duration | required | any duration |
| `targets[].direction` | string | required | `above` or `below` |
| `targets[].threshold` | number | required | finite |
| `targets[].quantile` | number | required | `0.5` to `0.9` in steps of `0.1` |
| `targets[].warn` | duration | required | at most `horizon` |
| `targets[].method` | string | required | `toto` or `linear` |
| `targets[].linear_window` | duration | required | at most `context`, at least two steps |
| `targets[].max_series` | integer | required | 1 to 500 |

Durations are an integer followed by `s`, `m`, `h` or `d`.

## Process environment

| Variable | Read by | Effect |
| --- | --- | --- |
| `PROMCAST_CACHE_DIR` | `parse_config`, `scan`/`init` discovery, `bench` | overrides `model.cache_dir` and is the default for `bench --cache-dir`; a blank value is ignored |
| `PROMETHEUS_URL` | `scan`/`init` | endpoint used when no URL argument is given |
| `COLUMNS` | board rendering | output width |

`PROMCAST_CACHE_DIR` applies after parsing, so it wins over the YAML value and
over any `${...}` expansion, and the `scan` command caches weights there even
when discovery uses no configuration file. Like the YAML value, it keeps a literal
`~` until a filesystem consumer expands it. `bench` has no configuration model;
it uses `--cache-dir`, which defaults to `PROMCAST_CACHE_DIR`.

## API and transport boundary

`promcast.core.config` exports `Config`, `TargetConfig`, `TargetsFragment`,
`parse_config(text, environment=...)`, `parse_fragment(text, environment=...)`
and the compatibility entry point `load_config(path, environment=...)`.
New file-loading callers use `promcast.shell.config.load_config`; the old
entry point resolves lazily to the same function. The environment argument is
required and accepts a mapping of strings. The shell passes its environment explicitly. Parsing
performs no filesystem reads; the shell reads only the supplied UTF-8 YAML file.

Duration fields become `datetime.timedelta` values. Target duration ratios
must give 64 to 2,048 context steps and at most 512 horizon steps, endpoints
included, and require a horizon of at least one step. A fragment requires `targets:` and permits `model:` to preserve the selected
Toto model. Apply both mappings to the live configuration; tuned quantiles and
warning periods belong to that model. The mappings reuse `TargetConfig` and
`ModelConfig`. Unknown fields are forbidden
throughout the model tree. Validation failures raise `ConfigError` with paths
such as `targets.0.linear_window` and `tsdb.headers.Authorization`.

The TSDB URL is a Pydantic `HttpUrl`; convert it with `str()` for HTTP clients.
URL userinfo is rejected at `tsdb.url`; credentials belong in `basic_auth`.
Authentication and CA filenames remain `Path` references. Validation checks
authentication exclusivity without reading credentials, checking file existence
or contacting a server. The HTTP shell must read the referenced credentials.
The model cache path retains its literal `~`; filesystem consumers must expand
the user's home directory when resolving it.

## Environment and secret behavior

Expansion happens after YAML parsing, once per configured scalar, using only
the supplied environment. Replacement strings cannot inject YAML and are not
expanded recursively. Missing variables report the variable name and field
path. Empty environment values are valid substitutions.

Every header value and the basic-auth username use `RedactedSecret`, a
`SecretStr` subclass. `str`, `repr`, Python model dumps and JSON model dumps
hide these values with `<redacted>`, including empty strings. The HTTP shell
can explicitly retrieve a value with `get_secret_value()`. Secret file paths
are references, not loaded credentials, and remain visible in diagnostics.
Use `model_dump(mode="json")` for future doctor output.

Pydantic's default `SecretStr` JSON serialization uses its own mask, so the
secret annotation has an explicit JSON serializer for the required marker.
Regression tests assert the marker and absence of the actual values. YAML
parser exceptions can quote source lines containing credentials; `ConfigError`
reports line and column without source snippets or an unsafe exception chain.
Validation diagnostics likewise omit rejected input values.

## Authenticated discovery

`promcast scan --config promcast.yaml` and `promcast init --config promcast.yaml`
read the same TSDB transport settings as `serve`. Discovery uses recipes for
targets, while retaining the configured model and cache settings. Use `targets: []`
when the file is only for discovery. A positional URL with `--config` is rejected.

```yaml
tsdb:
  url: https://metrics.example.com
  bearer_token_file: /path/to/token
  headers:
    X-Scope-OrgID: ${METRICS_TENANT}
  tls:
    ca_file: /path/to/ca.pem
targets: []
```

Use `tsdb.basic_auth.username` and `tsdb.basic_auth.password_file` instead of
`bearer_token_file` for basic authentication. Headers can reference environment
variables. Credential files and the CA bundle must be readable by the process;
`init` preserves these references, so mount the files into its container too.
On HTTP authentication rejection, correct the settings in the file and rerun
`promcast scan --config promcast.yaml`.
