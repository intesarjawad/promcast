import pytest

from promcast.core.diagnostics import safe_urls
from promcast.shell.tsdb import TsdbError


@pytest.mark.parametrize(
    "url",
    [
        "https://alice:password@host/path?token=secret&view=cpu",
        "https://alice:password@[::1]:9090/path?access%5Ftoken=secret&view=cpu",
        "https://alice:password@host/path?X-Amz-Signature=secret&view=cpu#secret",
    ],
)
def test_url_credentials_are_removed_without_losing_diagnostic_context(url: str):
    safe = safe_urls(f"failed {url}")
    for secret in ("alice", "password", "secret"):
        assert secret not in safe
    assert "/path" in safe
    assert "view=cpu" in safe


def test_transport_error_sanitizes_every_url():
    error = TsdbError(
        500, "https://a:password@one/?token=secret https://b:password@two/?api_key=secret"
    )
    assert "password" not in str(error)
    assert "secret" not in error.body


@pytest.mark.parametrize("parameter", ["bearer", "session_id", "code", "ticket"])
def test_token_like_query_parameters_are_not_printed(parameter: str):
    assert "secret" not in safe_urls(f"https://host/?{parameter}=secret&view=cpu")
