"""Remove URL credentials before diagnostics cross an output boundary."""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_SECRET_PARAMETER = re.compile(
    r"token|key|secret|pass|auth|bearer|session|signature|credential|jwt|^(sig|code|ticket)$", re.I
)


def safe_urls(text: str) -> str:
    """Strip URL userinfo and sensitive query values, preserving useful endpoint details."""

    def replace(match: re.Match[str]) -> str:
        try:
            url = urlsplit(match.group())
            query = urlencode(
                [
                    (key, value)
                    for key, value in parse_qsl(url.query, keep_blank_values=True)
                    if not _SECRET_PARAMETER.search(key)
                ]
            )
            return urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, query, ""))
        except ValueError:
            return "<redacted-url>"

    return _URL.sub(replace, text)
