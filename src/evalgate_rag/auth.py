"""Keyless LLM authentication with an Azure managed identity.

Azure OpenAI accepts a Microsoft Entra ID access token as the bearer token in
place of an API key. Inside Azure Container Apps (and App Service) the platform
hands such a token to the workload through a local HTTP endpoint, so there is
no secret to store, rotate, or leak into Terraform state.

This talks to that endpoint directly over httpx instead of pulling in the
azure-identity SDK: it is one GET, and the project keeps a single
provider-neutral HTTP client (see CLAUDE.md, "One LLM client").
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import httpx

# Refresh this long before expiry, so a token is never sent with seconds left
# and rejected mid-flight on a slow request.
REFRESH_MARGIN_S = 300.0
_API_VERSION = "2019-08-01"


class ManagedIdentityToken:
    """Callable returning a current access token, cached until near expiry.

    The endpoint and its header value come from `IDENTITY_ENDPOINT` and
    `IDENTITY_HEADER`, which the platform injects. `client_id` selects a
    user-assigned identity and defaults to `AZURE_CLIENT_ID`.
    """

    def __init__(
        self,
        resource: str,
        client_id: str | None = None,
        endpoint: str | None = None,
        header: str | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._resource = resource
        self._client_id = client_id or os.environ.get("AZURE_CLIENT_ID")
        self._endpoint = endpoint or os.environ.get("IDENTITY_ENDPOINT")
        self._header = header or os.environ.get("IDENTITY_HEADER")
        if not self._endpoint or not self._header:
            raise RuntimeError(
                "Managed identity is not available: IDENTITY_ENDPOINT and IDENTITY_HEADER are "
                "not set. They exist only inside Azure Container Apps or App Service with an "
                "identity assigned; use LLM__AUTH=api_key everywhere else."
            )
        self._client = httpx.Client(timeout=10.0, transport=transport)
        self._clock = clock
        # FastAPI runs /query in a threadpool, so concurrent requests can find
        # the token expired at the same moment; one of them refreshes it.
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    def __call__(self) -> str:
        with self._lock:
            if self._token is None or self._clock() >= self._expires_at - REFRESH_MARGIN_S:
                self._token, self._expires_at = self._fetch()
            return self._token

    def _fetch(self) -> tuple[str, float]:
        params = {"resource": self._resource, "api-version": _API_VERSION}
        if self._client_id:
            params["client_id"] = self._client_id
        resp = self._client.get(
            str(self._endpoint), params=params, headers={"X-IDENTITY-HEADER": str(self._header)}
        )
        resp.raise_for_status()
        body = resp.json()
        return body["access_token"], float(body["expires_on"])
