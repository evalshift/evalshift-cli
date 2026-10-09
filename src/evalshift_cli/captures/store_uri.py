"""Store URI grammar, shared verbatim with the SDK (``evalshift.stores.uri``).

Three forms are accepted::

    s3://<bucket>/<prefix>
    gs://<bucket>/<prefix>
    az://<account>/<container>/<prefix>

``<prefix>`` is optional and may be empty; a trailing slash is stripped. Credentials never
belong in a URI: any ``@`` or ``?`` is rejected so nobody is tempted to embed a key or a
query parameter, and every adapter uses its provider's default credential chain instead.

This module deliberately imports nothing from the project so :mod:`evalshift_cli.config.models`
can validate ``captures.store`` with it without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import urlsplit

Scheme = Literal["s3", "gs", "az"]

#: The three accepted forms, quoted in every grammar error.
STORE_URI_FORMS = (
    "s3://<bucket>/<prefix>, gs://<bucket>/<prefix>, az://<account>/<container>/<prefix>"
)

_EXTRA_FOR_SCHEME: dict[str, str] = {"s3": "s3", "gs": "gcs", "az": "azure"}

#: The pip distribution that provides each client module. Shared verbatim with the SDK.
_PACKAGE_FOR_MODULE: dict[str, str] = {
    "boto3": "boto3",
    "google.cloud.storage": "google-cloud-storage",
    "azure.storage.blob": "azure-storage-blob",
    "azure.identity": "azure-identity",
}

#: The ``pip install`` argument that installs everything a scheme needs, in one command.
#: Named in every message instead of the extras: a user who installed the CLI the normal way
#: has no reason to know what an extra is. Shared verbatim with the SDK.
_PACKAGES_FOR_SCHEME: dict[str, str] = {
    "s3": "boto3",
    "gs": "google-cloud-storage",
    "az": "azure-storage-blob azure-identity",
}


def package_for_module(module: str) -> str:
    """The pip distribution that provides ``module`` (one of the client modules above)."""
    return _PACKAGE_FOR_MODULE[module]


@dataclass(frozen=True, slots=True)
class StoreURI:
    """A parsed ``captures.store`` URI.

    Attributes:
        scheme: ``"s3"``, ``"gs"`` or ``"az"``.
        bucket: The S3 / GCS bucket, or the Azure storage *account*.
        container: The Azure blob container; ``None`` for S3 and GCS.
        prefix: Key prefix under which ``captures/`` and ``toolsets/`` live. May be ``""``.
    """

    scheme: Scheme
    bucket: str
    container: str | None
    prefix: str

    @property
    def extra(self) -> str:
        """The pip extra (``evalshift[<extra>]``) that installs this scheme's client."""
        return _EXTRA_FOR_SCHEME[self.scheme]

    @property
    def packages(self) -> str:
        """The ``pip install`` argument that installs this scheme's client library (or libraries)."""
        return _PACKAGES_FOR_SCHEME[self.scheme]


def parse_store_uri(uri: str) -> StoreURI:
    """Parse ``uri`` against the grammar above.

    Raises:
        ValueError: for an unknown scheme, a missing bucket or container, or a URI carrying
            credentials (``@``) or query parameters (``?``). The message names the accepted
            forms so a config error is self-explanatory, and never echoes the value: a rejected
            value may be a key, a signature or a pasted connection string.
    """
    text = uri.strip()
    if "@" in text:
        raise ValueError(
            "credentials are not accepted in a store URI; use the provider's credential chain"
        )
    if "?" in text:
        # Never echo the URI: a query string is where SAS tokens and signatures live.
        raise ValueError(
            "query parameters are not accepted in a store URI; use the provider's credential chain"
        )
    # No message below echoes the value either: a rejected value may be a pasted connection
    # string or key. Only the parsed scheme, which cannot hold one, is quoted back.
    parts = urlsplit(text)
    if not parts.scheme:
        raise ValueError(f"store URI has no scheme; accepted forms: {STORE_URI_FORMS}")
    if parts.scheme not in _EXTRA_FOR_SCHEME:
        raise ValueError(
            f"unsupported store URI scheme {parts.scheme!r}; accepted forms: {STORE_URI_FORMS}"
        )
    if not parts.netloc:
        raise ValueError(f"store URI names no bucket; accepted forms: {STORE_URI_FORMS}")
    scheme = cast(Scheme, parts.scheme)
    path = parts.path.strip("/")
    if scheme == "az":
        container, _, prefix = path.partition("/")
        if not container:
            raise ValueError(
                "Azure store URI names no container; expected az://<account>/<container>/<prefix>"
            )
        return StoreURI(scheme, parts.netloc, container, prefix.strip("/"))
    return StoreURI(scheme, parts.netloc, None, path)


__all__ = ["STORE_URI_FORMS", "Scheme", "StoreURI", "package_for_module", "parse_store_uri"]
