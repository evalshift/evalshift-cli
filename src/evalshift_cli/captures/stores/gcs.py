"""Google Cloud Storage read adapter. Extra: ``evalshift[gcs]``.

Credentials come from Application Default Credentials.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from evalshift_cli.captures.remote import ObjectInfo
from evalshift_cli.captures.stores import aware_utc


class GCSStore:
    """List and read objects under ``gs://<bucket>/<prefix>``."""

    def __init__(self, bucket: str, prefix: str = "", *, client: Any | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.uri = f"gs://{bucket}/{self.prefix}".rstrip("/")
        self._client = client

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _relative(self, full_key: str) -> str:
        return full_key.removeprefix(f"{self.prefix}/") if self.prefix else full_key

    def _get_client(self) -> Any:
        if self._client is None:
            from google.cloud import storage  # lazy: the [gcs] extra is optional

            self._client = storage.Client()
        return self._client

    def list(self, prefix: str) -> Iterator[ObjectInfo]:
        """Yield every object whose key starts with ``prefix`` (relative to the store prefix)."""
        for blob in self._get_client().list_blobs(self.bucket, prefix=self._full(prefix)):
            yield ObjectInfo(
                key=self._relative(blob.name),
                last_modified=aware_utc(blob.updated),
                size=blob.size,
            )

    def get(self, key: str) -> bytes:
        """Return the bytes of ``key`` (relative to the store prefix)."""
        blob = self._get_client().bucket(self.bucket).blob(self._full(key))
        data: bytes = blob.download_as_bytes()
        return data


__all__ = ["GCSStore"]
