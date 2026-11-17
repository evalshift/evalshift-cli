"""Azure Blob Storage read adapter. Extra: ``evalshift[azure]``.

The account URL is ``https://<account>.blob.core.windows.net``; credentials come from
``DefaultAzureCredential``.

The Azure client resolves ``.`` and ``..`` segments in the request URL, even across containers,
so a container, prefix or key with an empty, ``.`` or ``..`` segment raises ``ValueError``
(:func:`~evalshift_cli.captures.store_uri.has_unsafe_segment`): a read never reaches outside
the prefix. The same rule as the SDK's ``AzureBlobStore``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from evalshift_cli.captures.remote import ObjectInfo
from evalshift_cli.captures.store_uri import has_unsafe_segment
from evalshift_cli.captures.stores import aware_utc


class AzureBlobStore:
    """List and read blobs under ``az://<account>/<container>/<prefix>``."""

    def __init__(
        self, account: str, container: str, prefix: str = "", *, client: Any | None = None
    ) -> None:
        self.account = account
        self.container = container
        self.prefix = prefix.strip("/")
        if has_unsafe_segment(container) or (self.prefix and has_unsafe_segment(self.prefix)):
            raise ValueError(
                "Azure container and prefix may not have an empty, '.' or '..' segment: the "
                "Azure client would resolve it to another place, even another container"
            )
        self.account_url = f"https://{account}.blob.core.windows.net"
        self.uri = f"az://{account}/{container}/{self.prefix}".rstrip("/")
        self._client = client

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _relative(self, full_key: str) -> str:
        return full_key.removeprefix(f"{self.prefix}/") if self.prefix else full_key

    def _get_client(self) -> Any:
        if self._client is None:
            from azure.identity import DefaultAzureCredential  # lazy: [azure] extra
            from azure.storage.blob import BlobServiceClient

            self._client = BlobServiceClient(self.account_url, credential=DefaultAzureCredential())
        return self._client

    def list(self, prefix: str) -> Iterator[ObjectInfo]:
        """Yield every blob whose name starts with ``prefix`` (relative to the store prefix).

        ``prefix`` travels as a query parameter, not in the URL path, so it is not checked for
        dot segments; the container in the path was checked at construction.
        """
        container = self._get_client().get_container_client(self.container)
        for blob in container.list_blobs(name_starts_with=self._full(prefix)):
            yield ObjectInfo(
                key=self._relative(blob.name),
                last_modified=aware_utc(blob.last_modified),
                size=blob.size,
            )

    def get(self, key: str) -> bytes:
        """Return the bytes of ``key`` (relative to the store prefix)."""
        name = self._full(key)
        if has_unsafe_segment(name):
            raise ValueError(
                "Azure blob key has an empty, '.' or '..' segment: the Azure client would "
                "resolve it outside the store's prefix"
            )
        blob = self._get_client().get_blob_client(self.container, name)
        data: bytes = blob.download_blob().readall()
        return data


__all__ = ["AzureBlobStore"]
