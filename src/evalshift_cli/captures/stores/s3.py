"""Amazon S3 (and S3-compatible) read adapter. Extra: ``evalshift[s3]``.

boto3 honours ``AWS_ENDPOINT_URL``, which is how MinIO, Cloudflare R2, Backblaze B2 and Ceph
are reached with no endpoint knob here. Credentials come from boto3's default chain.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from evalshift_cli.captures.remote import ObjectInfo
from evalshift_cli.captures.stores import aware_utc


class S3Store:
    """List and read objects under ``s3://<bucket>/<prefix>``."""

    def __init__(self, bucket: str, prefix: str = "", *, client: Any | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.uri = f"s3://{bucket}/{self.prefix}".rstrip("/")
        self._client = client

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _relative(self, full_key: str) -> str:
        return full_key.removeprefix(f"{self.prefix}/") if self.prefix else full_key

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # lazy: the [s3] extra is optional

            self._client = boto3.client("s3")
        return self._client

    def list(self, prefix: str) -> Iterator[ObjectInfo]:
        """Yield every object whose key starts with ``prefix`` (relative to the store prefix)."""
        paginator = self._get_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._full(prefix)):
            for obj in page.get("Contents", []):
                yield ObjectInfo(
                    key=self._relative(obj["Key"]),
                    last_modified=aware_utc(obj.get("LastModified")),
                    size=obj.get("Size"),
                )

    def get(self, key: str) -> bytes:
        """Return the bytes of ``key`` (relative to the store prefix)."""
        body = self._get_client().get_object(Bucket=self.bucket, Key=self._full(key))["Body"]
        data: bytes = body.read()
        return data


__all__ = ["S3Store"]
