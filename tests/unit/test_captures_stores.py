"""Each adapter against a stub client: pagination, prefix handling, timestamps, lazy import."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import pytest

from evalshift_cli.captures import remote as remote_module
from evalshift_cli.captures.remote import RemoteStore, RemoteStoreUnavailable, open_store
from evalshift_cli.captures.store_uri import parse_store_uri
from evalshift_cli.captures.stores.azure import AzureBlobStore
from evalshift_cli.captures.stores.gcs import GCSStore
from evalshift_cli.captures.stores.s3 import S3Store

T0 = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)

# --- S3 ------------------------------------------------------------------------------------


class FakePaginator:
    def __init__(self, pages: list[list[dict[str, Any]]], seen: list[dict[str, Any]]) -> None:
        self._pages, self._seen = pages, seen

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._seen.append(kwargs)
        return [{"Contents": page} for page in self._pages] + [{}]  # a page with no Contents


class FakeS3Client:
    def __init__(self, pages: list[list[dict[str, Any]]], blobs: dict[str, bytes]) -> None:
        self.pages, self.blobs = pages, blobs
        self.paginate_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self.pages, self.paginate_calls)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        return {"Body": io.BytesIO(self.blobs[kwargs["Key"]])}


def test_s3_list_paginates_and_strips_prefix() -> None:
    client = FakeS3Client(
        pages=[
            [{"Key": "p/captures/a/cap_1.json", "LastModified": T0, "Size": 10}],
            [{"Key": "p/captures/a/cap_2.json", "LastModified": T0, "Size": 11}],
        ],
        blobs={},
    )
    store = S3Store("acme-evals", "p/", client=client)
    infos = list(store.list("captures/"))
    assert [i.key for i in infos] == ["captures/a/cap_1.json", "captures/a/cap_2.json"]
    assert infos[0].last_modified == T0 and infos[1].size == 11
    assert client.paginate_calls == [{"Bucket": "acme-evals", "Prefix": "p/captures/"}]
    assert store.uri == "s3://acme-evals/p"


def test_s3_get_reads_body_with_prefix() -> None:
    client = FakeS3Client(pages=[], blobs={"p/toolsets/ab.json": b"[]"})
    assert S3Store("acme-evals", "p", client=client).get("toolsets/ab.json") == b"[]"
    assert client.get_calls == [{"Bucket": "acme-evals", "Key": "p/toolsets/ab.json"}]


def test_s3_without_prefix_keys_are_relative_already() -> None:
    client = FakeS3Client(pages=[[{"Key": "captures/a/cap_1.json"}]], blobs={})
    [info] = S3Store("acme-evals", client=client).list("captures/")
    assert info.key == "captures/a/cap_1.json"
    assert info.last_modified is None and info.size is None


def test_s3_satisfies_protocol() -> None:
    assert isinstance(S3Store("b", client=FakeS3Client([], {})), RemoteStore)


def test_s3_naive_last_modified_is_taken_as_utc() -> None:
    naive = datetime(2026, 10, 6, 12, 0)
    client = FakeS3Client(
        pages=[[{"Key": "captures/a/cap_1.json", "LastModified": naive}]], blobs={}
    )
    [info] = S3Store("acme-evals", client=client).list("captures/")
    assert info.last_modified == T0 and info.last_modified.tzinfo is not None


# --- GCS -----------------------------------------------------------------------------------


class FakeGCSBlob:
    def __init__(self, name: str, data: bytes = b"", updated: datetime | None = T0) -> None:
        self.name, self._data, self.updated, self.size = name, data, updated, len(data)

    def download_as_bytes(self) -> bytes:
        return self._data


class FakeGCSBucket:
    def __init__(self, blobs: dict[str, FakeGCSBlob]) -> None:
        self._blobs = blobs

    def blob(self, key: str) -> FakeGCSBlob:
        return self._blobs[key]


class FakeGCSClient:
    def __init__(self, blobs: dict[str, FakeGCSBlob]) -> None:
        self._blobs = blobs
        self.list_calls: list[tuple[str, str]] = []

    def list_blobs(self, bucket: str, prefix: str) -> list[FakeGCSBlob]:
        self.list_calls.append((bucket, prefix))
        return [b for k, b in sorted(self._blobs.items()) if k.startswith(prefix)]

    def bucket(self, name: str) -> FakeGCSBucket:
        return FakeGCSBucket(self._blobs)


def test_gcs_list_and_get() -> None:
    blobs = {
        "p/captures/a/cap_1.json": FakeGCSBlob("p/captures/a/cap_1.json", b"{}"),
        "p/toolsets/ab.json": FakeGCSBlob("p/toolsets/ab.json", b"[]"),
    }
    client = FakeGCSClient(blobs)
    store = GCSStore("acme-evals", "p", client=client)
    [info] = store.list("captures/")
    assert info.key == "captures/a/cap_1.json" and info.size == 2 and info.last_modified == T0
    assert client.list_calls == [("acme-evals", "p/captures/")]
    assert store.get("toolsets/ab.json") == b"[]"
    assert store.uri == "gs://acme-evals/p"


# --- Azure ---------------------------------------------------------------------------------


class FakeBlobProperties:
    def __init__(self, name: str, size: int, last_modified: datetime) -> None:
        self.name, self.size, self.last_modified = name, size, last_modified


class FakeDownload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class FakeBlobClient:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def download_blob(self) -> FakeDownload:
        return FakeDownload(self._data)


class FakeContainerClient:
    def __init__(self, props: list[FakeBlobProperties], seen: list[str]) -> None:
        self._props, self._seen = props, seen

    def list_blobs(self, name_starts_with: str) -> list[FakeBlobProperties]:
        self._seen.append(name_starts_with)
        return [p for p in self._props if p.name.startswith(name_starts_with)]


class FakeBlobServiceClient:
    def __init__(self, props: list[FakeBlobProperties], blobs: dict[str, bytes]) -> None:
        self._props, self._blobs = props, blobs
        self.list_prefixes: list[str] = []
        self.blob_requests: list[tuple[str, str]] = []

    def get_container_client(self, container: str) -> FakeContainerClient:
        return FakeContainerClient(self._props, self.list_prefixes)

    def get_blob_client(self, container: str, blob: str) -> FakeBlobClient:
        self.blob_requests.append((container, blob))
        return FakeBlobClient(self._blobs[blob])


def test_azure_list_and_get() -> None:
    client = FakeBlobServiceClient(
        props=[FakeBlobProperties("p/captures/a/cap_1.json", 7, T0)],
        blobs={"p/captures/a/cap_1.json": b"{}"},
    )
    store = AzureBlobStore("acmeprod", "evals", "p", client=client)
    [info] = store.list("captures/")
    assert info.key == "captures/a/cap_1.json" and info.size == 7 and info.last_modified == T0
    assert client.list_prefixes == ["p/captures/"]
    assert store.get("captures/a/cap_1.json") == b"{}"
    assert client.blob_requests == [("evals", "p/captures/a/cap_1.json")]
    assert store.uri == "az://acmeprod/evals/p"
    assert store.account_url == "https://acmeprod.blob.core.windows.net"


# --- open_store ----------------------------------------------------------------------------


def test_open_store_dispatches_without_building_a_client() -> None:
    s3 = open_store(parse_store_uri("s3://b/p"))
    assert isinstance(s3, S3Store) and s3._client is None
    assert isinstance(open_store(parse_store_uri("gs://b/p")), GCSStore)
    assert isinstance(open_store(parse_store_uri("az://a/c/p")), AzureBlobStore)


def test_open_store_missing_extra_names_pip_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote_module, "_installed", lambda module: False)
    with pytest.raises(RemoteStoreUnavailable) as info:
        open_store(parse_store_uri("gs://b/p"))
    assert info.value.hint == 'install it with: pip install "evalshift[gcs]"'


def test_open_store_azure_needs_identity_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote_module, "_installed", lambda module: module != "azure.identity")
    with pytest.raises(RemoteStoreUnavailable) as info:
        open_store(parse_store_uri("az://a/c/p"))
    assert "'azure.identity'" in info.value.summary
    assert info.value.hint == 'install it with: pip install "evalshift[azure]"'
