"""An in-memory ``RemoteStore`` and a valid-capture builder for the remote-fetch tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from evalshift_cli.captures.remote import ObjectInfo

TOOLSET_HEX = "ab" * 32
TOOLSET_REF = f"sha256:{TOOLSET_HEX}"
TOOLSET_KEY = f"toolsets/{TOOLSET_HEX}.json"
TOOLSET_PAYLOAD = b'{"tools": []}'


def capture_payload(
    capture_id: str,
    suite: str = "alpha",
    *,
    model_input: Any = "where is order 12345?",
    created_at: str = "2026-06-16T12:00:00+00:00",
) -> bytes:
    """A minimal capture the CLI reader accepts: one model call, one final output."""
    envelope = {
        "schema_version": "2.0.0",
        "capture_id": capture_id,
        "suite": suite,
        "input_hash": f"hash-{capture_id}",
        "code_version": "v1",
        "created_at": created_at,
        "trace": {
            "run_id": capture_id,
            "prompt_id": suite,
            "example_id": capture_id,
            "role": "source",
            "events": [
                {
                    "type": "model_call",
                    "sequence_index": 0,
                    "timestamp": created_at,
                    "metadata": {},
                    "model_id": "m",
                    "input": model_input,
                    "output": "out",
                    "toolset_ref": TOOLSET_REF,
                    "tools_offered": [],
                },
                {
                    "type": "final_output",
                    "sequence_index": 1,
                    "timestamp": created_at,
                    "metadata": {},
                    "text": "done",
                },
            ],
        },
    }
    return json.dumps(envelope).encode("utf-8")


class FakeRemoteStore:
    """Dict-backed store: ``objects`` maps relative key -> bytes; ``fail_keys`` raise on get."""

    uri = "s3://fake/prefix"

    def __init__(
        self,
        objects: dict[str, bytes] | None = None,
        *,
        modified: dict[str, datetime | None] | None = None,
        sizes: dict[str, int | None] | None = None,
        fail_keys: set[str] | None = None,
    ) -> None:
        self.objects: dict[str, bytes] = dict(objects or {})
        # A key present with value None means "the provider reported nothing" (never filtered).
        self.modified: dict[str, datetime | None] = dict(modified or {})
        self.sizes: dict[str, int | None] = dict(sizes or {})
        self.fail_keys = set(fail_keys or ())
        self.gets: list[str] = []

    def list(self, prefix: str) -> Iterator[ObjectInfo]:
        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield ObjectInfo(
                    key=key,
                    last_modified=self.modified.get(key, datetime(2026, 6, 16, tzinfo=UTC)),
                    size=self.sizes.get(key, len(self.objects[key])),
                )

    def get(self, key: str) -> bytes:
        self.gets.append(key)
        if key in self.fail_keys:
            raise OSError(f"simulated failure reading {key}")
        return self.objects[key]


def populated_store(*capture_ids: str, suite: str = "alpha") -> FakeRemoteStore:
    """A store holding the shared toolset sidecar plus one capture per id.

    Each capture gets a distinct model input: ``capture sync`` drops content duplicates, so two
    captures with the same input would promote as one and hide a fetch bug behind a dedup.
    """
    objects = {TOOLSET_KEY: TOOLSET_PAYLOAD}
    for cid in capture_ids:
        objects[f"captures/{suite}/{cid}.json"] = capture_payload(
            cid, suite, model_input=f"where is order {cid}?"
        )
    return FakeRemoteStore(objects)
