"""Read and promote agent captures written by the ``evalshift-sdk`` package.

The capture SDK installs *inside a user's agent process* and records what the
agent does, writing one JSON file per invocation to
``<base>/captures/<suite>/<capture_id>.json``. This package is the CLI side of
that contract: it reads those files (:mod:`evalshift_cli.captures.reader`) and
*promotes* a capture into a golden suite case (:mod:`evalshift_cli.captures.promote`)
that ``evalshift run`` can evaluate.

The SDK and CLI never call each other — the only contract is the capture envelope and its
layout (``captures/<suite>/<capture_id>.json`` + ``toolsets/<hex>.json``), on local disk or
mirrored from an object store by :mod:`evalshift_cli.captures.remote`.
"""

from __future__ import annotations

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.promote import (
    BuiltExample,
    PromoteOptions,
    build_example_from_capture,
    rebuild_golden_jsonl,
    write_promoted_case,
)
from evalshift_cli.captures.reader import (
    CaptureError,
    CaptureRecord,
    capture_base,
    captures_root,
    find_capture,
    iter_captures,
    load_capture,
    promoted_capture_ids,
    suites_root,
)

__all__ = [
    "BuiltExample",
    "CaptureEnvelope",
    "CaptureError",
    "CaptureRecord",
    "PromoteOptions",
    "PromotedCase",
    "build_example_from_capture",
    "capture_base",
    "captures_root",
    "find_capture",
    "iter_captures",
    "load_capture",
    "promoted_capture_ids",
    "rebuild_golden_jsonl",
    "suites_root",
    "write_promoted_case",
]
