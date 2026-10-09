"""Read-side adapters for the object stores ``captures.store`` can name.

Each module imports its client library lazily, inside the method that needs it, so importing
this package never requires a cloud SDK; :func:`evalshift_cli.captures.remote.open_store` picks
the adapter from the URI scheme and reports a missing extra before any import happens.
"""

from __future__ import annotations

from datetime import UTC, datetime


def aware_utc(value: datetime | None) -> datetime | None:
    """``value`` with a naive timestamp taken as UTC; aware values and ``None`` pass through.

    boto3, google-cloud-storage and azure-storage-blob all return aware datetimes today; this
    keeps ``--since`` comparisons from raising if one ever hands back a naive one.
    """
    return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


__all__ = ["aware_utc"]
