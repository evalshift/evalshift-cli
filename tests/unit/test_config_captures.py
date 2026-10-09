"""The ``captures:`` block of ``evalshift.yaml`` — where the SDK's recordings live."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import CapturesConfig, EvalShiftConfig

_PROMPT = {"id": "p", "detection": "manual", "content": "hello {name}"}


def test_default_is_no_store() -> None:
    cfg = EvalShiftConfig(prompts=[_PROMPT])
    assert cfg.captures == CapturesConfig()
    assert cfg.captures.store is None


@pytest.mark.parametrize(
    "uri",
    ["s3://acme-evals/support-agent", "gs://acme-evals/x", "az://acmeprod/evals/x"],
)
def test_store_accepts_each_scheme(uri: str) -> None:
    cfg = EvalShiftConfig(prompts=[_PROMPT], captures={"store": uri})
    assert cfg.captures.store == uri


def test_store_is_stripped() -> None:
    cfg = EvalShiftConfig(prompts=[_PROMPT], captures={"store": "  s3://b/p  "})
    assert cfg.captures.store == "s3://b/p"


def test_store_rejects_unknown_scheme_naming_forms() -> None:
    with pytest.raises(ValidationError, match="accepted forms"):
        EvalShiftConfig(prompts=[_PROMPT], captures={"store": "ftp://b/p"})


@pytest.mark.parametrize("uri", ["s3://bucket/prefix?x=1", "s3://user@bucket/p", "az://acct"])
def test_store_rejects_credentials_and_query(uri: str) -> None:
    with pytest.raises(ValidationError, match="accepted forms"):
        EvalShiftConfig(prompts=[_PROMPT], captures={"store": uri})


def test_unknown_key_in_captures_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvalShiftConfig(prompts=[_PROMPT], captures={"store": "s3://b/p", "bucket": "b"})


def test_loader_reports_store_error_at_its_location(tmp_path: Path) -> None:
    path = tmp_path / "evalshift.yaml"
    path.write_text(
        "prompts:\n  - id: p\n    detection: manual\n    content: hi\n"
        "captures:\n  store: ftp://bucket/prefix\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_config(path)
    [detail] = info.value.details
    assert detail.location == "captures.store"
    assert "accepted forms" in detail.message


@pytest.mark.parametrize(
    ("store", "secrets"),
    [
        ("s3://AKIAKEY:SECRET@b/p", ["AKIAKEY", "SECRET"]),
        ("az://acct/c?sv=1&sig=SECRETSIG", ["SECRETSIG", "sig="]),
        (
            "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=SUPERSECRET==",
            ["SUPERSECRET", "AccountKey"],
        ),
        ("s3:///AKIASUPERSECRET/key", ["SUPERSECRET"]),
    ],
)
def test_loader_error_never_echoes_a_secret_store(
    tmp_path: Path, store: str, secrets: list[str]
) -> None:
    path = tmp_path / "evalshift.yaml"
    path.write_text(
        "prompts:\n  - id: p\n    detection: manual\n    content: hi\n"
        f"captures:\n  store: '{store}'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_config(path)
    console = Console(record=True, width=200)
    console.print(info.value.format_rich())
    rendered = console.export_text()
    plain = str(info.value)
    assert "accepted forms" in plain
    assert "accepted forms" in rendered
    for secret in secrets:
        assert secret not in plain
        assert secret not in rendered
