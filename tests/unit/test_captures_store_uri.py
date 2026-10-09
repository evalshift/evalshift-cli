"""The store URI grammar — shared verbatim with the SDK; these vectors are the contract."""

from __future__ import annotations

import pytest

from evalshift_cli.captures.store_uri import StoreURI, parse_store_uri


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://acme-evals/support-agent", StoreURI("s3", "acme-evals", None, "support-agent")),
        ("s3://acme-evals/support-agent/", StoreURI("s3", "acme-evals", None, "support-agent")),
        ("s3://acme-evals", StoreURI("s3", "acme-evals", None, "")),
        ("s3://acme-evals/", StoreURI("s3", "acme-evals", None, "")),
        ("s3://acme-evals/a/b/c", StoreURI("s3", "acme-evals", None, "a/b/c")),
        ("gs://acme-evals/support-agent", StoreURI("gs", "acme-evals", None, "support-agent")),
        ("az://acmeprod/evals/support-agent", StoreURI("az", "acmeprod", "evals", "support-agent")),
        ("az://acmeprod/evals", StoreURI("az", "acmeprod", "evals", "")),
        ("az://acmeprod/evals/a/b/", StoreURI("az", "acmeprod", "evals", "a/b")),
        ("  s3://acme-evals/x  ", StoreURI("s3", "acme-evals", None, "x")),
    ],
)
def test_accepted_forms(uri: str, expected: StoreURI) -> None:
    assert parse_store_uri(uri) == expected


@pytest.mark.parametrize(
    ("uri", "fragment"),
    [
        ("ftp://bucket/prefix", "accepted forms"),
        ("http://bucket/prefix", "accepted forms"),
        ("bucket/prefix", "accepted forms"),
        ("", "accepted forms"),
        ("s3:///prefix", "names no bucket"),
        ("az://acmeprod", "names no container"),
        ("az://acmeprod/", "names no container"),
        ("s3://key:secret@bucket/prefix", "credential"),
        ("s3://bucket/prefix?region=eu", "query"),
    ],
)
def test_rejected_forms(uri: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        parse_store_uri(uri)


def test_extra_name_per_scheme() -> None:
    assert parse_store_uri("s3://b").extra == "s3"
    assert parse_store_uri("gs://b").extra == "gcs"
    assert parse_store_uri("az://a/c").extra == "azure"


@pytest.mark.parametrize(
    ("uri", "secret"),
    [
        ("az://acct/c?sv=2024&sig=SECRETSIG", "SECRETSIG"),
        ("s3://AKIAKEY:SECRET@bucket/p", "SECRET"),
        ("s3://AKIAKEY:SECRET@bucket/p", "AKIAKEY"),
    ],
)
def test_credential_rejection_does_not_echo_secret(uri: str, secret: str) -> None:
    with pytest.raises(ValueError, match="credential chain") as excinfo:
        parse_store_uri(uri)
    assert secret not in str(excinfo.value)
