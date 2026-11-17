"""The store URI grammar — shared verbatim with the SDK; these vectors are the contract."""

from __future__ import annotations

import pytest

from evalshift_cli.captures.store_uri import StoreURI, has_unsafe_segment, parse_store_uri


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


@pytest.mark.parametrize(
    ("uri", "fragment", "secret"),
    [
        # An Azure connection string pasted as the store: no `@` or `?`, so it reaches the
        # scheme check, and that message must not echo it.
        (
            "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=SUPERSECRET==",
            "accepted forms",
            "SUPERSECRET",
        ),
        ("https://acct.blob.core.windows.net/c/SUPERSECRET", "accepted forms", "SUPERSECRET"),
        ("s3:///AKIASUPERSECRET/key", "names no bucket", "SUPERSECRET"),
        ("az://SUPERSECRETACCOUNT/", "names no container", "SUPERSECRET"),
    ],
)
def test_grammar_rejection_does_not_echo_value(uri: str, fragment: str, secret: str) -> None:
    with pytest.raises(ValueError, match=fragment) as excinfo:
        parse_store_uri(uri)
    assert secret not in str(excinfo.value)


def test_grammar_rejection_messages() -> None:
    with pytest.raises(ValueError) as unsupported:
        parse_store_uri("ftp://bucket/prefix")
    assert str(unsupported.value) == (
        "unsupported store URI scheme 'ftp'; accepted forms: "
        "s3://<bucket>/<prefix>, gs://<bucket>/<prefix>, az://<account>/<container>/<prefix>"
    )
    with pytest.raises(ValueError) as no_scheme:
        parse_store_uri("bucket/prefix")
    assert str(no_scheme.value).startswith("store URI has no scheme; accepted forms: s3://")
    with pytest.raises(ValueError) as no_bucket:
        parse_store_uri("s3:///prefix")
    assert str(no_bucket.value).startswith("store URI names no bucket; accepted forms: s3://")
    with pytest.raises(ValueError) as no_container:
        parse_store_uri("az://acmeprod")
    assert str(no_container.value) == (
        "Azure store URI names no container; expected az://<account>/<container>/<prefix>"
    )


def test_packages_names_every_library_the_scheme_needs() -> None:
    assert parse_store_uri("s3://b/p").packages == "boto3"
    assert parse_store_uri("gs://b/p").packages == "google-cloud-storage"
    assert parse_store_uri("az://a/c/p").packages == "azure-storage-blob azure-identity"


@pytest.mark.parametrize(
    ("uri", "netloc"),
    [
        # urlsplit hands a bracketed host to ipaddress, whose error quotes it.
        ("s3://[SECRETV6]/prefix", "SECRETV6"),
        # A fullwidth solidus NFKC-normalises to "/"; urlsplit's error quotes the whole netloc.
        ("az://SECRETNET\uff0fWORK/c/p", "SECRETNET"),
        # The other urlsplit netloc errors quote nothing, but must read the same.
        ("s3://[vSECRETFUTURE]/prefix", "SECRETFUTURE"),
        ("s3://[::1/prefix", "::1"),
        ("gs://[127.0.0.1]/prefix", "127.0.0.1"),
    ],
)
def test_bad_netloc_is_rejected_without_echoing_it(uri: str, netloc: str) -> None:
    with pytest.raises(ValueError, match="names an invalid bucket or account") as excinfo:
        parse_store_uri(uri)
    assert netloc not in str(excinfo.value)
    # No chained exception either: urlsplit's own error, which quotes the netloc, would ride along
    # in a traceback or on ``__context__``.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_bad_netloc_message_matches_the_sdk() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_store_uri("s3://[SECRETV6]/prefix")
    assert str(excinfo.value) == (
        "store URI names an invalid bucket or account; accepted forms: "
        "s3://<bucket>/<prefix>, gs://<bucket>/<prefix>, az://<account>/<container>/<prefix>"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "az://acmeprod/evals/../SECRETOTHER",
        "az://acmeprod/evals/team/../../SECRETOTHER",
        "az://acmeprod/../SECRETOTHER/team",
        "az://acmeprod/./SECRETOTHER",
        "az://acmeprod/evals/./SECRETOTHER",
        "az://acmeprod/evals//SECRETOTHER",
        "az://acmeprod/evals/team//SECRETOTHER",
    ],
)
def test_azure_path_with_dot_or_empty_segment_is_rejected(uri: str) -> None:
    # The Azure SDK resolves dot segments in the request URL, across containers.
    with pytest.raises(ValueError, match=r"empty, '\.' or '\.\.' segment") as exc:
        parse_store_uri(uri)
    assert "SECRETOTHER" not in str(exc.value)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        # S3 sends the key as given and GCS percent-encodes "/": both keep such prefixes literal.
        ("s3://acme-evals/team/../other", StoreURI("s3", "acme-evals", None, "team/../other")),
        ("gs://acme-evals/./team", StoreURI("gs", "acme-evals", None, "./team")),
        # Ordinary dots inside a segment are fine on Azure too.
        ("az://acmeprod/evals/team.v2/a..b", StoreURI("az", "acmeprod", "evals", "team.v2/a..b")),
    ],
)
def test_dot_segments_stay_literal_outside_azure(uri: str, expected: StoreURI) -> None:
    assert parse_store_uri(uri) == expected


@pytest.mark.parametrize(
    ("path", "unsafe"),
    [
        ("team/captures/s/cap_1.json", False),
        ("team.v2/cap_1..json", False),
        ("team/../other", True),
        ("..", True),
        ("./team", True),
        ("team//a", True),
        ("/team", True),
        ("team/", True),
        ("", True),
    ],
)
def test_has_unsafe_segment(path: str, unsafe: bool) -> None:
    assert has_unsafe_segment(path) is unsafe
