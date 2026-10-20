"""README.md links must be absolute, because PyPI renders it as the project page.

`pyproject.toml` ships README.md as the long description. On pypi.org a relative
link such as `docs/configuration.md` resolves against
`https://pypi.org/project/evalshift/`, so it lands on a PyPI 404. 35 of the 51
links in the 1.5.0 project description did. On GitHub, absolute links to the
repo work exactly as relative ones did, so nothing is lost by writing them out.

In-page anchors (`#section`) are allowed: they never leave the rendered page.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_LINK_TARGETS = (
    re.compile(r"\]\(\s*<?([^)\s>]+)"),  # [text](target) and ![alt](target)
    re.compile(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?\s*$", re.MULTILINE),  # [ref]: target
    re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']"""),  # inline HTML
)
_ABSOLUTE = re.compile(r"^(?:https?:|mailto:|#)")


def _link_targets(markdown: str) -> list[str]:
    prose = _FENCE.sub("", markdown)
    return [m.group(1) for pattern in _LINK_TARGETS for m in pattern.finditer(prose)]


def test_pyproject_ships_readme_as_the_long_description() -> None:
    """Guards the premise: if the readme field changes, revisit this file."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["readme"] == "README.md"


def test_readme_links_resolve_on_pypi() -> None:
    targets = _link_targets((REPO_ROOT / "README.md").read_text(encoding="utf-8"))

    relative = [t for t in targets if not _ABSOLUTE.match(t)]

    assert targets, "found no links in README.md; the extractor is broken"
    assert relative == [], (
        "README.md is the PyPI project page, where relative links 404; write them "
        "as https://github.com/evalshift/evalshift-cli/blob/main/<path> "
        f"(tree/main/ for directories): {relative}"
    )


def test_link_extractor_sees_every_link_form() -> None:
    sample = (
        "[a](docs/a.md) ![b](img/b.png) [c](https://x.dev) [d](#anchor)\n"
        "[ref]: docs/ref.md\n"
        '<a href="LICENSE">l</a> <img src="logo.svg">\n'
        "```\n[skipped](in/code.md)\n```\n"
    )

    assert _link_targets(sample) == [
        "docs/a.md",
        "img/b.png",
        "https://x.dev",
        "#anchor",
        "docs/ref.md",
        "LICENSE",
        "logo.svg",
    ]
