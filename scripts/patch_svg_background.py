"""Patch exported SVGs with a literal white background rectangle.

GitHub's HTML sanitizer strips the inline ``style="background: ..."`` the
draw.io exporter writes on the ``<svg>`` root, which leaves diagrams
transparent (and unreadable in dark mode) once rendered in a README. A
literal ``<rect>`` painted white survives the sanitizer because it is
ordinary SVG content, not a stripped style attribute.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
VIEWBOX_RE = re.compile(r'viewBox="0 0 (?P<w>[\d.]+) (?P<h>[\d.]+)"')
INSERTION_RE = re.compile(r"(<defs\s*/>\s*<g>)")


def patch_svg(path: Path) -> None:
    """Insert a white background ``<rect>`` as the first child of ``<g>``.

    Args:
        path: SVG file to patch in place.

    Raises:
        ValueError: If the file's ``viewBox`` or ``<defs/><g>`` anchor is
            not found, since the insertion point would be ambiguous.
    """
    text = path.read_text()
    viewbox_match = VIEWBOX_RE.search(text)
    if not viewbox_match:
        raise ValueError(f"{path} has no viewBox=\"0 0 W H\" to size the background rect from.")
    width, height = viewbox_match.group("w"), viewbox_match.group("h")

    if 'fill="#ffffff"/>' in text.split("<g>", 1)[-1][:80]:
        print(f"{path} already patched, skipping.")
        return

    insertion_match = INSERTION_RE.search(text)
    if not insertion_match:
        raise ValueError(f"{path} has no <defs/><g> anchor to insert the background rect after.")

    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
    patched = text[: insertion_match.end()] + rect + text[insertion_match.end() :]
    path.write_text(patched)
    print(f"patched {path} ({width}x{height})")


def main() -> int:
    svg_files = sorted(DOCS.glob("*.svg"))
    if not svg_files:
        print(f"No SVG files found under {DOCS}.", file=sys.stderr)
        return 1
    for path in svg_files:
        patch_svg(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
