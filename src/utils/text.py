from __future__ import annotations

import textwrap


def wrap_paragraph(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def as_yes_no(value: bool) -> str:
    return "Yes" if value else "No"
