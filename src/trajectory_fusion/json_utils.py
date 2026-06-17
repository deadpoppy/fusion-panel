from __future__ import annotations

import re
from typing import Any

from .tools import strip_reasoning_text


TEXT_DECISION_PATTERN = re.compile(
    r"<text_decision>\s*(none|replace)\s*</text_decision>",
    re.IGNORECASE,
)
TEXT_REPLACEMENT_PATTERN = re.compile(
    r"<text_replacement>\s*(.*?)\s*</text_replacement>",
    re.IGNORECASE | re.DOTALL,
)


def parse_text_decision(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    visible = strip_reasoning_text(text)
    if not visible:
        return {}

    decision_match = TEXT_DECISION_PATTERN.search(visible)
    if not decision_match:
        return {}

    operation = decision_match.group(1).lower()
    if operation == "none":
        return {"operation": "none", "text": None}

    replacement_match = TEXT_REPLACEMENT_PATTERN.search(visible)
    if not replacement_match:
        return {}
    replacement = strip_reasoning_text(replacement_match.group(1))
    if not replacement:
        return {}

    return {"operation": "replace", "text": replacement}
