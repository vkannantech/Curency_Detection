"""Default label metadata for the currency detection project."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CLASS_NAMES = [
    "inr_1_coin",
    "inr_2_coin",
    "inr_5_coin",
    "inr_10_coin",
    "inr_20_coin",
    "inr_10_note",
    "inr_20_note",
    "inr_50_note",
    "inr_100_note",
    "inr_200_note",
    "inr_500_note",
]

SPEAKABLE_LABELS = {
    "inr_1_coin": "1 rupee coin",
    "inr_2_coin": "2 rupee coin",
    "inr_5_coin": "5 rupee coin",
    "inr_10_coin": "10 rupee coin",
    "inr_20_coin": "20 rupee coin",
    "inr_10_note": "10 rupee note",
    "inr_20_note": "20 rupee note",
    "inr_50_note": "50 rupee note",
    "inr_100_note": "100 rupee note",
    "inr_200_note": "200 rupee note",
    "inr_500_note": "500 rupee note",
}


def to_spoken_label(label: str) -> str:
    """Convert a model class name into a user-friendly spoken phrase."""
    return SPEAKABLE_LABELS.get(label, label.replace("_", " "))


@dataclass(frozen=True, slots=True)
class CurrencyLabelInfo:
    label: str
    value: int | None
    kind: str


def parse_label_info(label: str) -> CurrencyLabelInfo:
    """Extract denomination and type from a class label like inr_100_note."""
    value_match = re.search(r"(\d+)", label)
    kind = "note" if "note" in label else "coin" if "coin" in label else "currency"
    value = int(value_match.group(1)) if value_match else None
    return CurrencyLabelInfo(label=label, value=value, kind=kind)


def rupee_value_for_label(label: str) -> int:
    """Return the rupee value for a label, or 0 if it cannot be parsed."""
    info = parse_label_info(label)
    return info.value or 0
