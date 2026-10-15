"""Ladowanie oznaczonego zbioru testowego."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "samples.jsonl"


@dataclass(frozen=True)
class Sample:
    id: str
    lang: str
    category: str
    text: str
    label: int  # 1 = manipulacyjny, 0 = czysty
    pattern: str
    hard_for_rules: bool
    note: str

    @property
    def is_manipulative(self) -> bool:
        return self.label == 1


def load_samples(path: Path | str = DATA_PATH) -> list[Sample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono zbioru testowego: {path}")
    samples: list[Sample] = []
    for line in _iter_nonempty_lines(path):
        raw = json.loads(line)
        samples.append(
            Sample(
                id=raw["id"],
                lang=raw["lang"],
                category=raw["category"],
                text=raw["text"],
                label=int(raw["label"]),
                pattern=raw.get("pattern", ""),
                hard_for_rules=bool(raw.get("hard_for_rules", False)),
                note=raw.get("note", ""),
            )
        )
    _validate(samples)
    return samples


def _iter_nonempty_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def _validate(samples: list[Sample]) -> None:
    seen: set[str] = set()
    for s in samples:
        if s.id in seen:
            raise ValueError(f"Zduplikowane id probki: {s.id}")
        seen.add(s.id)
        if s.label not in (0, 1):
            raise ValueError(f"Nieprawidlowa etykieta w {s.id}: {s.label}")


if __name__ == "__main__":
    data = load_samples()
    pos = sum(s.label for s in data)
    hard = sum(1 for s in data if s.hard_for_rules)
    print(f"Wczytano {len(data)} probek: {pos} manipulacyjnych, {len(data) - pos} czystych.")
    print(f"Oznaczonych jako trudne dla regul: {hard}")
    langs = sorted({s.lang for s in data})
    cats = sorted({s.category for s in data})
    print(f"Jezyki: {', '.join(langs)}")
    print(f"Kategorie: {', '.join(cats)}")
