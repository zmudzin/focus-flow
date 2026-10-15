"""Detektor regulowy (baseline).

Reprezentuje to, co osiagniesz BEZ modelu AI: lekki, szybki, deterministyczny
zestaw heurystyk slowo-kluczowych i regex, podobny w duchu do list typu
EasyList/Fanboy. To jest poprzeczka, ktora lokalny LLM musi przeskoczyc, zeby
caly pomysl mial sens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Kazdy wzorzec: (skompilowany regex, kategoria). Celowo szczodre, bo
# produkcyjne listy tez takie sa \u2014 chcemy pokazac realny baseline, lacznie z
# jego naturalna sklonnoscia do falszywych trafien.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # fake urgency / scarcity
    (re.compile(r"\b\d{1,2}:\d{2}\b"), "fake_urgency"),  # licznik mm:ss
    (re.compile(r"\bonly\s+\d+\s+left\b", re.I), "fake_urgency"),
    (re.compile(r"\b(hurry|last chance|before it'?s gone|act now|don'?t miss)\b", re.I), "fake_urgency"),
    (re.compile(r"\b(ends? (in|tonight|soon)|offer (ends|expires)|price going up)\b", re.I), "fake_urgency"),
    (re.compile(r"\b(zosta\u0142y|tylko)\s+\d+\s+(sztuk|sztuki)", re.I), "fake_urgency"),
    (re.compile(r"\b(po\u015bpiesz|ostatnia szansa|zanim b\u0119dzie za p\u00f3\u017ano|kup teraz)\b", re.I), "fake_urgency"),
    (re.compile(r"\bsold out\b|\bselling fast\b|\bremaining\b", re.I), "fake_urgency"),
    # confirmshaming
    (re.compile(r"\bno,?\s+(thanks|i)\b.*\b(don'?t want|prefer|rather|enjoy|miss out|stay)\b", re.I), "confirmshaming"),
    (re.compile(r"\bi'?ll stay (uninformed|un)", re.I), "confirmshaming"),
    (re.compile(r"\bwol\u0119 (przep\u0142aca\u0107|nie korzysta\u0107)", re.I), "confirmshaming"),
    # clickbait
    (re.compile(r"\byou won'?t believe\b", re.I), "clickbait"),
    (re.compile(r"\b(one weird trick|this one trick|simple (trick|secret))\b", re.I), "clickbait"),
    (re.compile(r"\b(doctors?|experts?) (hate|are furious|hate when)\b", re.I), "clickbait"),
    (re.compile(r"\bwhat happened next\b|\bwill shock you\b|\bleft .* speechless\b", re.I), "clickbait"),
    (re.compile(r"\bnumber \d+ will\b|\bkeep reading to find out\b|\bfind out\b", re.I), "clickbait"),
    (re.compile(r"\bnie uwierzysz\b|\bprosty trik\b|\beksperci w szoku\b", re.I), "clickbait"),
    # aggressive popup / nag
    (re.compile(r"\b(wait!?|before you (go|leave)|don'?t leave)\b", re.I), "aggressive_popup"),
    (re.compile(r"\b(unlock|claim) (your )?(\d+%|discount|prize|offer)\b", re.I), "aggressive_popup"),
    (re.compile(r"\b(enter your email|spin the wheel|won'?t appear again)\b", re.I), "aggressive_popup"),
    (re.compile(r"\bdo not close\b|\bkeep this (tab|window) open\b", re.I), "aggressive_popup"),
    # forced continuity
    (re.compile(r"\b(charged?|charge).{0,30}(month|monthly|/mo|automatically)\b", re.I), "forced_continuity"),
    (re.compile(r"\b(unless you cancel|cancellation requires|calling during)\b", re.I), "forced_continuity"),
    (re.compile(r"\bpobierzemy\b.*\b(mies|miesi\u0105c)", re.I), "forced_continuity"),
    # ai slop
    (re.compile(r"\b(in today'?s (fast-?paced|dynamic) (digital )?world|at the end of the day)\b", re.I), "ai_slop"),
    (re.compile(r"\b(leverage|synerg)\w*\b", re.I), "ai_slop"),
    (re.compile(r"\b(it is (worth noting|important to note)|comprehensive guide)\b", re.I), "ai_slop"),
    (re.compile(r"\b(w dzisiejszym dynamicznym \u015bwiecie|na ko\u0144cu dnia|synergia)\b", re.I), "ai_slop"),
]


@dataclass(frozen=True)
class Detection:
    is_manipulative: bool
    category: str | None
    matched: list[str]

    @property
    def strength(self) -> int:
        """Liczba trafien = prosty sygnal pewnosci reguly."""
        return len(self.matched)


def detect(text: str, allowed: set[str] | None = None) -> Detection:
    """Zwraca werdykt regulowy dla fragmentu tekstu.

    allowed: jesli podane, bierze pod uwage tylko wzorce z tych kategorii
    (uzywane do zadan zawezonych, np. tylko dark patterns albo tylko slop).
    """
    hits: list[tuple[str, str]] = []
    for pattern, category in _PATTERNS:
        if allowed is not None and category not in allowed:
            continue
        m = pattern.search(text)
        if m:
            hits.append((category, m.group(0)))
    if not hits:
        return Detection(False, None, [])
    # Kategoria = najczestsza wsrod trafien.
    counts: dict[str, int] = {}
    for cat, _ in hits:
        counts[cat] = counts.get(cat, 0) + 1
    best = max(counts, key=counts.get)
    return Detection(True, best, [frag for _, frag in hits])


if __name__ == "__main__":
    from .samples import load_samples

    for s in load_samples():
        d = detect(s.text)
        flag = "MANIP" if d.is_manipulative else "clean"
        print(f"[{s.id}] rule={flag:5} truth={'MANIP' if s.label else 'clean'}  {s.text[:70]}")
