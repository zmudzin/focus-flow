#!/usr/bin/env python3
"""Tanie eksperymenty w sensownej kolejnosci \u2014 diagnoza zanim dorzucimy compute.

Eksperyment A: ROZDZIELENIE ZADAN
    Zamiast jednego worka 7 kategorii, dwa osobne zadania binarne:
      - dark_pattern (presja/wina/oszustwo/bait)
      - slop (pusty wypelniacz vs tresc informacyjna)
    Hipoteza: waski, dobrze zdefiniowany problem jest latwiejszy dla malego
    modelu niz semantyczny worek. Zero pobierania, sama zmiana definicji.

Eksperyment B: HYBRYDA (reguly + LLM) na zadaniu dark_pattern
    - OR  (union)        : flaga gdy reguly LUB llm \u2014 maks. recall
    - AND (intersection) : flaga gdy reguly I llm   \u2014 maks. precision
    - routed             : silne trafienie regul (>=2) ufamy regulom;
                           slabe/zero \u2014 pytamy LLM. Celuje w obserwowane bledy
                           (reguly falszuja na slowach-pulapkach, LLM na UI).

Uruchom:
    python3 experiments.py                 # A + B (uzywa modelu)
    python3 experiments.py --model X.gguf  # inny model (np. 3B \u2014 eksperyment C)
"""

from __future__ import annotations

import argparse
import sys
import time

from slopscrubber import rules
from slopscrubber.evaluate import Metrics
from slopscrubber.llm import LlamaDetector
from slopscrubber.samples import Sample, load_samples

DARK_CATS = {"fake_urgency", "confirmshaming", "clickbait", "aggressive_popup", "forced_continuity"}
SLOP_CATS = {"ai_slop"}

DARK_PROMPT = (
    "You detect manipulative DARK PATTERNS in web UI text. A dark pattern pressures, "
    "guilt-trips, deceives, baits, or coerces the user (fake urgency, confirmshaming, "
    "clickbait, nagging popups, hidden recurring charges). Judge intent, not writing quality. "
    "Neutral UI labels (buttons, links, menu items), factual statements, transparent/fair "
    "disclosures, and honest headlines are NOT dark patterns. "
    'Respond with ONLY JSON: {"flag": true|false, "reason": "<=10 words"}'
)
DARK_FEWSHOT = [
    ("No thanks, I don't want to save money.", '{"flag": true, "reason": "confirmshaming decline"}'),
    ("Add to cart", '{"flag": false, "reason": "neutral button"}'),
    ("Unsubscribe", '{"flag": false, "reason": "neutral link"}'),
    ("The sale runs until Sunday; prices return Monday.", '{"flag": false, "reason": "factual deadline"}'),
    ("Only 2 left \u2014 hurry, offer ends in 04:59!", '{"flag": true, "reason": "fake urgency"}'),
    ("Cancel anytime in Settings, no phone call.", '{"flag": false, "reason": "fair disclosure"}'),
]

SLOP_PROMPT = (
    "You detect AI SLOP: content-free filler that says nothing \u2014 empty buzzwords, padding, "
    "tautologies, circular text, non-answers. Dense, specific, informative text is NOT slop, "
    "even if short. Judge information content, not tone. "
    'Respond with ONLY JSON: {"flag": true|false, "reason": "<=10 words"}'
)
SLOP_FEWSHOT = [
    (
        "In today's fast-paced world, leveraging synergies unlocks a myriad of opportunities.",
        '{"flag": true, "reason": "empty buzzword filler"}',
    ),
    (
        "The 2024 model adds a 48MP sensor and a 4,500 mAh battery, up from 3,900.",
        '{"flag": false, "reason": "specific informative facts"}',
    ),
    (
        "Ultimately the choice depends on your individual needs and preferences.",
        '{"flag": true, "reason": "non-answer, says nothing"}',
    ),
    ("Add to cart", '{"flag": false, "reason": "neutral label, not filler"}'),
]


def _subset(samples: list[Sample], positive_cats: set[str]) -> list[tuple[Sample, int]]:
    """Zwraca (probka, etykieta_zadania) tylko dla pozytywnych kategorii + clean."""
    out: list[tuple[Sample, int]] = []
    for s in samples:
        if s.category in positive_cats:
            out.append((s, 1))
        elif s.category == "clean":
            out.append((s, 0))
    return out


def _score(name: str, pairs: list[tuple[int, int]]) -> Metrics:
    m = Metrics(name)
    for truth, pred in pairs:
        m.add(truth, pred)
    return m


def _fmt(m: Metrics) -> str:
    return (
        f"acc={m.accuracy:.0%}  prec={m.precision:.0%}  rec={m.recall:.0%}  "
        f"F1={m.f1:.0%}  (TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn})"
    )


def run_task(
    title: str,
    samples: list[Sample],
    positive_cats: set[str],
    rule_allowed: set[str],
    det: LlamaDetector | None,
    system_prompt: str,
    fewshot: list[tuple[str, str]],
) -> dict:
    data = _subset(samples, positive_cats)
    print(f"\n{'=' * 72}\n{title}  ({len(data)} probek)\n{'=' * 72}")

    rule_pairs: list[tuple[int, int]] = []
    llm_pairs: list[tuple[int, int]] = []
    per_sample = []
    for s, truth in data:
        rd = rules.detect(s.text, allowed=rule_allowed)
        rpred = int(rd.is_manipulative)
        rule_pairs.append((truth, rpred))
        lpred = None
        if det and det.available:
            r = det.detect(s.text, system_prompt=system_prompt, fewshot=fewshot)
            lpred = int(bool(r.is_manipulative))
            llm_pairs.append((truth, lpred))
        per_sample.append((s, truth, rpred, lpred, rd.strength))

    rm = _score("Reguly", rule_pairs)
    print(f"[Reguly] {_fmt(rm)}")
    if llm_pairs:
        lm = _score("LLM", llm_pairs)
        print(f"[LLM]    {_fmt(lm)}")
        print(f"Delta F1 (LLM - reguly): {lm.f1 - rm.f1:+.0%}")
    return {"per_sample": per_sample, "rm": rm, "lm": _score("LLM", llm_pairs) if llm_pairs else None}


def run_hybrid(dark_result: dict) -> None:
    print(f"\n{'=' * 72}\nEKSPERYMENT B: HYBRYDA na zadaniu dark_pattern\n{'=' * 72}")
    rows = dark_result["per_sample"]
    if any(l is None for _, _, _, l, _ in rows):
        print("LLM niedostepny \u2014 pomijam hybryde.")
        return

    or_pairs, and_pairs, routed_pairs = [], [], []
    for _s, truth, r, l, strength in rows:
        or_pairs.append((truth, 1 if (r or l) else 0))
        and_pairs.append((truth, 1 if (r and l) else 0))
        # routed: silne trafienie regul -> ufaj regulom; inaczej glos LLM
        routed = r if strength >= 2 else l
        routed_pairs.append((truth, int(bool(routed))))

    print(f"[OR    ] {_fmt(_score('or', or_pairs))}")
    print(f"[AND   ] {_fmt(_score('and', and_pairs))}")
    print(f"[ROUTED] {_fmt(_score('routed', routed_pairs))}")
    print(f"[Reguly] {_fmt(dark_result['rm'])}  <- baseline do porownania")
    if dark_result["lm"]:
        print(f"[LLM   ] {_fmt(dark_result['lm'])}  <- sam LLM")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    samples = load_samples()
    det = None
    if not args.rules_only:
        print("Laduje model...")
        det = LlamaDetector(model_path=args.model)
        if not det.available:
            print(f"[UWAGA] Model niedostepny: {det.error}\n")

    t0 = time.time()
    print("\n### EKSPERYMENT A: ROZDZIELENIE ZADAN ###")
    dark = run_task(
        "ZADANIE 1/2: dark_pattern (presja/wina/oszustwo/bait)",
        samples, DARK_CATS, DARK_CATS, det, DARK_PROMPT, DARK_FEWSHOT,
    )
    run_task(
        "ZADANIE 2/2: slop (pusty wypelniacz vs tresc)",
        samples, SLOP_CATS, SLOP_CATS, det, SLOP_PROMPT, SLOP_FEWSHOT,
    )

    if det and det.available:
        run_hybrid(dark)
        print(f"\nCzas LLM lacznie: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
