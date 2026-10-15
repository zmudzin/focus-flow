#!/usr/bin/env python3
"""Punkt wejscia prototypu walidacyjnego Slop Scrubber.

Uruchom:
    python run.py                 # reguly + LLM (pobierze model przy pierwszym razie)
    python run.py --rules-only    # tylko baseline regulowy, bez modelu
    python run.py --model /sciezka/do/model.gguf

Pytanie, na ktore odpowiada ten skrypt:
    Czy maly lokalny model wykrywa manipulacje lepiej niz reguly \u2014
    zwlaszcza na przypadkach TRUDNYCH DLA REGUL?
"""

from __future__ import annotations

import argparse
import sys
import time

from slopscrubber import rules
from slopscrubber.evaluate import Prediction, build_report
from slopscrubber.llm import LlamaDetector
from slopscrubber.samples import load_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Walidacja detekcji manipulacji: reguly vs maly LLM")
    parser.add_argument("--rules-only", action="store_true", help="pomin model, pokaz tylko baseline")
    parser.add_argument("--model", default=None, help="sciezka do pliku .gguf (nadpisuje env)")
    parser.add_argument("--data", default=None, help="sciezka do pliku samples.jsonl")
    args = parser.parse_args()

    samples = load_samples(args.data) if args.data else load_samples()
    pos = sum(s.label for s in samples)
    print(f"Wczytano {len(samples)} probek ({pos} manipulacyjnych, {len(samples) - pos} czystych).")

    rule_preds: list[Prediction] = []
    for s in samples:
        d = rules.detect(s.text)
        rule_preds.append(Prediction(s.id, s.label, int(d.is_manipulative), d.category))

    llm_preds = None
    if not args.rules_only:
        print("Inicjalizuje lokalny model (pierwsze uruchomienie moze pobierac plik GGUF)...")
        det = LlamaDetector(model_path=args.model)
        if not det.available:
            print(f"\n[UWAGA] Model niedostepny: {det.error}")
            print("Pokazuje same wyniki regulowe. Zainstaluj zaleznosci i podaj model, aby porownac.\n")
        else:
            llm_preds = []
            t0 = time.time()
            for i, s in enumerate(samples, 1):
                r = det.detect(s.text)
                pred = 1 if r.is_manipulative else 0
                llm_preds.append(Prediction(s.id, s.label, pred, r.category))
                print(f"  LLM {i:>2}/{len(samples)}  {'M' if pred else 'c'}  {s.text[:55]}")
            dt = time.time() - t0
            print(f"\nLLM: {len(samples)} probek w {dt:.1f}s  (~{dt / len(samples) * 1000:.0f} ms/probke na CPU)")

    print()
    print(build_report(samples, rule_preds, llm_preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
