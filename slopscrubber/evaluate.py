"""Metryki i raport porownawczy: reguly vs LLM."""

from __future__ import annotations

from dataclasses import dataclass

from .samples import Sample


@dataclass
class Prediction:
    sample_id: str
    truth: int
    predicted: int  # 1 = manipulacyjny, 0 = czysty
    category_pred: str | None = None


@dataclass
class Metrics:
    name: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, truth: int, predicted: int) -> None:
        if truth == 1 and predicted == 1:
            self.tp += 1
        elif truth == 0 and predicted == 1:
            self.fp += 1
        elif truth == 0 and predicted == 0:
            self.tn += 1
        else:
            self.fn += 1


def score(name: str, preds: list[Prediction]) -> Metrics:
    m = Metrics(name)
    for p in preds:
        m.add(p.truth, p.predicted)
    return m


def _fmt_metrics(m: Metrics) -> str:
    return (
        f"  acc={m.accuracy:.0%}  precision={m.precision:.0%}  "
        f"recall={m.recall:.0%}  F1={m.f1:.0%}  "
        f"(TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn})"
    )


def build_report(
    samples: list[Sample],
    rule_preds: list[Prediction],
    llm_preds: list[Prediction] | None,
) -> str:
    lines: list[str] = []
    by_id = {s.id: s for s in samples}
    rp = {p.sample_id: p for p in rule_preds}
    lp = {p.sample_id: p for p in (llm_preds or [])}

    lines.append("=" * 72)
    lines.append("WYNIKI OGOLNE")
    lines.append("=" * 72)
    rm = score("Reguly", rule_preds)
    lines.append(f"[Reguly]{_fmt_metrics(rm)}")
    if llm_preds:
        lm = score("LLM", llm_preds)
        lines.append(f"[LLM]   {_fmt_metrics(lm)}")
        lines.append("")
        lines.append(f"Delta F1 (LLM - reguly): {lm.f1 - rm.f1:+.0%}")
    else:
        lines.append("[LLM]   niedostepny \u2014 pomijam porownanie.")

    # Podzbior 'trudne dla regul' \u2014 tu lezy cala teza projektu.
    hard = [s for s in samples if s.hard_for_rules]
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"PODZBIOR 'TRUDNE DLA REGUL' ({len(hard)} probek) \u2014 tu LLM musi wygrac")
    lines.append("=" * 72)
    rm_hard = score("Reguly(hard)", [rp[s.id] for s in hard])
    lines.append(f"[Reguly]{_fmt_metrics(rm_hard)}")
    if llm_preds:
        lm_hard = score("LLM(hard)", [lp[s.id] for s in hard])
        lines.append(f"[LLM]   {_fmt_metrics(lm_hard)}")
        lines.append("")
        lines.append(f"Delta F1 na trudnych (LLM - reguly): {lm_hard.f1 - rm_hard.f1:+.0%}")

    # Przypadki rozbiezne.
    if llm_preds:
        lines.append("")
        lines.append("=" * 72)
        lines.append("PRZYPADKI ROZBIEZNE (reguly != LLM)")
        lines.append("=" * 72)
        any_diff = False
        for s in samples:
            r = rp[s.id].predicted
            l = lp[s.id].predicted
            if r != l:
                any_diff = True
                truth = "MANIP" if s.label else "clean"
                who_right = "LLM" if l == s.label else "REGULY"
                lines.append(
                    f"[{s.id}] truth={truth:5} reguly={'M' if r else 'c'} "
                    f"llm={'M' if l else 'c'}  -> ma racje: {who_right}"
                )
                lines.append(f"        {s.text[:90]}")
        if not any_diff:
            lines.append("  (brak rozbieznosci)")

    # Bledy kazdego detektora osobno.
    lines.append("")
    lines.append("=" * 72)
    lines.append("BLEDY REGUL (FP = nadgorliwosc, FN = przeoczenia)")
    lines.append("=" * 72)
    _list_errors(lines, samples, rp)
    if llm_preds:
        lines.append("")
        lines.append("=" * 72)
        lines.append("BLEDY LLM (FP = nadgorliwosc, FN = przeoczenia)")
        lines.append("=" * 72)
        _list_errors(lines, samples, lp)

    return "\n".join(lines)


def _list_errors(lines: list[str], samples: list[Sample], preds: dict[str, Prediction]) -> None:
    had = False
    for s in samples:
        p = preds.get(s.id)
        if p is None:
            continue
        if p.predicted != s.label:
            had = True
            kind = "FP" if (s.label == 0 and p.predicted == 1) else "FN"
            lines.append(f"  {kind} [{s.id}] {s.category:16} {s.text[:70]}")
    if not had:
        lines.append("  (brak bledow)")
