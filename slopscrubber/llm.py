"""Detektor oparty o maly lokalny model (GGUF przez llama.cpp).

Backend jest wymienny i bezpieczny: jesli llama-cpp-python albo plik modelu
nie sa dostepne, detektor zwraca status 'unavailable' zamiast wysypywac caly
harness. Dzieki temu reszta kodu dziala tez na maszynie bez modelu.

Konfiguracja przez zmienne srodowiskowe:
  SLOP_MODEL_PATH   \u2014 sciezka do lokalnego pliku .gguf (ma pierwszenstwo)
  SLOP_MODEL_REPO   \u2014 repo HuggingFace (domyslnie Qwen2.5-1.5B-Instruct-GGUF)
  SLOP_MODEL_FILE   \u2014 nazwa pliku GGUF w repo
  SLOP_N_THREADS    \u2014 liczba watkow CPU
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = os.environ.get("SLOP_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
DEFAULT_FILE = os.environ.get("SLOP_MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")

CATEGORIES = [
    "fake_urgency",
    "confirmshaming",
    "clickbait",
    "aggressive_popup",
    "forced_continuity",
    "ai_slop",
    "clean",
]

SYSTEM_PROMPT = (
    "You are a strict detector of manipulative web content (dark patterns and AI slop). "
    "You judge the SEMANTICS, not keywords. Flag text as manipulative only if it actually "
    "pressures, guilt-trips, deceives, baits, or is content-free filler. Do NOT flag text "
    "that merely mentions or factually describes such things, states a real deadline fairly, "
    "or is a neutral UI label. Legitimate, transparent, information-dense text is clean.\n\n"
    "Categories: fake_urgency, confirmshaming, clickbait, aggressive_popup, forced_continuity, "
    "ai_slop, clean.\n"
    "Respond with ONLY a JSON object: "
    '{"manipulative": true|false, "category": "<one category>", "reason": "<=12 words"}'
)

FEWSHOT = [
    (
        "This article explains how fake countdown timers pressure shoppers into hasty decisions.",
        '{"manipulative": false, "category": "clean", "reason": "educational description, not itself manipulative"}',
    ),
    (
        "No thanks, I don't want to save money.",
        '{"manipulative": true, "category": "confirmshaming", "reason": "guilt-trips the decline"}',
    ),
    (
        "The 2024 model adds a 48MP sensor and a 4,500 mAh battery.",
        '{"manipulative": false, "category": "clean", "reason": "factual spec text"}',
    ),
    # Neutralne etykiety UI = clean (model mial tendencje do ich nadmiernego flagowania).
    (
        "Add to cart",
        '{"manipulative": false, "category": "clean", "reason": "neutral UI button"}',
    ),
    (
        "Unsubscribe",
        '{"manipulative": false, "category": "clean", "reason": "neutral link"}',
    ),
    # Pusty wypelniacz = ai_slop (model mial tendencje do uznawania go za clean).
    (
        "In today's fast-paced world, leveraging synergies unlocks a myriad of opportunities at the end of the day.",
        '{"manipulative": true, "category": "ai_slop", "reason": "content-free buzzword filler"}',
    ),
]


@dataclass(frozen=True)
class LlmResult:
    available: bool
    is_manipulative: bool | None
    category: str | None
    reason: str
    raw: str


class LlamaDetector:
    def __init__(self, model_path: str | None = None, n_threads: int | None = None, verbose: bool = False):
        self._llm = None
        self._error: str | None = None
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover - zalezne od srodowiska
            self._error = f"llama-cpp-python niedostepne: {exc}"
            return

        try:
            path = model_path or _resolve_model_path()
        except Exception as exc:
            self._error = f"Nie udalo sie uzyskac modelu: {exc}"
            return

        try:
            self._llm = Llama(
                model_path=str(path),
                n_ctx=2048,
                n_threads=n_threads or int(os.environ.get("SLOP_N_THREADS", os.cpu_count() or 4)),
                verbose=verbose,
            )
        except Exception as exc:  # pragma: no cover
            self._error = f"Nie udalo sie zaladowac modelu: {exc}"

    @property
    def available(self) -> bool:
        return self._llm is not None

    @property
    def error(self) -> str | None:
        return self._error

    def detect(self, text: str) -> LlmResult:
        if self._llm is None:
            return LlmResult(False, None, None, self._error or "unavailable", "")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_text, assistant_json in FEWSHOT:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_json})
        messages.append({"role": "user", "content": text})

        out = self._llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=96,
        )
        raw = out["choices"][0]["message"]["content"] or ""
        parsed = _parse_json_verdict(raw)
        if parsed is None:
            # Fallback: szukamy slowa manipulative/clean w surowej odpowiedzi.
            lowered = raw.lower()
            manip = "true" in lowered and "false" not in lowered.split("true")[0]
            return LlmResult(True, manip, None, "unparsed", raw)
        return LlmResult(
            available=True,
            is_manipulative=bool(parsed.get("manipulative")),
            category=parsed.get("category"),
            reason=str(parsed.get("reason", "")),
            raw=raw,
        )


def _resolve_model_path() -> Path:
    explicit = os.environ.get("SLOP_MODEL_PATH")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"SLOP_MODEL_PATH nie istnieje: {p}")
        return p

    from huggingface_hub import hf_hub_download  # type: ignore

    path = hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILE)
    return Path(path)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_json_verdict(raw: str) -> dict | None:
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    det = LlamaDetector()
    if not det.available:
        print(f"Model niedostepny: {det.error}")
    else:
        for t in [
            "Only 2 left \u2014 hurry, offer ends in 04:59!",
            "This article explains how fake countdown timers work.",
            "Add to cart",
        ]:
            r = det.detect(t)
            print(f"{r.is_manipulative!s:5} {r.category:16} {t[:50]}  ({r.reason})")
