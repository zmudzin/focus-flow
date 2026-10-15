# Slop Scrubber — prototyp walidacyjny (nie produkt)

To **nie jest** aplikacja. To mały harness eksperymentalny, który odpowiada na
jedno pytanie decydujące o sensowności całego pomysłu „Systemowego Neutralizatora
Manipulacji":

> **Czy mały, lokalny model (GGUF, 1–1.5B, Q4, na CPU) wykrywa manipulacyjne
> wzorce (dark patterns, AI slop) *lepiej* niż zwykły baseline regułowy —
> zwłaszcza na przypadkach, których reguły z natury nie łapią?**

Logika: lokalny LLM to jedyny realny wyróżnik produktu wobec darmowych narzędzi
(uBlock Origin, listy cookie, reader mode). Jeśli mały model nie bije reguł,
cały pomysł traci sens *zanim* zbudujesz Tauri, rozszerzenie czy aplikację na
Androida. To najtańszy możliwy test tej hipotezy.

## Co jest w środku

- `data/samples.jsonl` — 55 ręcznie oznaczonych próbek (EN + PL): fałszywe
  liczniki, confirmshaming, clickbait, agresywne pop-upy, forced continuity,
  AI slop oraz kontrole „czyste". 26 oznaczono jako **trudne dla reguł**
  (manipulacja bez słów-kluczy ORAZ czysty tekst zawierający słowa-pułapki).
- `slopscrubber/rules.py` — baseline regułowy (regex/heurystyki), reprezentuje
  „co osiągniesz bez AI".
- `slopscrubber/llm.py` — detektor na `llama-cpp-python` + GGUF (domyślnie
  Qwen2.5-1.5B-Instruct Q4). Backend wymienny, bezpieczny fallback gdy brak modelu.
- `slopscrubber/evaluate.py` — metryki (precision/recall/F1), podział na
  podzbiór „trudny", lista przypadków rozbieżnych i błędów każdego detektora.
- `run.py` — punkt wejścia.

## Uruchomienie

```bash
# tylko baseline regułowy (bez modelu, bez zależności):
python3 run.py --rules-only

# pełne porównanie reguły vs LLM (pobierze model GGUF przy pierwszym razie):
pip install -r requirements.txt
python3 run.py

# własny model:
python3 run.py --model /sciezka/do/model.gguf
```

Zmienne środowiskowe: `SLOP_MODEL_PATH`, `SLOP_MODEL_REPO`, `SLOP_MODEL_FILE`,
`SLOP_N_THREADS`.

## Wyniki pierwszego testu (Qwen2.5-1.5B-Instruct Q4, CPU, 4 wątki)

| Zestaw | Metryka | Reguły | LLM (prompt v1) | LLM (prompt v2) |
|---|---|---|---|---|
| Całość (55) | F1 | **85%** | 72% | 69% |
| Całość (55) | precision / recall | 86% / 83% | 68% / 77% | 76% / 63% |
| Trudne dla reguł (26) | F1 | **69%** | 62% | 48% |

Szybkość: ~1.1 s/próbkę na CPU (bez GPU).

### Wniosek (uczciwy i niewygodny)

Na tym zbiorze **mały lokalny model NIE bije reguł** — jest wyraźnie gorszy,
i to dokładnie na przypadkach „trudnych dla reguł", gdzie miał wygrać.
Poprawa promptu (v2) tylko przesunęła błędy (mniej fałszywych trafień, więcej
przeoczeń), nie zlikwidowała luki.

Zaobserwowane słabości modelu 1.5B:
- **Nadgorliwość** — flagował neutralne UI („Add to cart", „Unsubscribe",
  „Nie teraz") jako manipulację.
- **Ślepota na slop** — pusty wypełniacz AI uznawał za „czysty", bo nie jest
  „groźny"; to pokazuje, że *slop* i *dark pattern* to dwie różne osie, a nie
  jedna kategoria.

### Co to oznacza dla pomysłu

To **nie** jest ostateczny wyrok na pomysł, ale mocny sygnał ostrzegawczy:
teza „mały lokalny LLM = wyróżnik" nie broni się out-of-the-box. Zanim włożysz
miesiące w Tauri/rozszerzenie/Androida, warto najpierw sprawdzić, czy którakolwiek
z poniższych tanich zmian odwraca wynik:

1. **Większy model** (Qwen2.5-3B / Llama-3.2-3B Q4) — czy rozmiar zamyka lukę,
   kosztem RAM/latencji u użytkownika?
2. **Węższa definicja zadania** — rozdziel „dark pattern" od „slop"; model lepiej
   radzi sobie z jedną wąską osią niż z workiem siedmiu kategorii.
3. **Podejście hybrydowe** — reguły jako szybki pierwszy filtr, LLM tylko do
   rozstrzygania przypadków niejednoznacznych (tam, gdzie reguły mają niską
   pewność). To gra do mocnych stron obu.
4. **Większy, bardziej realistyczny zbiór** — 55 próbek to za mało na twarde
   wnioski; to pilot. Zbierz 200–500 realnych fragmentów ze stron.

Jeśli po tych eksperymentach model nadal nie bije reguł — pomysł w obecnej formie
(lokalny LLM jako wyróżnik) prawdopodobnie nie wypali, i oszczędzasz sobie miesiące.

## Eksperymenty A / B / C (druga tura)

Po pierwszym teście uruchomiono tanie eksperymenty w kolejności: najpierw darmowe
dźwignie diagnostyczne, dopiero na końcu większy model. Kod: `experiments.py`.
Surowe logi: `results/experiments.log` (1.5B) i `results/experiments_3b.log` (3B).

### A. Rozdzielenie zadań (dark pattern vs slop)

Hipoteza: „manipulacja" (presja/oszustwo) i „slop" (pusty wypełniacz) to dwie
różne osie; rozdzielenie powinno pomóc małemu modelowi.

| Zadanie | Model | Reguły F1 | LLM F1 |
|---|---|---|---|
| dark_pattern (49 próbek) | 1.5B | 85% | 70% |
| dark_pattern (49 próbek) | 3B | 85% | **86%** |
| slop (27 próbek) | 1.5B | 67% | 40% |
| slop (27 próbek) | 3B | 67% | 41% |

Wniosek: rozdzielenie **samo w sobie nie uratowało 1.5B**. Ale na zadaniu
**dark_pattern model 3B osiąga parytet/lekką przewagę** nad regułami.
**Slop to ślepa uliczka** dla małego lokalnego modelu: 1.5B przeocza, 3B flaguje
niemal wszystko (recall 100%, precision 26%). Slop lepiej zostawić regułom albo
w ogóle wyciąć z zakresu produktu.

### B. Hybryda (reguły + LLM) na zadaniu dark_pattern

| Wariant | Model | precision | recall | F1 |
|---|---|---|---|---|
| Reguły (baseline) | — | 88% | 82% | 85% |
| OR (union) | 3B | 80% | 86% | 83% |
| **AND (intersection)** | **3B** | **96%** | 82% | **88%** |
| ROUTED (silne reguły, reszta LLM) | 3B | 86% | 86% | 86% |

Wniosek: **hybryda AND (flaga tylko gdy reguły I LLM się zgadzają) jako jedyna
wyraźnie bije reguły** — F1 88% przy **96% precyzji**. Dla narzędzia, które
ukrywa/wycina treść i prosi o inwazyjne uprawnienia, wysoka precyzja (rzadkie
fałszywe wycięcia) jest ważniejsza niż recall — to realnie najcenniejszy wynik.

### C. Większy model (3B zamiast 1.5B)

To właśnie rozmiar modelu, nie prompt ani architektura, przesunął wynik z 70% do
86% na dark_pattern. Koszt: model ~2 GB (vs ~0.9 GB), ~1.5 s/próbkę na 4-rdz. CPU
bez GPU (vs ~1.1 s) i wyższe zużycie RAM. To dokładnie ta zmiana warunków
u użytkownika, przed którą ostrzegaliśmy.

### Zrewidowany werdykt

- Teza „lokalny LLM = wyróżnik" **przeżywa tylko w wąskiej formie**: 3B + hybryda
  AND, **wyłącznie dla dark patterns**, gdzie realnym atutem jest 96% precyzji.
- To przewaga **skromna** (+3 pkt F1 nad darmowymi regułami) kupiona **realnym
  kosztem** (cięższy model, wolniejsza inferencja, więcej RAM).
- **Slop lokalnie nie działa** — wytnij go z zakresu albo obsłuż regułami.
- To **nie** jest „najprostszy w produkcji, szybki zysk". Przewaga nad darmem jest
  mała i droga. Jeśli idziesz dalej, buduj wokół wąskiej obietnicy (precyzyjny,
  w pełni lokalny detektor dark patterns), nie wokół „neutralizatora wszystkiego".

(Zbiór to pilot \u2014 49/27 próbek; wyniki są kierunkowe, nie statystycznie twarde.)

## Ograniczenia

- Zbiór jest mały i ręcznie tworzony — służy do szybkiej iteracji, nie do
  publikowalnych wniosków statystycznych.
- Testowano jeden model w jednej kwantyzacji na CPU. Inne modele/kwantyzacje/
  prompty mogą dać inne wyniki — dlatego backend jest wymienny.
- „Prawda" (etykiety) odzwierciedla osąd autora o tym, co jest manipulacją;
  część przypadków jest z natury dyskusyjna (patrz pole `note` w próbkach).
