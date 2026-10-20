# `agent/physics` — парк как одна среда

Слой L3 оракула ENRG-AI. Каждый производитель — осциллятор, состояние которого
есть **частота**; то, что у двух производителей общего, записывается в **связь**
между ними. Отсюда четыре чтения на производителя: `silence`, `resonance`,
`phase`, `memory` → `score ∈ [0, 1]`.

Математика — порт `entrain/julia/kuramoto/basic.jl`, сверенный с Julia **до
последнего бита** (`tests/fixtures/entrain_reference.json`). Граница «что этот
слой делает, а чего не делает никогда» — `entrain/docs/ENTRAIN-BOUNDARY.md`;
история интеграции — `docs/PHYSICS_INTEGRATION_SUCCESS.md`.

## Что здесь лежит

| файл | что делает |
|---|---|
| `oscillator.py` | 11 функций книги: чистая математика, без I/O |
| `park.py` | чтение парка: демодуляция суток, частоты, связи, четыре чтения |
| `series.py` | пруфы оракула → матрица «производители × часы» (единственный ввод) |
| `synthetic.py` | детерминированный парк + четыре объявленных изменения |
| `signal.py` | `CoherenceSignal` и `SignalBundle` (`kind="coherence"`) |
| `benchmark.py` | сравнение с MAD-правилами на объявленной истине |
| `__main__.py` | CLI: читает парк и пишет `coherence.json` |

## Одиннадцать функций

Все — из `entrain/julia/kuramoto/basic.jl`; номера строк указаны, чтобы сверка
была построчной.

| функция | что читает | в книге |
|---|---|---|
| `overlap(delta, band)` | `S(Δ) = 1 / (1 + (Δ/Π)²)` — насколько совпали две частоты | `basic.jl:243` |
| `coherence(theta)` | `R = \|mean e^{iθ}\|` — согласованность (порядок) | `basic.jl:378` |
| `partial_coherence(theta, mode)` | тот же `R`, но по одному голосу аккорда | `basic.jl:381` |
| `resonance(omega)` | `max ω − min ω` — ширина всей среды | `basic.jl:408` |
| `spread(omega, index)` | ширина аккорда одного вибратора | `basic.jl:415` |
| `chord(omega, theta, t)` | `mean cos(ωt + θ)` одного аккорда в момент `t` | `basic.jl:393` |
| `wave(omega, theta, t)` | `mean cos(ωt + θ)` по всей среде | `basic.jl:400` |
| `unison(omega, omega0)` | `1 − spread/spread₀` — насколько аккорд сжался | `basic.jl:423` |
| `coincidence(theta, index)` | `mean cos(θp − θq)` внутри аккорда | `basic.jl:436` |
| `memory_goal(link, K, acc, weight, current, reference)` | цель записанной связи `(link + N₀·cos)/K` | `basic.jl:312` |
| `memory_step(coupling, target, eps, dt)` | один шаг записи `W += dt·ε·(target − W)` | `basic.jl:358` |
| `step_state(...)` | полный шаг закона: `ω`, `θ`, `W` вместе | `step!` |

Чтение парка (`park.py`) использует `overlap`, `coherence` и `memory_goal`;
остальные перенесены целиком, потому что порт либо полный, или его нет:
частичный порт нельзя сверить с эталоном построчно.

## Быстрый старт (CLI)

Из каталога `ENRG-AI` (нужен интерпретатор, где установлены `numpy` и сам
пакет — например `.venv/bin/python`):

```bash
# парк с четырьмя объявленными изменениями → coherence.json
python -m agent.physics --scenario mixed --output coherence.json

# тот же парк без изменений: чтение должно молчать
python -m agent.physics --scenario clean

# живой оракул; если он недоступен или в нём один производитель —
# тихий откат на офлайн-парк, source="offline-fallback" + warning
python -m agent.physics --source online --output coherence.json

# подписать Ed25519 (ключ — base64 seed в переменной окружения)
AXIS_AI_SIGNING_KEY="$(cat key.b64)" python -m agent.physics --sign \
    --output coherence.json

# сколько на самом деле ловит физика и что ловят правила
python -m agent.physics.benchmark
python -m agent.physics.benchmark --seed 11 --scenario clean
```

Вывод `python -m agent.physics.benchmark` (сид 7):

```
device     truth          coherence  spike_mad spike_mad_daily  trend_mad
-------------------------------------------------------------------------
dev-00     clean                  .          X          .          X
dev-02     outage                 X          X          .          X
dev-05     peak_shift             X          X          .          X
dev-06     clock_offset           X          X          .          X
dev-07     level_scale            .          X          .          X

coherence       caught 3/4  precision 1.00  recall 0.75  false alarms 0
spike_mad       caught 4/4  precision 0.50  recall 1.00  false alarms 4
spike_mad_daily caught 0/4  precision n/a   recall 0.00  false alarms 0
trend_mad       caught 4/4  precision 0.57  recall 1.00  false alarms 3
park coherence 0.916 (min 0.852), silent 0.035
```

Вывод `python -m agent.physics --scenario mixed`:

```
source            : offline
park              : 8 producers, 72 x 60-min buckets read
coherence         : mean 0.9156 min 0.8523 silent 0.0347

device        score  verdict  readings
------------------------------------------------------------------------------
dev-00        0.249  ok       -
dev-02        1.000  ANOMALY  silence
dev-05        1.000  ANOMALY  phase,memory
dev-06        0.634  ANOMALY  phase
dev-07        0.179  ok       -
```

### Флаги

| флаг | по умолчанию | смысл |
|---|---|---|
| `--source` | `offline` | `offline` — синтетический парк, `online` — пруфы оракула |
| `--scenario` | `mixed` | `mixed` — четыре изменения, `clean` — исправный парк |
| `--seed` | `7` | сид синтетики; на 7/11/23/42 вердикт один и тот же |
| `--bucket-minutes` | `60` | ширина корзины; 60 при суточном обороте = 24 отсчёта |
| `--window` | `0` | прочитать только последние N корзин (0 — всю историю) |
| `--baseline-hours` | `96` | сколько часов «нормы» парк выучивает до чтения |
| `--band` | `0.05` | Π: полоса, в которой частоты считаются совпавшими (рад/ч) |
| `--memory-hours` | `12.5` | время жизни записанной связи (`1/ε`) |
| `--threshold` | `0.5` | порог score, выше которого производитель помечен |
| `--limit` | `1000` | сколько пруфов забрать у оракула (`--source online`) |
| `--output` | — | путь для JSON; без него только печать |
| `--sign` | выкл. | подписать конверт `AXIS_AI_SIGNING_KEY` |

## Что внутри `coherence.json`

Это обычный `SignalBundle` оракула — тот же конверт, что подписывает
`agent.signals`; новый здесь только `kind`. Без `--sign` — три ключа,
с `--sign` — конверт `{message, signature, public_key}`.

```json
{
  "generated_at": "2026-10-20T11:00:00+00:00",
  "meta": {
    "source": "offline",
    "physics": {
      "engine": "entrain",
      "source_file": "entrain/julia/kuramoto/basic.jl",
      "ported": ["overlap", "coherence", "partial_coherence", "resonance",
                 "spread", "chord", "wave", "coincidence", "memory_goal",
                 "memory_step", "step_state"],
      "components": ["silence", "resonance", "phase", "memory"],
      "config": {"bucket_minutes": 60, "band_rad_per_hour": 0.05,
                 "memory_hours": 12.5, "baseline_hours": 96.0,
                 "threshold": 0.5, "...": "..."}
    },
    "park": {"coherence_mean": 0.9156, "coherence_min": 0.8523,
             "silent_fraction": 0.0347, "producers": 8, "window_buckets": 72},
    "devices": [
      {
        "device_id": "dev-00",
        "score": 0.249,
        "is_anomaly": false,
        "reasons": [],
        "components": {"silence": 0.0, "resonance": 0.0,
                       "phase": 0.0596, "memory": 0.249},
        "readings": {"silent_fraction": 0.0, "phase_drift_rad": 0.04682,
                     "memory_divergence": 0.099619, "resonance_s": 0.999999,
                     "detune_rad_per_hour": 3.7e-05,
                     "period_hours": 23.976646, "coherence": 0.915588},
        "anomaly_since": null
      }
    ],
    "warnings": []
  },
  "signals": [
    {
      "kind": "coherence",
      "domain": "pilot",
      "ts": "2026-10-20T11:00+00:00",
      "source": "offline",
      "value": 1.0,
      "unit": "score",
      "meta": {"device_id": "dev-02", "is_anomaly": true,
               "reasons": ["silence"], "components": {"...": "..."},
               "readings": {"...": "..."},
               "anomaly_since": "2026-10-18T23:00+00:00"}
    }
  ]
}
```

Два правила, которые стоит знать заранее:

- `signals` содержит **только помеченных** производителей (провод молчит), но
  `meta.devices` содержит **всех** — чтение никого не выбрасывает, оно просто
  не всех произносит вслух;
- `readings` — сырые величины, из которых сложился score: `phase_drift_rad`
  (сдвиг суточного пика в радианах), `period_hours` (собственный период
  производителя), `resonance_s` (`S(Δω)` — насколько его частота совпала с
  парком), `memory_divergence` (насколько разошлась записанная связь).

## Проверка подписи

```python
import json
from agent.signals import verify_bundle_signature

payload = json.load(open("coherence.json"))          # созданный с --sign
assert verify_bundle_signature(payload["public_key"], payload)   # True

payload["message"]["meta"]["devices"][0]["score"] = 0.0   # любая подмена
assert not verify_bundle_signature(payload["public_key"], payload)  # False
```

Формат не новый — проверка та же, что у остальных сигналов оракула
(`agent.signals.verify_bundle_signature`, канонический JSON из
`agent.fed.protocol`).

## Использование из Python

```python
from agent.forecast.energy import fetch_oracle_proofs
from agent.physics import (
    ParkConfig, coherence_bundle, observe_park, park_matrix, park_signals,
)

proofs = fetch_oracle_proofs(limit=1000)              # пруфы оракула (ADR-0010)
matrix = park_matrix(proofs, bucket_minutes=60)       # производители × часы
observation = observe_park(
    matrix.energy, matrix.device_ids, matrix.starts, ParkConfig(bucket_minutes=60)
)

observation.park_reading()   # {"coherence_mean": ..., "silent_fraction": ...}
observation.rows()           # все производители: score, readings, reasons
park_signals(observation)    # [CoherenceSignal, ...] — на каждого
bundle = coherence_bundle(observation, source="oracle")   # SignalBundle
```

Дальше сигнал идёт в существующий opt-in гейт политики (Axis-core), без нового
флага и без нового кода причины:

```python
from axis_core.policy import engine, models

signal = next(s for s in park_signals(observation) if s.is_anomaly)
engine.PolicyEngine.evaluate_ai_anomaly(
    policy=models.PolicyRegistry(), signal=signal
).allowed                                                   # True — флаг выключен
engine.PolicyEngine.evaluate_ai_anomaly(
    policy=models.PolicyRegistry(enforce_ai_anomaly=True), signal=signal
).reason                                                   # "ai_anomaly_flagged"
```

`coherence_bundle` возвращает обычный `SignalBundle`: `sign_bundle` и
`verify_bundle_signature` работают с ним без правок.

## Тесты и эталон Julia

```bash
python -m pytest tests/test_physics_oscillator.py   # порт против эталона Julia
python -m pytest tests/test_physics_park.py         # демодуляция, частоты, связи
python -m pytest tests/test_physics_signal.py       # сигналы, подпись, гейт
python -m pytest tests/test_physics_benchmark.py    # физика против MAD
```

Тесты бенчмарка и гейта помечены `pytest.importorskip("axis_core...")`: без
установленного `Axis-core` они аккуратно пропускаются, а не падают.

Перегенерировать эталон (нужен Julia и рядом лежащий репозиторий `entrain`):

```bash
julia --startup-file=no tests/fixtures/entrain_reference.jl \
      > tests/fixtures/entrain_reference.json
# другой checkout книги:
ENTRAIN_JL=/path/to/basic.jl julia --startup-file=no tests/fixtures/entrain_reference.jl
```

`tests/test_physics_oscillator.py` сравнивает **все** одиннадцать функций и три
шага закона с расхождением `0.0`, а не «в пределах допуска»: порт должен
совпадать побитово, иначе это уже не порт.

## Что слой не делает

- **Не видит уровень.** Множитель `×k` фазу не двигает: изменение уровня —
  работа правил (Axis-core), не физики.
- **Не заменяет подпись.** Согласованность не доказывает авторство; подпись не
  доказывает согласованность.
- **Ничего не разрешает.** Ни минта, ни начисления, ни расчёта ERS: только
  наблюдение для оператора (и opt-in гейт, если оператор его включил).
- **Не работает без истории и без соседей.** Нужны ≥ 2 производителя, один
  оборот (сутки) и базовая линия (`--baseline-hours`, по умолчанию 96 ч).
- **Это чтение отчётов, а не измерение физики.** Парк, который врёт
  согласованно, остаётся невидимым — свойство метода, записано в границе.



