# ENRG-AI — Intelligence layer of the AXIS ecosystem

ENRG-AI is the **intelligence layer** of the AXIS protocol: a hybrid AI
oracle (anomaly detection, generation forecast, market signals), a
federated learning aggregator, and a coding agent over the ENRG / Axis
codebase.

> **Constitution:** the AI oracle is a **source of signals, not decisions**.
> It produces observations; the **Policy Engine** (ADR-0003) decides.

## Ecosystem

ENRG-AI is **Layer 3** of the Axis ecosystem — the intelligence layer: the AI
oracle produces **signals, not decisions** (constitution C-1), federated
learning trains the global model without data leaving devices, and the DAO
governs model evolution.
One map of all layers: [**Ecosystem map**](https://github.com/AntonGrid/Axis-protocol/blob/main/docs/ECOSYSTEM.md) ·
[**Constitution**](https://github.com/AntonGrid/Axis-protocol/blob/main/docs/CONSTITUTION.md) ·
[**Glossary**](https://github.com/AntonGrid/Axis-protocol/blob/main/docs/GLOSSARY.md).

| Layer | Repo |
|---|---|
| L0 Standard | [Axis-protocol](https://github.com/AntonGrid/Axis-protocol) |
| L1 Reference implementation | [Axis-core](https://github.com/AntonGrid/Axis-core) |
| L2 Domain profile (energy) | [ENRG](https://github.com/AntonGrid/ENRG) |
| **L3 Intelligence** | **ENRG-AI (this repo)** |
| L4 Interfaces | [enrg-landing](https://github.com/AntonGrid/enrg-landing) · [Axis-connect](https://github.com/AntonGrid/Axis-connect) |

## Layout

```
agent/
├── main.py            # entry point
├── cli/               # interactive shell + dispatcher
├── commands/          # user-facing commands (analyze, impact, ...)
├── core/              # intent detection, reasoning, prompts
├── index/             # symbol index over codebase + ranking
├── graph/             # call graph + impact analysis
├── knowledge/         # extraction of nodes (symbols / imports)
├── context/           # building LLM context from search results
├── db/                # SQLite knowledge base (files / symbols / imports)
├── reader/            # raw file reading
├── fed/               # federated learning (signed contributions, FedAvg)
├── hfl/               # hierarchical FL: regions → global (axis-fed/2)
├── digital_feeds/     # universal open-world data feeds
├── digital_train/     # domain-agnostic self-training (trends/anomalies/links)
├── market/            # normalized price feeds (USD/kWh) + price×time model
├── pilot/             # closed-loop DePIN simulation (device→model→action→reward)
├── multidomain/       # shared backbone + per-domain heads (transfer learning)
├── evolution/         # DAO-driven self-evolution (propose→vote→A/B→canonicalize)
└── llm.py             # Ollama client (qwen2.5-coder:7b)
tests/                 # pytest suite
```

## Proof-of-Intelligence (ERS economy & leaderboard)

PoI is the second earning channel of the ecosystem: reputation (ERS) and
influence for **quality federated contributions**, screened by MAD and weighted
by ERS (constitution C-6 — *quality has a price*).

```bash
python -m agent.leaderboard --rounds 5 --gateways 5   # ERS loop simulation
```

- `agent/fed/ers.py` — ERS update after each round: accepted gain, rejected
  outliers decay to a floor; ERS feeds back as the next round's sample weight.
- `agent/fed/digest.py` — contribution SHA-256 digest (base58) + signed
  `axis-fed-commit/1` commitment contract; the on-chain write stays
  oracle-only (ADR-0010).
- Tests: `tests/test_ers.py`, `tests/test_leaderboard.py`.

## Hybrid AI oracle (signals)

The AI oracle is a **source of signals, not decisions** (ADR-0003 / C-1). One
heartbeat collects uncertainty-aware observations across the ecosystem:

- `generation_forecast` — Wh per bucket + 80% interval (Holt trend, `agent.forecast`);
- `generation_anomaly` — last point outside the model's MAD band;
- `market_forecast` — USD/kWh next steps for dayahead/p2p/spot + 80% intervals.

```bash
python -m agent.signals --source offline --horizon 8            # deterministic demo
python -m agent.signals --source online  --horizon 6            # live oracle, graceful fallback
python -m agent.signals --output signals.json                   # JSON output
```

Bundles are Ed25519-signable (`sign_bundle` / `verify_bundle_signature`,
canonical JSON per `agent.fed.protocol`) so a Policy Engine can verify the
signal source. Tests: `tests/test_signals.py`.

## Energy generation forecasting

Live forecast from the real oracle proof stream (Wh per bucket, Holt linear
trend with horizon-widening prediction intervals, numpy only):

```bash
python -m agent.forecast                        # live oracle, 15-min buckets, 8 steps
python -m agent.forecast --bucket-minutes 10 --horizon 12
python -m agent.forecast --source offline --horizon 6 --output forecast.json
python -m agent.forecast --csv out.csv          # same CSV contract as the TimesFM skill
```

The CLI writes JSON + CSV; the CSV follows the `timesfm-forecasting` skill
contract (`../../skills/`), so the engine can be swapped to TimesFM / aeon on
a beefier machine without changing callers. Unit tests: `tests/test_forecast.py`.

## Digital self-training (background)

The digital layer learns on open-world signals — weather, finance, macro,
news, blockchain, science — with no domain hardcoding:

```bash
# one training cycle on deterministic series (no network)
python -m agent.digital_train.pipeline --once --offline --points 48

# background loop (online: real APIs/RSS/RPC, history accumulates in state)
python -m agent.digital_train.pipeline --loop --interval 3600 --state digital_state.json
```

The trained weights are shaped as a signed federated contribution
(`agent.fed`), ready to join the global model.

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m agent.main
```

## Tests

```bash
pytest
```

## Roadmap (Phase 0–1)

### Phase 0 — foundation ✅
- [x] Packaging: `pyproject.toml`, `requirements.txt`
- [x] Tests migrated to pytest (`tests/`)
- [x] `axis_core/ai/` scaffold — `SignalProvider` neutral stub (in Axis Core)
- [x] `projects/` — wired ENRG, enrg-landing, Axis-core (symlinks, see `scripts/link_projects.sh`)

### Phase 1 — hybrid AI oracle ✅ (in Axis Core)
- [x] Anomaly detection over device history (energy/power spikes, nonce jumps)
- [x] Production forecast (dependency-free linear model)
- [x] Policy gate `enforce_ai_anomaly` / `ai_anomaly_threshold` (opt-in)
- [x] Oracle integration — `ai_anomaly_flagged` reason, default behavior unchanged

### Phase 2 — federated learning ✅
- [x] `agent/fed/protocol.py` — Ed25519-signed contribution format (canonical JSON)
- [x] `agent/fed/local_train.py` — light model trained on the gateway
- [x] `agent/fed/aggregate.py` — FedAvg + signature verification + MAD outlier removal
- [x] `agent/fed/simulate.py` — N-gateway demo (`python -m agent.fed.simulate`)
- [x] Tests: `tests/test_fed_*.py` — ENRG-AI: 45 passed

### Phase 2.5 — hierarchical FL (HFL) ✅
- [x] `agent/hfl/protocol.py` — `axis-fed/2` (level/domain/region/version/quality signed)
- [x] `agent/hfl/region.py` — `RegionAggregator`: verify + MAD + reputation-weighted FedAvg → signed regional contribution
- [x] `agent/hfl/global_aggregator.py` — `GlobalAggregator`: regions → read-only global model
- [x] `agent/hfl/weights.py` — ERS reputation multipliers (floor for new devices)
- [x] `agent/hfl/simulate.py` — gateways → regions → world demo (`python -m agent.hfl.simulate`)
- [x] Tests: `tests/test_hfl.py` — ENRG-AI: 78 passed

### Phase 3 — market & action (step B) ✅
- [x] `agent/market/feeds.py` — normalized USD/kWh feeds (dayahead/p2p/spot/macro) + TTL cache + offline generators
- [x] `agent/market/model.py` — price×time matrix
- [x] `axis_core/ai/recommender.py` — SELL/STORE/BUY/HOLD ranking (confidence + rationale)
- [x] `PolicyEngine.evaluate_trade` — trade right / volume limits / DAO gate (Axis Core)
- [x] `POST /market/recommend` — end-to-end API (Axis Core)
- [x] Tests: ENRG-AI **84 passed**, Axis Core **141 passed**

### Phase 3 — closed-loop pilot (step C) ✅
- [x] `agent/pilot/sim.py` — full loop: device (solar + storage) → seasonal forecast
      (retrained in-loop) → market price → Recommender → `evaluate_trade` →
      action → USD reward → ERS/forecast feedback
- [x] Recommender learned to sell stored energy above the average price
      (premium SELL rule, Axis Core)
- [x] `python -m agent.pilot.sim` — 50 devices × 168 h demo
- [x] Tests: ENRG-AI **91 passed**, Axis Core **144 passed**

### Phase 3 — multi-domain + DAO evolution (step D) ✅
- [x] `agent/multidomain/` — `MultiDomainModel`: shared temporal backbone +
      per-domain heads; transfer learning (new domain fits only its head,
      few-shot) and failure isolation
- [x] `agent/evolution/` — staked DAO voting (quorum/majority, append-only
      ledger) → A/B arena on the pilot → canonicalize / roll back
- [x] Demo: `python -m agent.evolution.loop` — the system improved itself
      (canonicalized `min_sell_storage_ratio 0.1`, +2.2%) and rejected a bad
      experiment (aggressive premium, rolled back)
- [x] Tests: ENRG-AI **104 passed**

See `GLOBAL_AI_ARCHITECTURE.md` for the full vision.

