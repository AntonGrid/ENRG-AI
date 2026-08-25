# ENRG-AI — Intelligence layer of the AXIS ecosystem

ENRG-AI is the **intelligence layer** of the AXIS protocol: a hybrid AI
oracle (anomaly detection, generation forecast, market signals), a
federated learning aggregator, and a coding agent over the ENRG / Axis
codebase.

> **Constitution:** the AI oracle is a **source of signals, not decisions**.
> It produces observations; the **Policy Engine** (ADR-0003) decides.

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
├── digital_feeds/     # universal open-world data feeds
├── digital_train/     # domain-agnostic self-training (trends/anomalies/links)
└── llm.py             # Ollama client (qwen2.5-coder:7b)
tests/                 # pytest suite
```

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

See `ANALYSIS_AND_PLAN.md` for the full plan (phase 3).

