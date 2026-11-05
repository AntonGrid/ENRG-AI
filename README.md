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

## Closed economic loop (Phase 3)

The loop connects the layers that already exist: `agent.signals` (forecast +
market price) → `axis_core.ai.recommender` (SELL/STORE/BUY/HOLD ranking) →
`axis_core.policy.engine.evaluate_trade` (trading right, volume limits, DAO
gate) → simulated reward → ERS / experience.

```bash
python -m agent.trade --rounds 4                      # default loop
python -m agent.trade --dao-gated                     # DAO gate blocks every action
python -m agent.trade --capacity-wh 500 --storage-init-wh 495
```

Requires the `axis-core` reference implementation on `PYTHONPATH`
(`pip install -e ../Axis-core`). Tests: `tests/test_trade.py`.

## Proof-of-Intelligence (ERS economy & leaderboard)

PoI is the second earning channel of the ecosystem: reputation (ERS) and
influence for **quality federated contributions**, screened by MAD and weighted
by ERS (constitution C-6 — *quality has a price*).

```bash
python -m agent.leaderboard --rounds 5 --gateways 5   # ERS loop simulation
```

- `agent/fed/ers.py` — ERS update after each round: accepted gain, rejected
  outliers decay to a floor; ERS feeds back as the next round's sample weight.
- `agent/fed/ers_loop.py` — AI anomaly signals → on-chain `report_anomaly`
  severity (mirrors the ENRG program's `apply_anomaly_penalty`).
- `agent/fed/digest.py` — contribution SHA-256 digest (base58) + signed
  `axis-fed-commit/1` commitment contract; the on-chain write stays
  oracle-only (ADR-0010).
- `agent/fed/transport.py` — **real FL transport** (audit P3-4): a stdlib HTTP
  aggregator that verifies every Ed25519-signed contribution, collects the
  round, runs FedAvg + MAD screening and serves the global weights; an httpx
  client (`submit_contribution` / `close_round` / `fetch_weights`). Tests:
  `tests/test_fed_transport.py`.
- Tests: `tests/test_ers.py`, `tests/test_ers_loop.py`, `tests/test_leaderboard.py`.

### Live AI-oracle attestation (setup)

The landing widget reads a **signed** attestation from
`https://antongrid.github.io/ENRG-AI/ai/assessments.json`. To make it LIVE:

```bash
# 1. One-time: enable Pages + create the signing secret (needs a token):
GITHUB_TOKEN=ghp_xxx .venv/bin/python scripts/setup_pages.py --key <base64-seed>
# 2. Then run the workflow once, or let the cron (30 min) do it:
#    https://github.com/AntonGrid/ENRG-AI/actions/workflows/signals.yml
```

Manual path: Settings → Pages → branch `gh-pages`; Settings → Secrets →
Actions → `AXIS_AI_SIGNING_KEY` (base64 Ed25519 seed).

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

## Coherence physics (entrain, L3)

A park of producers is read as one medium: each producer is an oscillator whose
state is a frequency, and what two producers have in common is written into the
link between them. The maths is a bit-exact port of the oscillator book's
reference implementation (`entrain/julia/kuramoto/basic.jl`), so every number can
be compared with Julia one by one.

```bash
python -m agent.physics --scenario mixed --output coherence.json  # a week with 4 declared changes
python -m agent.physics --scenario clean                          # a park that is fine
python -m agent.physics --source online --output coherence.json   # live oracle, graceful fallback
python -m agent.physics.benchmark                                 # this layer vs the MAD rules
```

Four readings per producer — `silence`, `resonance` (`S(Δω)`), `phase` (`Δθ`
against the park's own drift), `memory` (the written link `W`) — combine into a
`score ∈ [0, 1]`. The reading is a day's reading: a phase is demodulated over
exactly one revolution, so a value at `t` is what the last day sounded like.

Signals leave in the oracle's own `SignalBundle` (`kind="coherence"`), so
`sign_bundle` / `verify_bundle_signature` work unchanged, and `CoherenceSignal`
fits the existing opt-in gate `PolicyEngine.evaluate_ai_anomaly` — no new flag,
no new reason code, no protocol change.

The layer is **blind to level by construction** (a factor does not move a phase)
and quiet in a quiet park; the measured matrix against the MAD rules is in
`tests/test_physics_benchmark.py`. The boundary with the protocol is written
down in `../entrain/docs/ENTRAIN-BOUNDARY.md`.

How to run it, what the 11 functions are and what is inside `coherence.json`:
[`agent/physics/README.md`](agent/physics/README.md). For an outside reader
(investor, partner, auditor): [`docs/PHYSICS_INTEGRATION_SUCCESS.md`](docs/PHYSICS_INTEGRATION_SUCCESS.md).

Tests: `tests/test_physics_oscillator.py` (the port against a Julia fixture),
`tests/test_physics_park.py`, `tests/test_physics_signal.py`,
`tests/test_physics_benchmark.py`.

## Statement #87: is the relief too perfect? (sterility index)

`agent.physics.sterility` ports Statements #87 and #88 of the book
(`entrain/julia/kuramoto/consistency.jl`) and compares with the Julia run in
`tests/fixtures/sterility_reference.json`:

- **#87 — the sterility index.** Residuals of a memory relief against the
  theoretical decay, `δ_t = relief(t) − R₀·e^(−εt)`, are normalised and read
  two ways: Anderson–Darling `A²` (a lived relief leaves non-normal residuals,
  a formula leaves none, so `A² = 0` is *caught*) and `CV(relief)` (below 0.5 %
  the spread is under the physical floor the written 1 % miss puts there). Fail
  — `A² < 0.1` or `CV < 0.005` — and the reading is
  `synthetic_collusion_detected`; the Axis-core Policy Engine can block a mint
  on that reading (`enforce_sterility`, opt-in, see
  `Axis-core/docs/policy-engine.md`).
- **#88 — why the λ\* iteration has a limit.** `smooth_max` is the LogSumExp
  smoothing of the binding requirement, `banach_iteration` is the iteration
  with Banach's estimate read next to the limit.

The picture Anton asked for — counter, coherence and relief against
`e^(−εt)` on one page:

```python
from agent.physics import plot_depin_dynamics

plot_depin_dynamics(timestamps, energy_wh, R, share_spread, relief_trace,
                    memory=1e-3, path="depin_test_plot.png")
```

`matplotlib` is optional: `pip install 'enrg-ai[plot]'`. Tests:
`tests/test_physics_sterility.py`, `tests/test_physics_plots.py`.

## Curiosity: which stream deserves the next look (L3)

Above the park sits the body from `agent.physics.organism` — sensorium, homeostasis,
mirror. The mirror answers *"is this the world, or my own emission?"*. The next
question is cheaper and harder: **which stream is worth another look**, when looking
at all of them at once is not possible. `agent/physics/curiosity.py` answers it with
an expected information gain in four measured factors, each of which can be printed
and disputed on its own:

```
curiosity = reality(R) × 2V/(V + V_ref) × (1 − e^(−Λ/0.25)) × 1/(1 + 0.5·looks)
```

Every factor is a gate, not a bonus: a stream that does not move costs exactly zero
even if it is real, and a stream that moves but does not explain *itself* costs zero
even if it moves more than anything else. Structure is tested against the exact null
`P(R² > r) = (1 − r)^((n−2)/2)` with a Bonferroni correction over lags — measured on
pure noise it fires 0/120 at any α from 0.5 down to 0.01, and 0 of 33 no-repeat
streams (permutations, a random walk, AR(1), a frozen sensor) fire at all.

Measured on the offline sensorium, with no label anywhere: a forgery (a block of a
real feed re-served) ranks **0.815** and a replay **0.807**, while a control copy of
the very same bytes ranks **0.000** — the difference is the repeat, not the
provenance. A frozen sensor ranks 0.000. The organism then asks six questions
(*where*, *what next*, *what is it really*, twice) and stops because curiosity
crossed the threshold (0.815 → 0.546 → 0.410 → 0.326), not because a counter ran out.

```bash
python -m agent.physics.organism --offline --explore --curiosity-plot curiosity.png
```

The limitations are measured, not disclaimed: the detector catches *the same window
again* — the question a forgery actually asks — and does not catch periodicity of a
window summary, nor a forgery shorter than two fixations. Demo:
`notebooks/curious_agent.ipynb` (its rule is declared before the run). Details and
numbers: [`agent/physics/README.md`](agent/physics/README.md). Tests:
`tests/test_physics_curiosity.py`, `tests/test_physics_plots.py`.

## The economic agent: a verdict with a price (L3)

A body that is never hungry never learns to be careful. `agent/physics/economy_sim.py`
is the market the body works in: three streams per contract, one of them synthetic,
**50 credits** for a correct verdict, **30** for a wrong one, and **10 credits of
rent every four iterations** whether or not there is work. Which kind of synthetic
is public; which *stream* is not — the `Contract` object does not carry the answer,
and the truth surfaces only in the settlement, after the verdict is paid for.

Four kinds sit in the mix: `repeat` (a block re-served) and `replay` (a half copied)
are solvable by the structure arm; `surrogate` (phase-randomised, same spectrum) and
`noise` are not, and the whole exercise is discovering which half is which. The
penalty is 30 rather than Anton's 20 for an arithmetic reason: at 50/20 and three
streams a blind guess is worth `+3.33` per contract, so "take everything and guess"
would be the optimal policy and a refusal would be an error rather than caution.
At 50/30 the same guess is `−3.33` and the break-even accuracy is `30/80 = 0.375`.

`agent/physics/worker.py` is the decision. Its confidence is a blend of what this
contract's survey says (`Certainty.probability`, with the epistemic weight being how
often the answer survives five perturbations of the data) and a *pessimistic* bound
on the kind's record (`EpisodicMemory.belief`: the mean minus one standard error).
It bids when `p ≥ 0.375`, refuses when it does not, and buys at most two paid
lessons per kind while the balance can absorb the fine. Both banks are compared
against the policies that do not think: refuse everything dies of rent, guess
everything dies of fines.

Measured, seed 11, 120 iterations: 65 taken, 62 won (95%), 55 refused, the 55
refusals saved **1050 credits** of fines that would have been charged, and the
balance ends at **2760** from a start of 50. The surrogate and noise kinds are
taken twice each — the two paid lessons — and refused forever after; the replay
kind is taken 19 times and won 18.

```bash
python -m agent.physics.worker --iterations 120 --plot economy.png
```

Demo: `notebooks/economic_agent.ipynb` (balance, refusals that saved credits, UCB
and the attention share; the acceptance rule is asserted in the last cell). Tests:
`tests/test_physics_economy.py`, `tests/test_physics_worker.py`. No exchange, no
wallet, no key, no network: credits exist as one float per simulation.

## The digital entrepreneur: capital, a trap and a switch (L4)

The worker earned one contract at a time. L4 asks what a body with a balance does
*between* jobs, and the answer is a cap rather than a forecast:
`agent/physics/portfolio.py` splits the empire **60 / 30 / 10** across a safe
staking vault, an active bounty/MEV book and an experimental farm — so a protocol
that goes to zero costs a tenth of the empire — and refuses to send a credit to any
address the operator has not approved. The whitelist is necessary and *not*
sufficient: every scam in the simulation is on it, because an approved address says
who runs it and nothing about whether its curve is a record.

`agent/physics/risk_assessment.py` answers the second question. Statement #87 is
applied to the return curve itself — a lived record leaves non-normal residuals, a
formula leaves none — and the same `reducibility` that names a re-served contract in
`worker.py` names a re-served record here. Neither arm can convict alone, and that is
measured rather than argued: a *bond* is sterile too (`CV = 1e-4`), so a sterile
record with a plausible claim is refused as `SYNTHETIC_COLLUSION` while the same
sterility under an impossible claim is a `SCAM_TRAP`. The agent's kill switch watches
three things — the 15 % cycle drawdown the brief asked for, the market's volatility
against its own history, and the mirror's `internal_resonance` — and takes the risky
buckets to cash when any of them fires.

Measured on the declared market (720 steps, seed 7, one body): **100 → 149 credits**,
five protocols refused at every one of 28 reviews, both traps convicted *before* they
stopped paying (YieldMax at step 280 by sterility, AurumDAO at 336 by `R² = 1.00`), the
switch fired on the drawdown at 403 (15.6 % in a cycle) and on the volatility at 475,
and the worst fall from the running peak was **15.9 %** against **59.9 %** for the same
caps without an assessment and **100 %** for "everything into the highest claim". The
price of caution is printed too: refusing the sterile bond cost 18 credits of forgone
yield.

```bash
python -m agent.physics.portfolio --steps 720 --body --against --plot empire.png
```

Demo: `notebooks/digital_entrepreneur.ipynb` (capital against two credulous policies,
the moment of truth, and the switch working; the acceptance rule is asserted in the
last cell). Tests: `tests/test_physics_portfolio.py`,
`tests/test_physics_risk_assessment.py`. No key, no signature, no RPC: an address is
`0x` plus forty hex digits of a SHA-256 of the protocol's name, and the venue is
swappable for a real Web3 adapter behind `ProtocolMarket`.


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

