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
`0x` plus forty hex digits of a SHA-256 of the protocol's name.

### The adapter: one line between the logic and the source (Phase 4)

`agent/physics/interfaces.py` is the seam. `IWeb3Provider` is five operations —
`get_protocol_list`, `fetch_protocol_history`, `submit_validation_proof`,
`execute_stake`, `withdraw` — and three facts: `name`, `step` (the venue's clock, and
therefore the frontier of what may be read) and `whitelist`. The assessment, the cap and
the switch are written against those eight things and nothing else, so the source of the
numbers is a constructor argument:

```python
from agent.physics.interfaces import MockWeb3Provider

provider = MockWeb3Provider()                       # the declared market, unchanged
empire = Empire.launch(body=body, provider=provider)
```

`MockWeb3Provider` *is* that market behind the interface — the same seven protocols, the
same seeds, the same two traps at steps 300 and 350 — and `tests/test_physics_interfaces.py`
asserts the parity rather than claiming it: a run through the adapter and a run through
the market end on the same credit, to twelve digits. Capital moves through the interface
too (`Portfolio.deposit` → `execute_stake`, `liquidate` → `withdraw`), and a venue that
refuses a transfer raises `StakeRefused` instead of recording a position the venue never
accepted. A bare `Portfolio` is handed a `NoVenueProvider`, which is a venue that says
"there is no venue" — so there is no code path in which a credit moves by some other
route.

What the layer does *not* contain is any ability to reach a network: no endpoint, no ABI,
no signer, no key. The test parses the module's imports and checks them, and the notebook
shows the real venue below, so the size of the step stays visible — five operations, three
facts, and `provider =` in one line.

### First contact: the same seam, answered by a chain (Phase 5)

`agent/physics/real_testnet_provider.py` is that one line, written out: a venue that
speaks to Sepolia, or to a local Anvil, through `web3.py`. Nothing above it changed — the
assessment, the cap, the switch, `Portfolio` and `Empire` still know only `IWeb3Provider`.

```bash
pip install 'enrg-ai[testnet]'
export TESTNET_RPC_URL=https://sepolia.example/v3/KEY     # the node
export TESTNET_PRIVATE_KEY=0x...                         # optional: without it, read-only
export TESTNET_PROTOCOLS_FILE=book.json                  # the operator's address book
```

```python
from agent.physics import RealTestnetProvider

provider = RealTestnetProvider.from_env()
for protocol in provider.get_protocol_list():
    print(protocol.name, f"{protocol.claim_apy:.1%}", protocol.returns.size, "epochs")
empire = Empire.launch(provider=provider)                # market=None: there is no market
```

Three properties are the reason the file is longer than the two lines it replaces:

* **configuration comes from the environment and nowhere else** — `TESTNET_RPC_URL`,
  `TESTNET_PRIVATE_KEY` or `TESTNET_UNLOCKED_ACCOUNT`, `TESTNET_PROTOCOLS[_FILE]`, plus
  timeouts and decimals. No endpoint, no key and no address is written in the source, and
  a test parses the file to prove it. The whitelist *is* the operator's book and cannot be
  widened by the venue; a rejected address is `False`, which `Portfolio` turns into a
  refusal;
* **a transaction is awaited** — estimate, sign, send, `wait_for_transaction_receipt`, and
  the local nonce is dropped on any failure so the next call asks the node. A loop cannot
  queue transactions it can never land. A reverted receipt and a refused gas estimate are
  `StakeRefused`, never a position;
* **`dry_run()` signs nothing** — `eth_call` says whether the call would revert,
  `estimate_gas` says what it would cost, and neither creates a hash. A mainnet chain id is
  refused unless `TESTNET_ALLOW_MAINNET=1` *and* `allow_mainnet=True`: two keys, one lock.

Tests: `tests/test_physics_real_testnet.py` — a recording `web3` double for the
translation and the failures (no network, no key), and, when `anvil` and `forge` are on
`PATH`, a real local chain: a token and a vault compiled and deployed, real receipts, real
balances, and `Portfolio` moving capital through it. Without foundry it skips by name.


### The discovery engine: 200 protocols, two levels and one rule

Seven declared protocols prove a rule; two hundred ask whether it survives scale.
`agent/physics/market_generator.py` draws a hostile market in one vectorised pass —
one `standard_normal((n, steps))` per kind, so 500 protocols of 720 steps take well
under a second — with the brief's mixture: **60 % scams** (a curve with no market
noise, an APY of 200…4000 %, half of them publishing no turnover, three quarters
hiding a `rug_step`), **30 % junk** (noise around a negative mean, a book thinner
than a credit a step) and **10 % honest** records (a real rate with market noise and
a claim equal to what is paid). The history is drawn **once**: the venue's clock
decides how much of it has happened, and no simulation step draws a new number.

`agent/physics/discovery.py` puts two levels in cost order, and neither does the
other's work. **Level 1** is a `Scanner`: it reads one published record and computes
a mean, a spread and a `CV` — microseconds per protocol — and drops what arithmetic
can see: dead, too short, below the turnover floor, noise instead of a yield, a
promise of profit on a record that loses money. It does **not** judge the curve: a
perfect formula has an ordinary mean and an ordinary `CV`, and reading that is
Statement #87's job. **Level 2** is the portfolio: it takes only the `watchlist` and
runs the Entrain assessment, and it alone writes the wallet's `execution_whitelist`.
The safety rule is enforced by types rather than discipline — `Scanner` has no method
that returns a `WhiteList` and `Candidate` carries no approval — so the scanner can
put a protocol in front of an assessment and never in front of money.

Measured at step 72 on a market of 200 (720 steps, seed 7): **200 published → 120
dropped by arithmetic, each with its own reason (no turnover, noise instead of a
yield, an unpaid record) → 60 refused by Entrain and 10 finding no room under a cap
of ten addresses → 10 honest funded**. Over 720 steps the scanned empire ends at
**211 credits** from 100 with all 32 of its deposits in honest protocols; the same
body with the same caps and no scanner reads all 200, puts **45 of its 52 deposits**
into junk and ends at **50**. No scam is funded by either arm — a sterile curve is
convicted by the assessment alone. Junk is the class with no substitute for level 1:
it is not a thief, it is just not a yield, and only the arithmetic of the record
shows that.

It is worth saying what happens to that funnel as the run goes on: at the last review
(step 792) the market, not the gate, has changed — all 90 rugs have fired, so level 1
drops **163** of 200 (*stopped paying* 90, *no turnover* 43, *noise, not a yield* 30) and
the same funnel reads **200 → 37 → 10**. A gate that narrows because the market rotted is
a measurement; a gate that narrowed because somebody tuned it would be a lie.

```python
import agent.physics.market_generator as M
from agent.physics import portfolio as P
from agent.physics.discovery import Scanner
from agent.physics.interfaces import MockWeb3Provider

market = M.generate_market(count=200, steps=720, seed=7)
venue = MockWeb3Provider(market)
venue.advance(72)                 # the venue's clock: what may be read
empire = P.Empire.launch(provider=venue, scanner=Scanner(), steps=720, seed=7)
empire.run(720)
print(empire.summary_text())      # the funnel of the last review, and the wallet
```

Demo: section 9 of `notebooks/digital_entrepreneur.ipynb` (the funnel, then the two
arms' equity on one plot with the deposits split by ground truth). Tests:
`tests/test_physics_discovery.py` — 54 of them, covering the generator's mixture
(120 scams of 200, the rug share, the vectorised draw) as well as the funnel's
identity, the `200 → 80 → 10` counts, "a scanner cannot approve", the 500-protocol
market under a second, and an `Empire.report()` that carries the funnel and survives
`json.dumps`.


### The wild market: 200 live DeFiLlama pools (Phase 6)

`agent/physics/defillama_adapter.py` points the same two gates at a market nobody designed.
Nothing in `Scanner`, `evaluate_opportunity`, the cap or the switch changed — the adapter
answers the shape a venue already has (`get_protocol_list()`, `protocol(name)`, `step`,
`name`), and the records arrive through it:

```python
from agent.physics import DefiLlamaScanner, DiscoveryEngine

funnel = DiscoveryEngine(mode="real_defillama").run(200)
print(funnel.summary_text())        # level 1, level 2, approved
funnel.top_rejected(3)              # the loudest promises the record refused, and why
```

Four decisions are what the file is for, and each is a place a careless adapter invents a
number:

* **a pool has no address** — `0x…` is the SHA-256 of the name (`portfolio.sim_address`),
  the same provably-not-a-wallet the drawn market uses;
* **the history is APYs, not returns** — `(1 + apy)^(1/365) − 1` is the exact inverse of
  `portfolio.annualise`, so a claim and a record are compared in the same units, and `CV`
  of the rates is `CV` of the log returns;
* **a missing APY is not a zero** — gaps are dropped and counted, and a series more than a
  quarter empty is refused rather than filled;
* **`429` is not "these pools publish nothing"** — measured on the first real run, at eight
  concurrent requests the host throttled 125 of the 200 pools. The adapter now paces itself
  (four in flight, two retries with doubling pauses) and names every pool it still could
  not read, instead of quietly reporting a smaller, cleaner market.

The one rule the adapter adds to the core's loop is the survey's window arithmetic: an
Entrain survey needs four fixations of 128 samples at a hop of 64, so a daily series
younger than about ten months is `UNMEASURED` — waiting, *not* refused — and the reason says
how many points produced how many windows. A dead feed is an empty funnel with the reason in
`error` (exit code 3 from the CLI), never a traceback.

Measured here — one scan, 5 Nov 2026, top 200 by TVL (cold cache ≈ 4 min, warm ≈ 1 s):

```text
mode real_defillama: the venue (defillama) publishes 200 pools, 195 of them readable
  level 1 (scanner): kept 152, dropped 43 (noise, not a yield 36, record too short 7)
  level 2 (entrain): judged 83, refused 31, unmeasured 69 (HONEST 52, SUSPECT 31, UNMEASURED 69)
  approved for investment: 52 of 152
```

The interesting number is not the approval rate but *which arm did the work*: among the 143
measured records the smallest `A²` was `1.29` and the median `30.7` against a sterility floor
of `0.1` — **no live pool was sterile**, because real APYs are noisier than any formula —
while 72 records claimed more than their own history paid. On this market the funnel is
convicted by the claim arm, not by Statement #87, and the figure prints both so that a
reader can see the difference rather than take it on trust.

Demo: section 10 of `notebooks/digital_entrepreneur.ipynb` (`REAL_MARKET = True`, tables of
the refusals with `A²`, `CV` and the σ gap, plus the funnel figure). Tests:
`tests/test_physics_defillama.py` — 36 of them, every response a `Mock`: the unit mapping,
the gaps, the throttle, the window rule, "one dead uuid costs one pool", a dead feed as a
value, and the assertion that no surface of the adapter — engine, scanner, market — has a
method that could move a credit.


### Alpha hunting: a panic is a price, a lie is a formula (Phase 7)

`agent/physics/volatility_hunter.py` asks the funnel's fifty-two approved pools a second
question, about a *day* rather than about a protocol: liquidity left, the same yield is now
shared by fewer dollars, the advertised APY tripled — **is the record still a market while
the price looks like fear?**

The obvious detector is four lines of numpy, and it is wrong in a way this repository can
show. Measured on the same live scan, "current APY more than two median absolute deviations
above its own median" ranks `blackrock-buidl BUIDL (BSC)` at **+19.70 σ** on a seven-basis-
point step — a tokenised Treasury bill as the most panicked asset in DeFi, because its MAD is
a few hundredths of a point. So the price arm has floors: `PANIC_SIGMA_FLOOR` (0.25 pp),
`PANIC_GAP_PP` (1 pp) and `PANIC_GAP_RATIO` (1.25×), after which the same pool reads
`2.63 σ / 0.66 pp / 1.20×` — a bill stepping with the Fed, and nothing else.

```python
from agent.physics import VolatilityHunter, PanicDetector

report = VolatilityHunter().scan(funnel)      # the funnel's HONEST pools, two series each
print(report.summary_text())                  # verdicts, then the panics
report.best()                                 # the loudest signal, or None
```

Two arms, and only both together are `PANIC` — the same rule the core keeps for its own:

| arm | measures | refuses with |
|---|---|---|
| `dislocation` | last 14 days against the resting level: `z`, points, multiple, and **the depth that left** | `CALM`, `GROWTH` |
| `organic` | Statement #87 on the whole record **and on the raised segment** + the assessment at rest | `FORMULA`, `DISHONEST` |

Two decisions carry the file. The claim a panicking pool is judged against is its **resting
level** (`resting_claim`, the median of everything before the last fortnight), never the rate
the panic advertises — otherwise the claim arm refuses every panic by construction. And the
record is judged at *two* horizons, because a pool can be noisy for a year and then announce a
flat 300 % for a fortnight, which is precisely the fortnight the strategy would be buying.

Measured here (same cache, 5 Nov 2026):

* **today**: 52 approved pools read, **zero panics** — the strongest reading among all 200 is
  `2.63 σ`, and "no panic today" is a result, not an empty screen;
* **replay**, the same arithmetic asked of every historical day of every pool (`replay` takes
  `apy[:day+1]`, so no verdict sees the future): **53 panics across 17 pools**, `41 of 53`
  still paying above the pool's resting level a month later, median `+1.35 pp` against
  `+0.00 pp` for every thirtieth ordinary day of the same pools — the control that makes the
  number mean something.

Sizing is the book's, not the detector's:
`Portfolio.alpha_hunting_weight(signal)` gives a fired signal `1 + ALPHA_BOOST_CAP ×
confidence` on its own bucket (half, at most), `boosted_targets()` funds it out of cash above
`ALPHA_CASH_FLOOR` (5 %, never spent) and logs every grant, and `rebalance(..., targets=...)`
runs the *same* accounting against the new numbers — a panic gets a bigger weight, not a
second code path.

Demo: section 11 of `notebooks/digital_entrepreneur.ipynb` — today's verdicts, the replay with
its control, two episodes side by side (one that paid and one that did not, `panic_replay.png`)
and the two books. Tests: `tests/test_physics_volatility_hunter.py` — 53 of them, all mocked:
the floors, each arm refusing alone, the resting claim, no look-ahead in the replay, the cap
and the cash floor, and the assertion that nothing here can sign, stake or transfer.


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

692 tests, no network, no key, no chain. Three dependencies are *workspace* ones,
and are named rather than hidden: the Anvil tests skip when foundry is absent, the
tests for the sibling `axis_core` package skip when that package is not installed,
and the `entrain_infer` core plus the scikit-learn datasets are needed by the
genome/sensorium/organism tests — on a box without them those tests say so by
name, which is why the full run is a *workspace* command, not a pip-install one.
Nothing on the deployment path needs any of the three.

## Production Readiness

What is already a property of the code, what an operator must supply, and what
this repository still does not have. Written for one VPS, because that is the
size of the thing: nothing here needs a cluster, a broker or a database.

### What is ready

* **One process, no services of its own.** The package is pure Python
  (`requires-python >= 3.10`, 3.12 tested) and **opens no port**: every entry
  point is a module run (`python -m agent.<module>`). State is a JSON file whose
  path is an argument, so "deploy" is a checkout, a venv and a unit file.
* **A working offline mode.** `--offline` on the self-training loop, the
  coherence CLI and the forecast CLI yields the same numbers every time with no
  egress at all — a box with a broken route still trains, forecasts and reports
  instead of idling.
* **No secrets in the tree.** No endpoint, no key and no address is written in
  the source: `RealTestnetProvider` reads the environment and nothing else, and
  a test parses the module to prove it. `describe()` prints the RPC *host* and a
  masked signer, because a report usually ends up somewhere shared.
* **One seam for every venue.** `IWeb3Provider` is five operations and three
  facts; a mock, a local Anvil and Sepolia are one constructor argument apart.
  Nothing above the seam knows which one it has, so the same code that runs in
  the tests runs on a chain.
* **A cost model that refuses rather than hopes.** `TransactionCostAnalyzer`
  prices gas and slippage (92 000 gas × 2 gwei × $3 000 = **$0.5520** a
  transaction on Ethereum, **$0.00276** on an L2 at 0.01 gwei) and `Portfolio`
  refuses an entry whose net alpha is below the floor, recording it in `cost_log`
  / `friction_summary()`. Exits are never refused and never charged: the kill
  switch is not hostage to a gas estimate.
* **A venue switched by variables, not by code.** No chain id is written into the
  provider: `TESTNET_CHAIN_ID` is read and asserted, and
  `TESTNET_PROFILE=arbitrum` lets the names a node provider hands out
  (`ARBITRUM_RPC_URL`, `ARBITRUM_PRIVATE_KEY`, `ARBITRUM_CHAIN_ID`) answer for the
  canonical ones. A profile is a spelling, not a permission — 42161, 137 and 8453
  are in the mainnet lock, so an L2 still needs both yeses to sign.
* **A suite that runs like a deployment.** 692 tests, no network, no key, no
  chain; a live chain is an extra (`pip install 'enrg-ai[testnet]'`) and the
  Anvil tests skip by name when foundry is absent.

### Why an L2, in one measurement

TCA priced the *trade*, not the idea: on Ethereum, 92 000 gas at 2 gwei and
$3 000/ETH is **$0.5520** before the position exists — on a $50 ticket that is
1.1 % of the capital, an order of magnitude above the 0.1 % floor in `Portfolio`.
The same call on Arbitrum One, at ~0.01 gwei, is **$0.00276**:

| Where | gas price | 92 000 gas costs | on a $50 ticket |
|---|---|---|---|
| Ethereum L1 | 2 gwei | $0.5520 | 1.1 % — above the 0.1 % floor |
| Arbitrum One | 0.02 gwei | $0.00552 | 0.011 % — below it |
| Arbitrum One | 0.01 gwei | $0.00276 | 0.0055 % — two orders below |

Two honest footnotes. A rollup also charges an **L1 data fee** for the calldata it
posts — a few cents for a stake, in practice — which this model does not price, so
read the L2 column as "cents, not half a dollar". And the model is a parameter,
not a religion: `TransactionCostAnalyzer(gas_price_gwei=0.01)` is exactly how the
number above was produced, and `friction_summary()` reports what the floor
actually refused.

### How to deploy on a VPS (Arbitrum One), no Docker

#### 1. The box

Any VPS with Ubuntu 22.04 (24.04 also tested), 2 vCPU / 2 GB RAM / 20 GB SSD is
more than enough: the agent opens **no port** and needs no database. Work as a
non-root user, keep the key out of `root`'s environment, and open nothing inbound:

```bash
sudo adduser --disabled-password --gecos "" enrg
sudo install -d -o enrg -g enrg /opt/ENRG-AI /var/lib/enrg-ai
sudo ufw allow OpenSSH && sudo ufw enable     # SSH only: nothing here listens
```

#### 2. Python, git, the venv

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
sudo -iu enrg
git clone https://github.com/AntonGrid/ENRG-AI.git /opt/ENRG-AI && cd /opt/ENRG-AI
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install '.[testnet,market,plot]'
pytest -q                                   # the standalone suite (see "Tests")
cp .env.example .env && chmod 600 .env      # fill in only what you need
```

A smoke test that touches no money — three CLIs, deterministic, offline:

```bash
python -m agent.physics --source offline --output coherence.json
python -m agent.forecast --source offline --horizon 6 --output forecast.json
python -m agent.digital_train.pipeline --once --offline --points 48 \
       --state /var/lib/enrg-ai/digital_state.json
```

#### 3. `.env` — a real Arbitrum node and a funded key

An endpoint from Infura, Alchemy or the public `https://arb1.arbitrum.io/rpc`, and
a **fresh** address for the signer:

```ini
# which chain: the profile names it, and the ARBITRUM_* names may answer for TESTNET_*
TESTNET_PROFILE=arbitrum
# the node (this string usually contains an API key — it is a secret too)
TESTNET_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY_HERE
# the signer: a dedicated address, funded with what you can afford to lose
TESTNET_PRIVATE_KEY=0x...
# asserted against the node's own chain id; 42161 = Arbitrum One (empty = trust the node)
TESTNET_CHAIN_ID=42161
# the whitelist: nothing outside this book can be staked into
TESTNET_PROTOCOLS_FILE=book.json
# still 0: 42161 is a chain where credits are money, and the other half of the lock is in code
TESTNET_ALLOW_MAINNET=0
```

The three names a node provider prints — `ARBITRUM_RPC_URL`,
`ARBITRUM_PRIVATE_KEY`, `ARBITRUM_CHAIN_ID` — are honoured as well, which is what
`TESTNET_PROFILE` buys: the canonical `TESTNET_*` name wins when both are set, and
`describe()` records which one answered (`{"TESTNET_RPC_URL":
"ARBITRUM_RPC_URL"}`), so provenance is never guessed. A bare `RPC_URL` or
`PRIVATE_KEY` is deliberately **not** an alias: a stray variable from another
project must not be able to become a signer.

Then the chain, read-only, *before* anything signs:

```python
from agent.physics import RealTestnetProvider

provider = RealTestnetProvider.from_env()   # no node => TestnetNotConfigured, never a default
print(provider.describe()["config"])
# {'rpc_host': 'arb-mainnet.g.alchemy.com', 'signer': 'read-only', 'chain_id': 42161,
#  'profile': 'arbitrum', 'sources': {}, 'decimals': 6, ...}
print(provider.describe().get("unavailable"))
# chain 42161 is a mainnet: this venue is configured to move credits on a testnet, and
# letting it sign here would need TESTNET_ALLOW_MAINNET=1 and allow_mainnet=True
#   ^ the lock doing its job, not a bug: this is the state to *start* in

# On Arbitrum Sepolia (TESTNET_PROFILE=arbitrum + ARBITRUM_CHAIN_ID=421614) the whole
# loop runs and signs with no lock at all. Only after that, on One, give both yeses:
provider = RealTestnetProvider.from_env(allow_mainnet=True)   # env var too, or this raises
for p in provider.get_protocol_list():
    print(p.name, f"{p.claim_apy:.1%}")
print(provider.dry_run("stake", "<vault from your book>", 250.0))   # nothing is signed
```

Signing on 42161 needs **both** yeses — `TESTNET_ALLOW_MAINNET=1` in `.env` *and*
`allow_mainnet=True` in the calling code. Arbitrum Sepolia (421614) is the rehearsal
space: the same profile, a different id, no lock, no money.

#### 4. Run it forever: systemd

A unit file at `/etc/systemd/system/enrg-ai-trainer.service` — this is the part
that outlives your laptop, your SSH session and a reboot:

```ini
[Unit]
Description=ENRG-AI self-training loop
After=network-online.target

[Service]
User=enrg
WorkingDirectory=/opt/ENRG-AI
EnvironmentFile=/opt/ENRG-AI/.env
ExecStart=/opt/ENRG-AI/.venv/bin/python -m agent.digital_train.pipeline \
          --loop --interval 3600 --state /var/lib/enrg-ai/digital_state.json
Restart=always
RestartSec=10
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/enrg-ai

[Install]
WantedBy=multi-user.target
```

```bash
sudo chown enrg:enrg /opt/ENRG-AI/.env && sudo chmod 600 /opt/ENRG-AI/.env
sudo systemctl daemon-reload && sudo systemctl enable --now enrg-ai-trainer
journalctl -u enrg-ai-trainer -f
```

#### 5. The fast way: tmux (a look, not a deployment)

```bash
tmux new -s enrg
cd /opt/ENRG-AI && . .venv/bin/activate
set -a && . ./.env && set +a         # export the file into this shell only
python -m agent.digital_train.pipeline --loop --interval 3600 \
       --state /var/lib/enrg-ai/digital_state.json
# Ctrl-b then d to detach; `tmux attach -t enrg` to look again
```

tmux survives a dropped SSH session but **not** a reboot, and it does not restart
a loop that died. Use it to watch the first cycle; use the unit for the standing
job.

### Deploy with Docker

```bash
cp .env.example .env && chmod 600 .env
mkdir -p data && sudo chown 10001:10001 data   # see below: the image is not root
docker compose up -d --build
docker compose logs -f trainer
docker compose run --rm coherence            # one-shot job, behind the `jobs` profile
docker compose exec trainer pytest -q        # the suite, as far as it ships in the image
```

The `mkdir`/`chown` is not decoration: `data/` is gitignored and `.dockerignore`d,
so on a fresh box compose creates it as `root:root` and the container — uid
`10001`, on purpose — cannot write its state file. The symptom is a loop that
dies with `PermissionError: [Errno 13] ... /data/digital_state.json` on the first
cycle; the fix is the line above, once.

The image is `python:3.12-slim` plus the package's extras, runs as uid `10001`,
and holds **no key, no `.env` and no address** — both are `.dockerignore`d, so
`docker build` cannot bake a secret into a layer. `trainer` is the long-running
service (the self-training loop, one cycle an hour, `${TRAIN_INTERVAL}` to
change it); `coherence` is the same seam as a one-shot job. One health check,
and it is the honest one: the state file is written at the end of every cycle,
so a file older than three hours means the loop is not working and compose
restarts it.

The `.env` from step 3 is the file compose reads: the same
`TESTNET_PROFILE=arbitrum` and the same key, handed to the container's
environment at start — never copied into the image, never a build argument.

The image carries the suite and the `dev` extra so it can be *measured*, not
because a deploy needs them: `pytest -q` in the container runs every test whose
dependencies ship in this repository, and the ones that need a workspace sibling
(the `entrain_infer` core, `axis_core`, the scikit-learn datasets) say so by
name. The whole 692 run on a host that holds the workspace. What to check after
the first build: the image runs as uid `10001`, writes `/data/digital_state.json`
through the mounted volume when given `--once --offline`, and its health check
answers 1 before that file exists and 0 after.

### Environment variables

`TESTNET_*` is the venue (`agent/physics/real_testnet_provider.py`). Nothing in
this table has a default that could move value: missing configuration is an
error with a name, never a silent fallback.

| Variable | Needed for | Default |
|---|---|---|
| `TESTNET_RPC_URL` | every chain call — **required** for a real venue | none → `TestnetNotConfigured` |
| `TESTNET_PRIVATE_KEY` | signing capital calls | none → the venue is **read-only** |
| `TESTNET_UNLOCKED_ACCOUNT` | signing by the node (Anvil/Hardhat) instead of a key | none |
| `TESTNET_CHAIN_ID` | asserted against the node's own chain id; a mismatch refuses to connect | none → accepts what the node reports |
| `TESTNET_PROFILE` | `arbitrum` (42161), `base` (8453), `polygon` (137) or `sepolia` (11155111): names the chain, so the provider's own spellings answer and the id need not be written down | unset → `TESTNET_*` only |
| `ARBITRUM_RPC_URL` / `ARBITRUM_PRIVATE_KEY` / `ARBITRUM_CHAIN_ID` | the same three values under a node provider's names — read only when `TESTNET_PROFILE=arbitrum` **and** the canonical name is empty | the canonical name always wins |
| `TESTNET_PROTOCOLS_FILE` / `TESTNET_PROTOCOLS` | the operator's address book — the whitelist | empty book (nothing stakeable) |
| `TESTNET_DECIMALS` | credits per token unit | `6` |
| `TESTNET_LOOKBACK` | blocks scanned for a record | `5000` |
| `TESTNET_RECEIPT_TIMEOUT` | seconds to await a receipt | `120` |
| `TESTNET_BLOCK_TIMEOUT` | seconds to await a block | `30` |
| `TESTNET_ALLOW_MAINNET` | `1` to let a real-money chain through (with `allow_mainnet=True` in code) | unset → refused |
| `TESTNET_ABI_FILE` | override the built-in vault ABI | built in |
| `AXIS_AI_SIGNING_KEY` | `--sign` on `agent.signals` / `agent.physics` (base64 Ed25519 seed) | unset → `--sign` exits 2 |
| `ENRG_CACHE_DIR` | where the market cache lives | `.cache` |

The three names a deploy normally touches map straight onto that table:
**`RPC_URL` → `TESTNET_RPC_URL`**, **`PRIVATE_KEY` → `TESTNET_PRIVATE_KEY`**,
**`CHAIN_ID` → `TESTNET_CHAIN_ID`** — but the mapping is done by *you*, in the
file, not by the code. The code reads the canonical names, plus (when
`TESTNET_PROFILE` names a chain) that chain's prefixed ones. A bare `RPC_URL` or
`PRIVATE_KEY` in the environment is ignored on purpose: another project's variable
must never be able to become this project's signer.

A `.env` file is read when `python-dotenv` is installed (the `testnet` extra); it
is never written, and an exported variable always wins — a file that can override
the environment is a file that can silently redirect a signer.

### A real Web3 wallet is required — and it is a hot key

`TESTNET_PRIVATE_KEY` is a plain EOA key in an environment variable. That is the
weakest link in any deployment of this package, so it is stated rather than
implied: whoever reads that variable can move everything the address holds.

* **a dedicated address, funded with ~$50** — real money, and only as much as you
  would shrug at losing: it is enough for the loop to be real (the L2 gas for
  thousands of transactions is a few dollars) and small enough that a mistake is
  survivable. The code does not enforce the amount; it is a rule you keep.
* **that key is shown to nobody.** Not in a screenshot, not in a chat, not in a
  commit, not as a `docker build --build-arg`. Everything a report needs is the
  masked form `describe()` prints (`0x...abab`); the key itself never leaves
  `.env`, and `.env` is `chmod 600`, owned by the service user.
* **Sepolia or a local Anvil first** — until the whole loop has run there. A
  mainnet chain id is refused unless `TESTNET_ALLOW_MAINNET=1` *and* the code
  passes `allow_mainnet=True`: two keys, one lock, because the mistake this
  guards against is expensive and one keystroke away. Arbitrum, Base and Polygon
  are mainnets by that definition, however cheap their gas.
* **`chmod 600 .env`**, owned by the service user; with systemd, an
  `EnvironmentFile=` read by root; with Docker, `env_file:` (never a build
  argument), so it stays out of every layer.
* **one writer per key.** The nonce is dropped on any failure and re-asked from
  the node, but it lives in memory: two processes signing with the same key
  produce a replacement-transaction race, not a queue. Scale by adding keys, not
  replicas.
* **`dry_run()` before every signature** is the operator's habit, not the code's
  guarantee: `eth_call` says whether it would revert, `estimate_gas` says what it
  would cost, and neither creates a hash.
* **rotation is a restart** — change the variable, restart the unit, and the old
  key is out. There is no revocation list here; that is what the operator's
  address book is for.

### What this is not

An honest list, so nothing above is read as more than it is:

* **no metrics endpoint, no alerts, no tracing.** Logs go to stdout, a supervisor
  (`systemd`, `compose restart`) is the entire availability story, and the health
  check is a state file's mtime. Put the unit behind your own monitoring.
* **no secret manager** — no KMS, no HSM, no multisig, no per-request key
  scoping. An operator who needs those should keep signing in their own service
  and hand this package a provider.
* **no lock file** — dependencies are floors in `pyproject.toml`, so a rebuild can
  move them. `pip freeze > constraints.txt` on the box that works, and rebuild
  from that.
* **state is local JSON** (`digital_state.json`, `.cache/`) — back it up or lose
  the accumulated history.
* **nothing here is a trading bot.** There is no scheduler in the venue, one call
  to `execute_stake` is one transaction, and `Portfolio` refuses far more often
  than it buys. The system is deployable; it is not autonomous capital.



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

