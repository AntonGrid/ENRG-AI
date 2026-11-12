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
├── physics/           # coherence, curiosity, the economic agent, the DeFiLlama venue
├── trade/             # the trading cycle: book → market → hunter → size → a stake
├── backtest/          # the same hunt on history: candles → verdicts → a report
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
  refused unless `TESTNET_ALLOW_MAINNET=1` *and* `allow_mainnet=True`: two keys, one lock;
* **a fee is bid above the base fee** — `maxFeePerGas` is `2 × base_fee + tip` and then +12%
  (`FEE_BUFFER_PERCENT`), with a legacy `gasPrice` buffered the same way on a chain that has
  no base fee, because a cap signed *at* the node's number is refused the moment a block
  ticks it up. The first real `approve` on Arbitrum failed exactly that way; a refusal now
  quotes what was bid, what the chain asks after it, and the difference.

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


## The trading cycle: from a verdict to a signature (Phase 8)

Everything above this section *measures*. `agent/trade/loop.py` is the file that signs —
and the file the container runs. One cycle answers four questions in this order, and every
answer can stop it:

| # | question | answered by | what stops the cycle |
|---|---|---|---|
| 1 | what may be traded? | `TESTNET_PROTOCOLS` — the operator's book — plus the record each vault published (`RealTestnetProvider.get_protocol_list`) | an empty book: *nothing to price* |
| 2 | what does the market say? | `DiscoveryEngine` (DeFiLlama, two gates) → `VolatilityHunter`: the **same** verdicts its own CLI prints | `THIN`, `FORMULA`, `DISHONEST`, `UNMEASURED` |
| 3 | how much? | `provider.balance()` — read *on that cycle* — then `agent/trade/capital.size_trade` | no credits, no gas, no ceiling, an unreadable wallet |
| 4 | sign, or say it? | `DRY_RUN` | unset means **dry** |

```bash
python -m agent.trade.loop --once --market synthetic --dry-run   # rehearse, no egress
python -m agent.trade.loop --once --dry-run --json cycle.json    # one cycle, as a file
python -m agent.trade.loop --loop --interval 300                 # the container's command
```

### How wide the hunt is: `SCAN_LIMIT` and `MEASURE_LIMIT`

A cycle reads the market in two stages, and both widths are the operator's, not the code's:
`SCAN_LIMIT` pools go through the two gates (default **200**, the feed's own
`POOLS_LIMIT` — a test holds the two equal so this file cannot disagree with the funnel it
describes), then `MEASURE_LIMIT` of the approved ones are measured by the hunter (default
**50**). A first deployment ran `50` and `10`: enough to see the pipe work, not enough to
call it a hunt.

```bash
SCAN_LIMIT=350 MEASURE_LIMIT=20 python -m agent.trade.loop --once
python -m agent.trade.loop --once --limit 9 --hunt 3    # a flag beats the environment
```

Both are honoured from `.env` (the container reads it), a flag overrides either, and `0`,
`-5` or `lots` stop the process with the variable named — a scan of nothing is not
"unlimited", and a loop that read it that way would hammer a feed until it stopped
answering. The cycle prints the width it used: `market real (scan 200, measure 50)`.

### Two ceilings, and a wallet smaller than both

`MAX_TRADE_SIZE_USD` is what the operator will risk on a single transaction.
`MAX_TRADE_FRACTION` (default `0.20`) is a share of what the wallet holds **at that
moment** — the balance is read from the node on the same cycle that sizes the trade, never
remembered from a previous one. The smaller number wins.

`.env.example` ships `MAX_TRADE_SIZE_USD=10.0`, because the wallet this was written for
holds about `$18`: the trade is then `$3.60`, and the **fraction** is what decided it. A
ceiling above the wallet never binds and is not a guard — it is a number in a file. Raise it
when the wallet grows; leave the fraction where it is.

The double below is deliberately larger — a `$50` ceiling and a `$68` wallet — so that both
ceilings can be seen at once, with the log naming the one that decided:

```text
── cycle 1 · DRY RUN (nothing will be signed) · chain 11155111 · block 101
   book  Aave V3 USDC             +3.72% implied
   market 1 pools read · panics 1 · verdicts {'PANIC': 1}
   would stake  Aave V3 USDC  13.600000 ($13.60)  [PANIC 1.00] — MAX_TRADE_FRACTION
                decides it: 20.0% of the wallet at $13.60 of $68.00
```

(Those four lines are the test double from `tests/test_trade_loop.py`, not a live chain:
a `$50` ceiling, a `$68` wallet — 50 credits and 0.008 ETH priced at $2,250.)

Three refusals matter more than the number:

* **ether is not a balance.** A stake spends the vault's ERC20 and *costs* a little of the
  native coin; a wallet with $18 of ETH and no credits cannot stake, and the cycle says
  exactly that instead of sending a transaction that reverts.
* **the gas reserve is a refusal, not a ceiling.** Below `GAS_RESERVE_ETH` (default
  `0.0005`) the approve could land and the stake could not.
* **a panic may lift the dollar ceiling by half — never the fraction.** `ALPHA_BOOST ×
  confidence` applies to `MAX_TRADE_SIZE_USD` only: "a fifth of the wallet" is the answer
  to *what is the worst one mistake can do*, and a measurement is not allowed to enlarge
  the mistake it is measuring.

### The seam this loop refuses to invent

A DeFiLlama pool is not a vault: the feed publishes a project, a chain and a UUID, the book
publishes a contract address, and no code here can prove one implements the other. So a
signal is acted on only when the pool and a book entry share a name (normalised: letters
and digits). A panic with nothing behind it in the book is a **signal**, not an action:

```text
   signal   aave-v3-usdc  [PANIC 0.42] — a measured dislocation, but no vault in the
            operator's book implements this pool: nothing was sized and nothing was signed
```

That line is the honest answer on a first run, and on most runs. No amount is printed for
it, because without a vault there is no token to stake: a dollar figure there would be a
number about nothing.

### The address book: why a fresh deployment prints *book empty*

The book is not a list of protocols you like. It is the list of addresses this venue may
stake into, and an entry has to be a **vault of this interface**:

```solidity
event YieldDistributed(uint256 indexed epoch, uint256 rateE18);   // the record
function stake(uint256 amount) external;                          // capital in
function withdraw(uint256 amount) external;                       // capital out
```

Two facts follow, and both of them surprise a first run:

* **no `YieldDistributed` log, no price.** `get_protocol_list()` reads those logs out of
  the last `TESTNET_LOOKBACK` blocks (`5000` by default). A book entry with no log is
  *not published* — correctly, because a protocol with no record cannot be measured, and
  the hunter will not size a position on nothing. The cycle now says which entry failed
  instead of one word about the whole book:
  `aave-v3 USDC (Arbitrum) (0x724d…c637): no YieldDistributed(uint256,uint256) log in the last 5000 blocks`.
* **a dApp is not a vault — unless the book says what it is.** Aave's `aUSDC`, a Uniswap
  V3 pool, a Curve gauge: real addresses, real money, and none of them answer
  `stake(uint256)` or emit our event. Two of those are now *usable* rather than merely
  diagnosable, and only because the book can say what they are:
  `"kind": "aave-v3"` names a lending pool whose address is checked against
  `agent/physics/aave_v3.py` (a table keyed by chain id), and the money then moves with
  `supply(asset, amount, onBehalfOf, referralCode)` instead of `stake`. Everything else —
  a Uniswap pool, a gauge, an aToken — stays a diagnosis, because nothing in this file can
  name the contract that would answer for it.

### The second source: `TESTNET_HISTORY`, or why the book is no longer empty

A record is what the hunter measures, and it has exactly two possible sources. The vault's
own logs (above) are the only statement about the **address** — and almost every real DeFi
address publishes nothing in our event. So the venue can price an entry from DeFiLlama
instead:

| `TESTNET_HISTORY` | where the record comes from |
| --- | --- |
| `chain` (default) | the address's own `YieldDistributed` logs, and nothing else |
| `feed` | `https://yields.llama.fi/chart/{uuid}` — the daily APY series of the entry's `name`, or of the `pool` uuid the book pins. **The node is not asked at all.** |
| `auto` | the log first, the feed when the log says nothing — an address that publishes its own record is never replaced by a name match |

The feed's series is converted to per-step rates with the same `(1+apy)^(1/365) − 1` the
funnel uses (`agent/physics/defillama_adapter.py`, `FeedRecord.rates`), so the book's
`implied_apy` and the hunter's reading are the same arithmetic on the same curve.

**What this does not do is let the loop sign into an address it has never read.** A record
that came from the feed is a fact about a *name*; `stake(uint256)` is a fact about an
*address*. So:

* `RealTestnetProvider.history_source(address)` answers `chain` for a record read from the
  address itself and the feed's label — `the feed aave-v3 USDC (Arbitrum) [d9fa8e]`, or
  `the feed pool [d9fa8e14] (pinned by the book)` — otherwise;
* `RealTestnetProvider.protocol_kind(address)` answers what money into that address
  *calls*: `vault` for `stake(uint256)`, `aave-v3` for `supply(…)`, `""` for an address the
  book does not contain. The two answers are the whole story a signature needs;
* the protocol's `note` carries the same sentence;
* the cycle prints provenance on the book line and **holds** instead of signing — this is
  one dry-run cycle of the two-entry example book with the market arm answering `PANIC`
  for the first pool (the block below is verbatim from the run described under *how that
  was produced*, except that the provenance lines are shown in full):

```text
── cycle 1 · 2025-10-03 17:00:00 · DRY RUN (nothing will be signed) · chain 42161 · block 20000000
   book  aave-v3 USDC (Arbitrum)  +2.89% implied · the feed pool [d9fa8e14] (pinned by the book) · aave-v3
   book  uniswap-v3 WETH-USDC (Arbitrum) +18.77% implied · the feed pool [a14bd201] (pinned by the book)
   market 2 pools read · panics 1 · verdicts {'PANIC': 1}
   would stake  aave-v3 USDC (Arbitrum)  1.000000 ($1.00)  [PANIC 1.00] — TEST_STAKE_AMOUNT_USD decides it: 7.2% of the wallet, stable credits only — no ETH_USD_PRICE, so the 0.001950 ETH is not valued — supplied through the Aave v3 Pool, so the position becomes 1.000000 and its receipt is an aToken in this wallet
   hold     uniswap-v3 WETH-USDC (Arbitrum) — the market has no pool named like this vault — a signal about a pool is not a signal about this address, and this loop does not guess which contract implements which pool
```

One entry is signed and the other is held, on the *same* feed and the *same* wallet, and
the difference is the kind: the Aave pool's address is not inferred from a name — it is the
pool the deployment table files under chain `42161`, checked before the first byte is
signed — so the feed's rate is allowed to price what that address pays. The Uniswap pool
has nobody to vouch for it, so its record stays a measurement. Deploy your own vault
(`tests/test_physics_real_testnet.py` carries the reference contract and compiles it with
Foundry), publish one `YieldDistributed`, and a `hold` of that kind becomes a `stake` with
no configuration change, because the source becomes `chain`.

The `stake` line above is also where the two ceilings the operator asked for show up by
name: `TEST_STAKE_AMOUNT_USD=1.0` (send one dollar before trusting the loop with thirteen)
and `MAX_TRADE_FRACTION` (a fifth of the wallet, never lifted by a panic). On a venue that
can be **topped up** — an Aave `supply` grows an aToken position — those ceilings size the
*position* and a cycle carries the difference, so raising the cap later adds to the
position instead of opening a second one, and a difference under `$0.50` is refused rather
than paid for in fees.

**How a name becomes a pool.** `agent/physics/defillama_adapter.FeedHistory` ranks the
catalogue by evidence and counts the ties: an exact symbol, then the whole
`project symbol (chain)` label, then the same *set* of words (`USDC-WETH` →
`WETH-USDC`), then a symbol that carries the name, then a name that carries the symbol.
The deepest pool on the venue's chain wins a tie, and the report says how many tied
(`the feed … (deepest of 41)`). A name nothing carries is an absence with the feed's own
reason — never the nearest pool. Pinning `pool` in the book skips the match entirely.

`book.example.json` in the repo root is a file the loader accepts, and it holds both facts:
two **verified** Arbitrum One addresses (Aave V3 `aArbUSDCn`
`0x724dc807b04555b71ed48a6896b6F41593b8C637` over native USDC
`0xaf88d065e77c8cC2239327C5EDb3A432268e5831`; Uniswap V3 USDC/WETH 0.05 %
`0xC6962004f452bE9203591991D15f6b388e09E8D0`, read back from the factory's `getPool`),
written down as the *diagnosis* they are, **plus the DeFiLlama `pool` uuid each address
actually is** — `d9fa8e14-…` for the Aave row (its `underlyingTokens[0]` is that USDC) and
`a14bd201-…` for the Uniswap row (its `poolMeta` is `0.05%`) — and the template for your own
vault. Copy it and ask the question:

```bash
cp book.example.json data/book.json          # the container reads /data/book.json
docker compose exec trader python -m agent.trade.loop --book-report
```

With `TESTNET_HISTORY=chain` (the code's default) the answer is "nothing to price" and, new
in this step, the line that says why and what to do about it:

```text
book report — 2 entry/entries · lookback 5000 blocks · chain 42161 · TESTNET_HISTORY=chain
  aave-v3 USDC (Arbitrum)  0x794a61358d6845594f94dc1db02a252b5b4814ad  aave-v3  no record
                           market: market off
                           stakable: no
  uniswap-v3 WETH-USDC (Arbitrum) 0xc6962004f452be9203591991d15f6b388e09e8d0  vault  no record
                           market: market off
                           stakable: no
  nothing to price: an entry has to emit YieldDistributed(uint256,uint256) (and a cycle has to see it inside the lookback window) before the hunter can measure it
  the feed was not asked — TESTNET_HISTORY=auto prices an entry from DeFiLlama's daily history of its name when the vault itself published nothing, and =feed asks only the feed
```

With `TESTNET_HISTORY=auto` (what `.env.example` ships) the same book is priced, and the
report says where each record came from:

```text
book report — 2 entry/entries · lookback 5000 blocks · chain 42161 · TESTNET_HISTORY=auto
  aave-v3 USDC (Arbitrum)  0x794a61358d6845594f94dc1db02a252b5b4814ad  aave-v3  120 record(s) · the feed pool [d9fa8e14] (pinned by the book)
                           stakable: the address is the Aave pool this file knows for this chain, so the feed names the rate and supply() moves the money — the deployment is verified, not inferred from the name
                           market: market off
                           stakable: yes
  uniswap-v3 WETH-USDC (Arbitrum) 0xc6962004f452be9203591991d15f6b388e09e8d0  vault  120 record(s) · the feed pool [a14bd201] (pinned by the book)
                           not stakable as-is: the record above is the feed's about this name, and this loop signs only into a vault whose own log it has read
                           market: market off
                           stakable: no
  2 of 2 record(s) came from the feed, not from an address's own log — 1 of them about a name the venue cannot verify as a vault, and those are not staked
```

How those two blocks were produced, exactly. Real: `book.example.json` loaded by the
shipping loader; the pool catalogue from the feed's own **cached** document on this machine
(15 758 pools, fetched 2026-11-07), against which `FeedHistory.find("USDC-WETH")`
independently resolves to `a14bd201-…` — `uniswap-v3 WETH-USDC (Arbitrum)`, $36.7 M, the
deepest of 18 pools carrying that pair — which is the uuid the book pins; the APY→rate
conversion, `loop._book_report` and the cycle above. Stubbed, because this box has no
egress: the two `/chart/{uuid}` series (120 daily points at each pool's own published APY —
2.89 % for Aave USDC, 18.77 % for the Uniswap pool), the node (a double answering chain
`42161` and no logs, which is what `history_source` and `protocol_kind` are asked about),
and the market arm (one `PANIC` handed over instead of the funnel's two hundred pools and
fifty measured). `tests/test_physics_defillama.py` and `tests/test_trade_loop.py` pin the
same behaviour against a fully mocked wire; `tests/test_physics_real_testnet.py` pins the
policy and `tests/test_physics_aave_v3.py` pins the deployment table.

`--book-report` exits `4` when nothing can be priced — the same code the hunter uses for
"nothing to hunt in" — so a script can tell an empty book from a broken node.

There are now two ways to get a book that prices *and* can be funded, and both of them are
an address this file can vouch for:

* **your own vault.** `tests/test_physics_real_testnet.py` carries the reference contract
  (`TestVault.sol`: `publish(rateE18)`, `stake`, `withdraw`, the four events) and compiles
  it with Foundry. Deploy it, publish one `YieldDistributed`, and put **its** address in
  `data/book.json` with `token` set to the ERC20 it pulls, `decimals` to that token's own
  (6 for USDC) and `name` spelled like the DeFiLlama pool you want it measured against.
  A vault is matched to a verdict by that name and nothing else, exactly (normalised:
  letters and digits), so `USDC` does **not** match `aave-v3 USDC (Arbitrum)`; `pool` pins
  the history, `name` pairs the verdict.
* **Aave v3, which is already deployed and audited.** One line in the book —
  `"kind": "aave-v3"` with the Pool as `address` and USDC as `token` — and the venue
  supplies on a feed-priced record, because that address is checked against a table rather
  than guessed from a name. See *Aave v3* below.

### Aave v3: somebody else's interface, and the table that vouches for it

A vault is trusted because it published its own record. A lending pool cannot be: it emits
no `YieldDistributed`, and it does not answer `stake(uint256)` at all. So the trust moves
to a different place, and it has to be a place that **refuses**:

```python
# agent/physics/aave_v3.py — the only pools this file will sign into, by chain id
42161: AaveDeployment(pool="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
                      asset="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # native USDC
                      symbol="USDC", decimals=6,
                      a_token="0x724dc807b04555b71ed48a6896b6f41593b8c637")
```

* a book entry with `"kind": "aave-v3"` is checked as a **pair** — this pool *and* this
  asset, on the chain id the *node* reported — and refused before a single byte is signed
  if the pair is not in the table. A book cannot declare a venue into existence, which is
  the whole reason `kind` is safe to have;
* the call is Aave's: `supply(asset, amount, onBehalfOf, referralCode)` with the signer as
  its own beneficiary and `referralCode = 0` (this loop is nobody's affiliate), and
  `withdraw(asset, amount, to)` paying the signer. An `approve` to the Pool comes first,
  and is awaited like any other transaction;
* **the case of an address is applied at the boundary, never stored.** EIP-55 is a typo
  check, not data, so the table keeps its addresses in one form, the book may spell them in
  another, and the signer arrives as the key produced it. web3's encoder accepts only the
  checksummed form and says so *late* — which is how the first real `supply` was refused
  one block after its `approve` had been paid for: `0xaf88d065e77c8c…` is a perfectly
  correct address and an uncodable argument. Every string a Pool call receives — the
  asset, the beneficiary, the recipient, and the Pool itself — goes through
  `Web3.to_checksum_address` in `_pool_call`, the one place that hands Aave an argument;
  the dry run builds its call there too, because a simulation that could not be signed is
  a green light pointing at a wall. `tests/test_physics_real_testnet.py` makes its web3
  double refuse a lower-case address with the sentence the real encoder sent, so a
  forgotten conversion fails in a test instead of on a chain;
* the **position is read off the aToken**, not remembered: `balance()` reports
  `position_token` / `position_balance`, which is the supplied amount plus every interest
  payment since — so a lost state file, a second operator, or a withdrawal made by hand
  changes the number the next cycle sizes against;
* the **rate is the feed's**: `claim_apy` is the book's claim (a Pool has no
  `claimApyE18`), and the record is DeFiLlama's history of the pinned pool uuid. That is
  what an `aave-v3` entry needs `TESTNET_HISTORY=auto` for, and the venue says so by name
  when the policy is `chain`;
* a **position can be topped up**, so it is sized as a target: the same ceilings
  (`MAX_TRADE_SIZE_USD`, `MAX_TRADE_FRACTION`, `TEST_STAKE_AMOUNT_USD`) decide the
  position, each cycle carries the difference, and a difference under `$0.50` is a refusal.
* the **match does not depend on spelling**. A book spells `aave-v3 USDC (Arbitrum)` for a
  human and DeFiLlama spells its row `USDC` under project `aave-v3` for a database, so an
  `aave-v3` entry may declare the feed's own words — `"project": "aave-v3"`, `"symbol":
  "USDC"`, `"chain": "Arbitrum"` — and is recognised by the pinned `pool` uuid first, then
  by project + symbol + chain, then by its name. Those three are checked against the
  deployment table before a signature (an entry that pins `USDT` on the USDC pool is
  refused, because it would be matching another row's verdict). A vault gets none of this:
  its address is a stranger until its own log speaks, so its name still has to match.
* **a pool the report never reached is not a refusal.** `--once` on a quiet day, or with a
  `SCAN_LIMIT` that never saw this pool, used to print *hold — the market has no pool named
  like this vault* about the one address the table vouches for. It now reads
  `[UNMATCHED]` and supplies: the address does not need a label to be a real pool, and the
  record that priced it is the feed's history of the pinned uuid. The verdict is still
  printed, and `--book-report` says `matched by declaration` when that is how it was found.

Your first dollar, end to end:

```bash
# .env — Arbitrum One, a funded key, the example book with its Aave entry
TESTNET_PROFILE=arbitrum
TESTNET_ALLOW_MAINNET=1          # plus allow_mainnet=True in the caller: two yeses
TESTNET_HISTORY=auto
TEST_STAKE_AMOUNT_USD=1.0        # one dollar first, whatever the other ceilings allow
DRY_RUN=1                        # look before signing

python -m agent.trade.loop --book-report     # "stakable: yes" on the Aave line
python -m agent.trade.loop --once            # prints the $1 supply and signs nothing
DRY_RUN=0 python -m agent.trade.loop --once  # signs it: approve + supply, then a receipt
```

`FORCE_TRADE=1` (or `--force-trade`) is the flag for the run whose *only* job is to prove the
pipe: it lets an `aave-v3` entry be supplied while the market has nothing to say about it —
a calm day, a pool the hunter could not measure, a report that never reached it — and it
says so out loud before the amount:

```text
   force  FORCE_TRADE is on — an aave-v3 entry may be supplied while the market has no verdict for it; every ceiling still applies
Force trade: I would supply $1.00 USDC to Aave.
```

It is a test tool, and the limits are what make it safe to have: it never lifts a ceiling
(`TEST_STAKE_AMOUNT_USD` still sizes the send), never signs in a dry run, never skips the
deployment table, and never overrules a hunter that *convicted* the pool — `THIN`,
`FORMULA` and `DISHONEST` are evidence, and a flag on a command line is not. A live run says
`Force trade: I supplied $1.00 USDC to Aave.` only after the venue reports the supply
landed; a refused send is quoted instead of claimed.

`stake` records the *asset* amount it supplied; a withdrawal asks the Pool for that amount
of the asset, so interest earned above it stays in the position as aToken dust until the
operator takes it out.

#### The fee is bid above the base fee, because the base fee moves

The first real transaction this venue ever signed — the `approve` before that dollar of
supply — was refused by Arbitrum One:

```text
approve 0xaf88… (1.0 credits) failed: the venue refused (max fee per gas less than block
base fee: maxFeePerGas: 20016000 baseFee: 20046000)
```

Nothing was estimated wrongly. The fee was signed *at* `eth_gasPrice` and the base fee rose
by 3 000 wei — a tenth of a percent — before the sequencer checked the signature. A cap at
exactly the current base fee can only be mined in the block it was read in, which on an L2
is a coin flip. So the venue now prices a transaction the way web3.py documents it and then
adds a margin:

* **a chain with a base fee** signs a type-2 transaction: `maxPriorityFeePerGas` is the
  node's own `eth_maxPriorityFeePerGas` (0 on Arbitrum, and a tip is not a ceiling, so it
  is never multiplied), and `maxFeePerGas` is `2 × base_fee + tip`, then **+12%**
  (`FEE_BUFFER_PERCENT = 112` in `agent/physics/real_testnet_provider.py`). Two blocks of
  drift are already inside the `2 ×`, and the 12% covers the gap after the estimate;
* **a chain without one** (Sepolia, Anvil, anything pre-1559) still signs one legacy
  `gasPrice`, buffered the same way;
* **integer arithmetic** — `(x × 112 + 99) // 100` — so the margin cannot be rounded away
  at small numbers: `int(7 × 1.12)` is 7, which is no margin at all.

A refusal still quotes the node's words, and now also both sides of the bid, with the base
fee re-read *after* the failure so the gap is a number rather than a guess:

```text
approve 0xaf88… (1.0 credits) failed: the venue refused (max fee per gas less than block
base fee: …) [the fee this file bid: maxFeePerGas 44835840 wei (0.044835840 gwei); tip 0
wei (0.000000000 gwei); the base fee reads 20046000 wei (0.020046000 gwei) after the
refusal; 24789840 wei (0.024789840 gwei) to spare]
```

That last clause is the one that answers the question an operator actually has: was the
*bid* short, or is something else wrong? `short` means the buffer is too small for this
chain (raise `FEE_BUFFER_PERCENT`); `to spare` means the cap was never the problem.
`scripts/first_breath.py` prices its one transfer by the same imported rule rather than a
second copy of it — two copies of a safety margin is one copy too many.

### Dry run is the default, and it is not a flag inside the venue

`DRY_RUN` unset is a dry run; only `0/false/no/off` makes the loop sign, so a typo
(`DRY_RUN=ture`) leaves it in the safe mode. In a dry run the branch returns **before**
`execute_stake` is called at all — `test_a_dry_cycle_decides_and_signs_nothing` asserts on
an empty venue ledger, not on a boolean, because a boolean inside a method is a promise and
an empty ledger is a fact.

### The state file, and why a corrupt one stops the loop

Positions live in `TRADER_STATE` (`/data/trader_state.json` in the container) and are
written after every cycle: address, amount, the verdict it was bought on, the confidence.
It is what stops a restart from staking the same money twice — so a file that cannot be read
is **not** treated as an empty portfolio. The process stops and names the file.

Three details make the file trustworthy enough to hold the only memory a loop has:

* **the write is atomic** — a temporary neighbour, an `fsync`, an `os.replace`, and a
  directory sync. A power cut leaves either the old document or the new one, never half of
  either; a truncated file would be read as a stopped agent on the next boot.
* **the position is written *before* it is signed** (a write-ahead). A kill between the
  signature and the save was the one window in which a restart would stake the same dollars a
  second time — so the position goes to disk first, and it is taken back out (`_unremember`,
  with the file rewritten immediately) if the venue refuses or the send raises. A kill
  *during* the send keeps it: the loop cannot know whether that signature moved money, and
  "unknown" costs one cycle of "already at the ceiling", while guessing "nothing happened"
  costs the same dollars twice.
* **a live loop that cannot write it stops** (`TradeError`) instead of trading blind. A dry
  run, where nothing is at risk, downgrades the same failure to a warning.

### Leaving a position: a conviction lost, or a panic that passed

A position is closed for exactly two reasons, and both are spelled out in
`agent/trade/loop.py` as constants rather than as conditions inside a method:

| verdicts | what happened | what the loop does |
| --- | --- | --- |
| `THIN`, `FORMULA`, `DISHONEST` (`EXIT_VERDICTS`) | the pool is no longer what it was bought as | withdraw |
| `CALM`, `GROWTH` (`TAKE_PROFIT_VERDICTS`) **on a position opened on `PANIC`** | the dislocation *was* the thesis, and it has closed | withdraw (take-profit) |
| `CALM`, `GROWTH` on a position opened on `CALM` | nothing: it was bought for carry | hold |
| `UNMEASURED` | the feed stopped publishing, which is not evidence against the pool | hold |
| the vault published no record this cycle | "no answer today" is not a sale | hold, and say so in a warning |

The take-profit is the difference between a loop that can only leave a pool when something
goes *wrong* with it and one that also banks a trade when something goes *right*: without it
every winner stayed open until its verdict turned, which means the only exits were bad ones.

Two further rules are what make the exit an exit rather than a gesture:

* **the amount is what the venue says, not what the file remembers.** An Aave position is an
  aToken balance — principal plus interest — and withdrawing the recorded principal would
  leave the profit in the pool. A zero from a chain that answered is a *fact*: the stale
  entry is dropped and nothing is signed, because a withdrawal of nothing is a revert and a
  fee.
* **a position that is leaving is not topped up on the way out.** The closing set is computed
  before the entries are considered, so a `CALM` cycle cannot supply a dollar to an
  `aave-v3` entry and withdraw it in the same block — two fees to stand still.

Tests: `tests/test_trade_loop.py` — 94 of them, no network and no chain: the caps, the
fail-closed refusals, the dry run's empty ledger, the exit path (a conviction, a take-profit,
the whole aToken balance, a position the chain says is gone), the write-ahead and the refusal
that takes it back, the state file's refusal to lie, the widths of the hunt
(`SCAN_LIMIT`/`MEASURE_LIMIT`, and the flag that beats them), the address book's two kinds of
empty, the feed-priced hold (a record the feed vouched for is measured, printed and never
signed), the book report, and the CLI's mode. The wire behind it
has its own: `tests/test_physics_real_testnet.py` (eighty, twelve of them about
`TESTNET_HISTORY` — the default is the vault's own log, `auto` prices a silent vault from the
feed and keeps the log when there is one, `feed` never touches the node, `steps` takes a
tail, a pinned `pool` reaches the feed and a malformed one is refused by name, a name the
feed does not know is no record *and* a reason, a raising feed is a sentence rather than a
cycle, an unknown policy is a named error) and `tests/test_physics_defillama.py` (forty-five,
nine of them about the name→pool bridge: the deepest of several, the chain filter and where a
name lives instead, the same pair spelled either way, a pinned uuid that is never re-matched
by name, a dead chart as a reason, the tail).

## Backtesting: the same hunt, on history, in an hour (Phase 9)

The live cycle answers one question per cycle, with money. `agent/backtest/` asks the same
question of a price history, in isolation, and answers with a report instead of a signature:
it walks a public candle feed bar by bar through the *same* `PanicDetector` and
`risk_assessment` the loop uses, and buys and sells on the *same* verdict table.

```bash
# 180 days of 4-hour BTC/USDT from HTX's public kline endpoint, a $1000 book
python -m agent.backtest.run --exchange htx --symbol btcusdt --days 180 --capital 1000

# the same, offline, from a chart an operator can edit in a spreadsheet
python -m agent.backtest.run --exchange csv --csv chart.csv --capital 1000
```

Nothing in `agent/physics/` changed for this: the hunter is read, never bent. There is no
RPC, no key, no signature and no transaction anywhere in the package, and it never touches
`data/trader_state.json` — the live loop's memory is not a backtest's business.

### The one invented quantity: what a candle pays a buyer

The hunter's two inputs are a pool's **APY** (per cent a year) and its **depth** (dollars of
TVL). A candle feed publishes neither, so the package states the bridge once, in
`agent/backtest/data.py`, and prints it in every report:

| the hunter wants | a candle gives | the reasoning |
| --- | --- | --- |
| `apy` — the rate a holder earns | the bar's **decline**, in per cent (`max(0, -return)`) | a pool pays in APY; a chart pays in a lower price. A bar that fell 2.5 % paid a buyer 2.5; a bar that rose paid nothing |
| `tvl` — the depth | **quote turnover** (`vol` on HTX; `amount × close` when a feed publishes only base volume) | depth is what leaves when people run; turnover is the same measurement for an instrument |

Annualising a bar (`(1+r)^365-1`) was rejected because it produces thousands of per cent, and
`risk_assessment` would convict *every* record of promising more than `PLAUSIBLE_APY` allows —
a backtest refused by its own claim ceiling measures nothing. A signed return was rejected
because a pool's APY cannot be negative and `to_returns` maps every non-positive growth to
`-1.0`: half a chart of down bars would arrive as a record of total losses and every honest
instrument would be convicted as a ghost.

So `PANIC` on a chart reads exactly as it does on a pool: *the instrument is suddenly paying
far more than its own rest while participation drained* — a capitulation on drying volume.
`GROWTH` reads as *the rate rose and the depth rose with it*, which on a chart is the same
fall on rising volume.

### The files, and what each one owns

| file | what it is |
| --- | --- |
| `data.py` | candles from HTX (`/market/history/kline`, 2000 bars a page — the venue ignores `from`/`to`, so there is no pagination to get wrong) or from a CSV (`timestamp,open,high,low,close,volume`); the decline/turnover conversion; the venue's ticker list, ranked by turnover, for a sweep; one sentence per failure |
| `simulated_provider.py` | a venue that answers the `IWeb3Provider` operations from the candles: `execute_stake`, `withdraw`, `balance`, `advance` (the clock is a *bar*), plus an empty whitelist and a proof channel that always refuses |
| `engine.py` | the decisions: one verdict per bar on the prefix that ends there, the entry table, the exit table, the stop, and the equity row per bar |
| `reporter.py` | the readings: ROI, max drawdown, win rate, trades, mean hold, exposure, fees — and **buy & hold**, the benchmark that keeps an honest backtest honest |
| `run.py`, `__main__.py` | one pair, one command, one report |
| `sweep.py` | many pairs, one table: the venue's top pairs by turnover, a row per pair (with ROI, buy & hold and the difference), and `sweep.csv` |

### The fills are assumptions, and each one is a field

* **slippage 0.1 %** against the trader on every fill, both ways;
* **fee 0.1 %** per side, printed in total so a strategy that only works before costs is
  visible instead of impressive;
* **latency one bar**, so an order is filled at the *next* bar's open — the close a decision
  was made on has already happened by the time it could arrive. `--latency-bars 0` is the
  optimistic case a reader should compare against;
* **the stop fills inside the bar that crosses it** (at the stop price), and **at the open
  when the bar gapped below it** — a backtest that filled a gap at the stop would invent the
  difference on every crash.

### The exit table is the live loop's table

| verdict | on a position opened on `PANIC` | on a position opened on `CALM`/`GROWTH` |
| --- | --- | --- |
| `CALM`, `GROWTH` (`TAKE_PROFIT_VERDICTS`) | sell — the dislocation has closed | hold: it was bought for carry and has no dislocation to have passed |
| `THIN`, `FORMULA`, `DISHONEST` (`EXIT_VERDICTS`) | sell — the record stopped being what it was bought as | sell |
| `UNMEASURED` | hold — blindness is not a verdict | hold |
| the stop (`--stop 0.02`) | sell, by the venue's own order, with no verdict attached | sell |

A test compares the two packages' constants (`test_the_verdict_sets_are_the_live_loops_sets`)
rather than restating them, because a backtest of a different table would be a backtest of a
different strategy.

### The report: three files, and the number that shames a strategy

`--out data/backtest` (the default) receives `backtest_report.json` (settings, the window, the
venue's assumptions, every trade, the whole equity curve, the verdict histogram, the warnings),
`backtest_trades.csv` (one row per round trip, ISO timestamps, the delimiter Excel expects) and
`backtest_equity.csv` (the book bar by bar — the curve the drawdown is measured on).
`--delimiter ';'` writes the locale that wants one.

```
── backtest · btcusdt 4hour · 2000 bars · 2025-12-13 → 2026-11-11
   capital $1,000.00 → $994.47 · ROI -0.55% · buy & hold -17.64%
   max drawdown 2.52% · win rate 50.0% (1/2) · profit factor 0.76
   trades 2 closed + 0 open · mean hold 0.5 bars (0.1 d) · fees $4.03 · exposure 0%
   exits take-profit 1, stop 1
   verdicts UNMEASURED 64 · CALM 1933 · PANIC 3 · judged 1936 bars
   warn  2400 bars of 4hour were asked for and the venue serves at most 2000 in one
         answer, so the window is 2000 bars (333.3 days) instead of 400
```

Exit codes mean what they mean everywhere else in this layer: `0` answered (whatever it
earned), `2` a configuration that cannot mean anything, `3` a feed that did not answer, `4` a
window the detector cannot speak on (fewer bars than `--min-points`). A window the venue
cannot serve, and a strategy that never got a decision to make, are printed as warnings —
never swallowed. A stop followed by a re-entry on the next bar is the strategy as written;
`--cooldown-bars 30` (the hunter's own `PANIC_COOLDOWN`) makes one episode one bet instead of
one bet per bar, and the report tells you which it was.

### One pair is an anecdote: the sweep, over the market's own top twenty

A single pair over a single window answers "what would this have paid" — one instrument in one
regime. The next question is whether the thesis holds *across* the market, and the only honest
universe for that is the venue's own: `agent/backtest/sweep.py` ranks HTX's ticker list by
24-hour quote turnover and runs the same engine once per pair, writing one report per pair plus
a `sweep.csv` and printing one table.

```bash
# the twenty pairs HTX traded most, on daily bars for 5.5 years (2000 = the venue's cap)
python -m agent.backtest.sweep --top 20 --period 1day --days 2000 --capital 1000

# a list an operator chose, or a finer bar over a shorter window
python -m agent.backtest.sweep --symbols btcusdt,ethusdt,solusdt --days 400
python -m agent.backtest.sweep --top 20 --period 4hour --days 333 --cooldown-bars 30
```

Real output, 400 daily bars, 2025-10-08 to 2026-11-11 — a bear year:

```
── sweep · 3 pairs · 1day · 400 days asked · $1,000.00 each · entries panic
   · btcusdt      ROI +3.01% · buy & hold -38.39% · 3 trades
   · ethusdt      ROI +14.57% · buy & hold -51.41% · 5 trades
   · solusdt      ROI +13.80% · buy & hold -60.97% · 5 trades

symbol   bars  window                 trades      roi  buy & hold     edge  drawdown  win    fees
───────  ────  ─────────────────────  ──────  ───────  ──────────  ───────  ────────  ───  ──────
btcusdt   400  2025-10-08→2026-11-11       3   +3.01%     -38.39%  +41.40%     4.10%  67%   $6.09
ethusdt   400  2025-10-08→2026-11-11       5  +14.57%     -51.41%  +65.98%     4.70%  40%  $10.10
solusdt   400  2025-10-08→2026-11-11       5  +13.80%     -60.97%  +74.76%     6.72%  40%  $10.11
   3 pairs measured · mean ROI +10.46% · median +13.80% · mean buy & hold -50.26% ·
   mean edge +60.72% · beat buy & hold 3/3 · 13 trades · $26.30 in fees
```

Three columns are the argument: `roi`, `buy & hold`, and `edge` — their difference. A pair
where the strategy lost 5 % while the coin lost 40 % is a pair where the thesis worked; a pair
where it made 8 % while the coin made 60 % is a pair where it merely stood in the way less.
Sorting by ROI alone cannot tell the two apart. A failure is a *row*, never a gap: a dead feed
or a pair listed too recently keeps its place with the sentence that explains it, and
`sweep.csv` keeps the fractions, the exit mix and that sentence for a spreadsheet.

And the long window is the one that argues back. The same engine on the venue's own top
twenty (daily bars, up to 2000 = the venue's cap, so the oldest pairs start 2021-05-22),
measured 19 of 20:

```
symbol        bars  window                 trades      roi  buy & hold       edge  drawdown  win     fees
────────────  ────  ─────────────────────  ──────  ───────  ──────────  ─────────  ────────  ───  ───────
apepeusdt      476  2025-07-24→2026-11-11       2   -4.53%     -25.88%    +21.35%     4.53%   0%    $3.91
apeusdt       1702  2022-03-16→2026-11-11     141  -69.72%     -98.66%    +28.94%    70.18%  19%  $159.28
btcusdt       2000  2021-05-22→2026-11-11      12  +13.66%    +123.47%   -109.81%     6.35%  50%   $26.04
bttusdt       1749  2022-01-28→2026-11-11      12  +10.63%     -83.25%    +93.89%     8.03%  33%   $23.97
dogeusdt      2000  2021-05-22→2026-11-11      58  -23.80%     -76.03%    +52.22%    44.35%  31%   $89.36
ethusdt       2000  2021-05-22→2026-11-11      43  -27.36%      +8.66%    -36.02%    48.07%  16%   $75.00
htxusdt       1015  2024-02-01→2026-11-11       8   +0.30%     -15.47%    +15.77%     6.72%  38%   $15.61
hypeusdt       266  2026-02-19→2026-11-11     0+1   +2.86%    +149.44%   -146.58%     0.00%   0%    $1.00
jstusdt       2000  2021-05-22→2026-11-11      53   -8.38%    +163.84%   -172.23%    25.81%  28%   $98.89
marscoinusdt    70  2026-09-03→2026-11-11       0   +0.00%     -76.15%    +76.15%     0.00%   0%    $0.00
qntusdt          —  —                           —        —           —          —         —    —        —
pepeusdt      1304  2023-04-18→2026-11-11    35+1  -32.70%   +1506.45%  -1539.16%    34.66%  17%   $58.39
solusdt       2000  2021-05-22→2026-11-11      79  -17.73%    +271.89%   -289.62%    62.96%  16%  $113.65
suiusdt       1290  2023-05-02→2026-11-11      28  -16.09%     -34.77%    +18.68%    21.33%  21%   $51.62
usatusdt       289  2026-01-27→2026-11-11       0   +0.00%      -0.21%     +0.21%     0.00%   0%    $0.00
winusdt       1715  2022-03-03→2026-11-11      33   -8.01%     -78.16%    +70.15%    21.57%  21%   $59.29
xlmusdt       2000  2021-05-22→2026-11-11      36  -14.88%     -51.72%    +36.84%    29.45%  31%   $58.89
xmrusdt       1414  2022-12-29→2026-11-11       1   -2.29%    +314.73%   -317.03%     2.29%   0%    $1.98
xrpusdt       2000  2021-05-22→2026-11-11      31  -13.15%     +49.90%    -63.05%    26.35%  32%   $53.61
zecusdt       1423  2022-12-20→2026-11-11       5   +2.32%   +2272.87%  -2270.55%     4.70%  60%   $10.23
   ! qntusdt — 36 bars of 1day is not enough for a verdict — the detector needs more than 64;
     this pair was listed too recently for the window, or the window is too short
   19 pairs measured, 1 failed · mean ROI -10.99% · median -8.01% · mean buy & hold
   +227.42% · mean edge -238.41% · median edge +0.21% · beat buy & hold 10/19 · 577 trades
   · $900.73 in fees
```

Read the three numbers together and the bear-year rows above stop being a thesis. Over five
years the strategy's **median** pair returned -8.01 % and its **median edge was +0.21 %** — a
coin flip against holding — while the **mean** edge is -238 %, because ten of nineteen pairs
were measured against a coin that multiplied (ZEC +2272 %, PEPE +1506 %, SOL +272 %) and no
exit rule that sells within days can sit through that. Where the thesis does work it works as
designed: DOGE, APE, SUI, WIN, XLM and BTT all fell harder than the strategy did, and BTC's
+13.66 % came with a 6.35 % drawdown against 123 % of holding risk. The honest summary of this
first sweep is *"the fade of a capitulation is a defensive trade, not an offensive one"* — and
the next experiment is a shorter bar (4hour, 333 days) or a cooldown, not a longer one.



### The exit grid: ten books, one question, and no verdict

`walkforward.py` is the backtest that refuses to answer. It fits the policy on the past, sells
into the future a fold at a time, pools every fold and pair into one reading per ablation, and
then asks the question the sweep cannot: *could this many trades have seen the gate at all?*
The arithmetic is `agent/backtest/power.py` — a gate of 30 bps read at α 0.05 with power 0.9
needs a spread and a count, and a book with too little of either prints **unmeasured**, not
**fail**, because a sample that cannot see an edge has not disproved one.

Ten books, one command (`bash scripts/exit_grid.sh`, about 25 minutes on 16 cores), and
`python scripts/exit_grid_report.py` reads the `criteria.json` files back as this table:

```
book                       period   folds  trades  bps/trade       t   σ bps  ± bps  needed
base (fixed 2 % stop)      4hour        5       8       +2.7   +0.04     174    233     356
base (ATR-2 stop)          4hour        5       8      +57.7   +1.43     114    153     155
ATR-2                      4hour        5       8      +57.7   +1.43     114    153     155
ATR-2 · risk 2 %           4hour        5       8      +57.7   +1.43     114    153     155
ATR-2 · take-profit 2R     4hour        5       8      +57.7   +1.43     114    153     155
ATR-2 · 2R + 1R trail      4hour        5       8      +57.7   +1.43     114    153     155
ATR-2 · time stop 48 bars  4hour        5       8      +57.7   +1.43     114    153     155
base                      1day        14      31      +76.6   +1.00     428    257    2140
base (ATR-2 stop)         1day        14      15      +22.7   +0.12     748    674    6533
ATR-2 · take-profit 2R     1day        14      15      +22.7   +0.12     748    674    6533
```

Read the middle column first: eight trades on the four-hour book and thirty-one on the daily
one, against 155 and 2140 needed. **Not one row resolves the gate**, and the honest verdict on
all ten is `unmeasured` — the gate is not cleared and not refused.

What the grid did establish is which knob moves the number that *could* be resolved later. The
stop is the only one that did anything: a fixed 2 % stop leaves +2.7 bps a trade against a 174
bps spread, while an ATR-2 stop leaves +57.7 against 114 — the money a trade earns rises
twenty-onefold and the spread falls by a third. Take-profit, trailing and the time stop moved
nothing, and the reason is a property of the bars rather than a wiring bug. A direct probe of
the same exits on one pair (btcusdt, daily, 2000 bars) shows the shape: nine trades, mean hold
**2.7 bars**, and the exits read `take-profit 7, stop 2` — the signal closes seven of nine
before a 2R target, a 1R trail or a clock can fire. Asking the clock for five bars moves
exactly one trade and takes the profit factor from 0.89 to 0.88; the knobs work, there is
nothing for them to grab.

The daily book is *worse*, which is the opposite of what "more history" suggests: fourteen
folds and half as many trades per fold, with a spread of 428–748 bps because every trade is a
week-long swing. Resolution is bought with **count**, and the count is bought with breadth —
`4hour` is capped at 2000 bars by the venue, so a longer window is not on offer; ~1.6 trades
per pair per year means the 155 trades this gate needs are about a hundred pairs, not five
years.

Every row above is checkable afterwards, which the grid also had to fix. Each run writes
`criteria.json` beside its table: the gate, its sha256 from `criteria.py`, the exits it ran
with, and the sha256 of the bars it read — so a table and the criteria it was judged against
cannot drift apart without the hash saying so. A live run is deterministic only *given* its
candles, and the venue's candles move (one new bar closes and every fold boundary shifts), so
the same command an hour later is a different experiment. `--save-candles` freezes what a run
read and `--candles` replays it:

```
python -m agent.backtest.walkforward --symbols btcusdt --period 4hour --days 340 \
    --save-candles data/frozen/4hour --out data/grid/verify-4hour
bash scripts/exit_grid.sh --replay data/frozen/4hour 4hour
```

`data/frozen/` holds those bars (one pair, two periods, ~280 kB), and replaying them on another
machine must print the same 64-character hash for `walkforward.csv`. On this machine it does,
byte for byte: `b3156728c363fa4f50ef00910b0436c23c2c358c95387a5d7a4250ca08e54076`.

Getting that check to run inside the deployment image found a real packaging bug: `power.py`
imports `scipy` to read the gate's power from a noncentral t distribution, and `scipy` was
declared only in the `organism` extra — so a base install, including the image's own extras,
could not import `agent.backtest.walkforward` at all. The gate's arithmetic is part of the
verdict, so `scipy` is now a base dependency. With that fixed, an image built from this commit
reproduces the hash in the container (Python 3.12.16, site-packages) exactly as the host does
(Python 3.12.3, source tree):

```
docker compose build trader
mkdir -p /tmp/enrg-replay
docker compose run --rm -v "$PWD/data/frozen:/frozen:ro" -v /tmp/enrg-replay:/out \
    -w /frozen trader python -m agent.backtest.walkforward \
    --candles /frozen/4hour --period 4hour \
    --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02 --out /out
sha256sum /tmp/enrg-replay/walkforward.csv
```


Tests: `tests/test_backtest_*.py` — 276 of them, and **not one opens a socket**: the transport
is a parameter, the venue is a list of hand-written bars, the chart is a CSV in `tmp_path`, and
the sweep's universe is a dictionary. What they pin: the newest-first answer reversed once,
quote-versus-base volume, every failure as a sentence, the decline mapping (a bar that rose
pays 0, a warm-up bar pays `nan`), the settlement lag, the slippage and the fee on both sides,
the stop inside a bar and through a gap, one position at a time, the refusal with its reason,
the exit table against the live loop's constants, **the prefix every verdict was measured on**
(the detector is wrapped and every prefix length recorded, so a lookahead is a failing test and
not a suspicion), the metrics read from a hand-built curve, the three files, the CLI's four exit
codes, the universe's ranking and its tie-break, a failed pair kept in the table as a row, the
sweep's totals, the criteria file that survives an infinite measurement, the freeze that reads
back as the same floats, the short ledger's stop above, target below, trail down and the gap that
fills at the open through each of them, the mirror that makes `climbs` on a rising chart equal
`declines` on its reflection, and the wall itself: no module in `agent/backtest/` imports `web3`,
`agent.trade`, `real_testnet_provider` or `subprocess`, and the only `urllib` import in the
package is *inside* the default transport, where `--csv` and `fetch=` never reach it.


### The other side: a short book, and the count it needs

Everything above measures one book, and it buys. That is not a preference: the detector's question
— *is this a panic?* — is a question about a **fall with the depth drained out of it**, so a fall is
the shape it was written to recognise. The same question has a mirror, though, and a rise whose
turnover drains as it climbs is the shape a short seller is looking for. `--side` asks for it:

```
python -m agent.backtest.walkforward --top 100 --period 4hour --days 340 --side both
python -m agent.backtest.walkforward --symbols btcusdt --period 4hour --days 340 --side short
```

`climbs()` in `data.py` is `declines()` with the sign turned: the same `nan` warm-up over the
detector's record, and a bar that fell pays 0 where `declines` pays 0 for a bar that rose. It is a
second function rather than `-declines` because the negation clamps at zero — a short book handed
the long book's series would never fire, and a table of zeros reads like a market with no setups
instead of a sign error. Nothing else in the detector changes: features, horizons, the policy, the
labels' cost and the gate are the same code, so a `short:policy+atr` row can be read beside
`policy+atr`. The side is in the row's *name* because a table whose rows differ by a setting that is
not in the row's name is a table of anonymous numbers.

The book is the mirror, in `simulated_provider.py`. A short sells units it does not own, so the
position is **negative units**, the sale's proceeds sit in `cash`, and `equity = cash + units ×
price` stays sign-correct on both sides with no second formula. A stop sits *above* the fill and a
target *below* it; a bar that gaps through either fills at **the open**, which is worse than the
stop the position named and is the gap's real money; the trail ratchets down and never up; and both
sides reset through one `_flat()`, so a long and a short cannot leave different residue behind.
Fees and slippage are charged both ways.

What is **not** modelled is named rather than zeroed: no borrow rate, no funding, no liquidation —
a 1× price position, not a financed one. A short row read as if it had paid borrow would be read as
a worse book than it is, and one read as if it could be liquidated would be read as a safer book
than it is, so every run that measures a short book prints that sentence above its table:

```
── walkforward · 1 pairs · 4hour · long / short · exits: stop atr · 2x ATR · risk 2.00%
── walk-forward · 1 pairs · 4hour · 2000 days asked · benchmark btcusdt · gate 30 bps / t 2 / pf 1.3 / sharpe 1
   shorts: 1x, fee and slippage charged both ways, no borrow, no funding, no liquidation — a price position, not a financed one
```

Run that over `data/frozen/4hour` and the two books answer differently on the same bars, which is
the point: two trades for the long book, **zero** for the short one, and the short rows read
`unmeasured — 0 trade(s): no dispersion, so no resolution` rather than `fail`. btcusdt fell over that
window; a short book with nothing to sell is a count, not a verdict, and the gate says so in as many
words. It is one pair, and it is the reason the next run is a hundred.

On the daily frozen pair the short book does trade, and it loses: five trades, −108 bps a trade,
profit factor **0.00** — every one of the five — against the long book's +79 bps and pf 1.87 over the
same three folds and the same bars. Five trades is not a verdict (that row is `unmeasured`: 221 bps
of spread against a gate of 30, so 335 trades would be needed), and one pair is not a universe. What
the row does establish is that the mirror is a *different book with a different answer*, not the
same book relabelled — which is the only way a two-sided agent can be measured at all.

Three things hold by construction. `--side both` reads the candles **once** and runs two detector
passes over mirrored series, so the short book's folds, warm-up and labels are the same measurement
as the long book's, and the two never share a row in the pooled table. `climbs` on a mirrored series
is `declines` on the original number for number, so a difference between the two books is a
difference in the market and not in the measurement — that is a test, not a claim. And the replay
hash is unchanged, because a long book's arithmetic never sees a short: the same frozen directory
still gives `b3156728c363fa4f50ef00910b0436c23c2c358c95387a5d7a4250ca08e54076` after this, byte for
byte.

The side is worth the code for the reason the exit grid already found: **resolution is bought with
count**, and the gate needs 155 trades on the four-hour book against the 8 that one pair produces.
One pair, one side, ~1.6 trades a year; `--side both` doubles the count without doubling the
download or the folds, and `--top 100` is the other half of the same multiplication. That run is the
next experiment and it is *not* in this commit — what is here is the machinery to make it, verified
on hand-built charts (a pump chart the short book trades and the long book refuses), on the frozen
replay's byte-identical hash, and by 27 tests across both books and both ledgers.


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

1135 tests, no network, no key, no chain. Three dependencies are *workspace* ones,
and are named rather than hidden: the Anvil tests skip when foundry is absent, the
tests for the sibling `axis_core` package skip when that package is not installed,
and the `entrain_infer` core plus the scikit-learn datasets are needed by the
genome/sensorium/organism tests — on a box without them those tests say so by
name, which is why the full run is a *workspace* command, not a pip-install one.
Nothing on the deployment path needs any of the three: the trading cycle's own 85
(`tests/test_trade_loop.py`) and the provider's 109 run anywhere Python does — one of
those asks for foundry and one for `web3`, and each skips itself by name rather than
passing quietly. The backtest's 276 (`tests/test_backtest_*.py`) are in the same
category and even further from a network: the transport is a parameter, so a test hands
the loader a dictionary and a run hands it a CSV.


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
* **A suite that runs like a deployment.** 850 tests, no network, no key, no
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
# the whitelist: nothing outside this book can be staked into, and an entry has to be
# a vault of this interface (stake/withdraw/YieldDistributed) — see book.example.json.
# In the container the path is /data/book.json; on a host, ./book.json
TESTNET_PROTOCOLS_FILE=/data/book.json
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

#### 5. The trader's own unit (24/7, no Docker)

The same shape, pointed at the cycle that moves money. One cycle every five minutes,
positions remembered in `/var/lib/enrg-ai/trader_state.json`:

```ini
[Unit]
Description=ENRG-AI trading cycle
After=network-online.target
Wants=network-online.target
# A state file that cannot be read stops the loop *on purpose*, so a restart storm would only
# bury the message it stopped to print. Five failed starts in five minutes and systemd leaves
# the unit `failed` with the reason in the journal — visible, and not a loop that trades while
# nobody is watching.
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
User=enrg
WorkingDirectory=/opt/ENRG-AI
EnvironmentFile=/opt/ENRG-AI/.env
ExecStart=/opt/ENRG-AI/.venv/bin/python -m agent.trade.loop \
          --loop --interval 300 \
          --state /var/lib/enrg-ai/trader_state.json \
          --json /var/lib/enrg-ai/trader_cycle.json
Restart=always
RestartSec=15
# A stop is SIGTERM, and the loop's own `--interval` sleep is what it interrupts:
# a position is never mid-write, because the write is atomic.
KillSignal=SIGTERM
TimeoutStopSec=30
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/enrg-ai

[Install]
WantedBy=multi-user.target
```

```bash
sudo install -d -o enrg -g enrg /var/lib/enrg-ai
sudo systemctl daemon-reload && sudo systemctl enable --now enrg-ai-trader
journalctl -u enrg-ai-trader -f          # one block per cycle
```

`Restart=always` is the whole auto-restart story, and it is safe to lean on **because a
corrupt state file stops the loop by name** (see above): a restart after a failure either
finds a whole file and continues, or refuses to trade and says which file to look at. A loop
that retried from an empty portfolio on every crash would stake the same money again every
`RestartSec`. `StartLimitBurst` is the other half — a failure that repeats is a *report*, not
a rhythm, so after five identical starts systemd stops trying and the unit stays `failed`
until someone reads `journalctl -u enrg-ai-trader`.

#### 6. The fast way: tmux (a look, not a deployment)

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
docker compose up -d --build                   # `trader`, the trading cycle
docker compose logs -f trader                  # the decisions, one cycle at a time
docker compose run --rm coherence              # one-shot job, behind the `jobs` profile
docker compose --profile training up -d trainer  # the training loop, when you want it
docker compose exec trader pytest -q           # the suite, as far as it ships in the image
```

Three services, one image, and the default one **hunts**: `trader` runs
`python -m agent.trade.loop --loop --interval ${TRADE_INTERVAL:-300}` and, with
`DRY_RUN=1` (the default, and what an unset variable means), it computes and logs every
decision without signing anything. `trainer` is the self-training loop and now lives behind
`--profile training`, so a box that trades does not also re-fit itself every hour unless
someone asked; `coherence` is the same seam as a one-shot job.

The `mkdir`/`chown` is not decoration: `data/` is gitignored and `.dockerignore`d,
so on a fresh box compose creates it as `root:root` and the container — uid
`10001`, on purpose — cannot write its state file. The symptom is a loop that
dies with `PermissionError: [Errno 13] ... /data/trader_state.json` on the first
cycle; the fix is the line above, once.

The image is `python:3.12-slim` plus the package's extras, runs as uid `10001`,
and holds **no key, no `.env` and no address** — both are `.dockerignore`d, so
`docker build` cannot bake a secret into a layer. One health check, and it is the
honest one: both loops write their state file at the end of every cycle, so a
`/data/trader_state.json` (or `/data/digital_state.json`) older than ninety minutes
means the loop is not working and compose restarts it.

#### And it comes back after a reboot

`restart: unless-stopped` is the container's half of "runs forever"; the other half is
Docker's own service, which not every distribution enables at boot:

```bash
sudo systemctl enable --now docker                        # the daemon returns after a reboot
docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' \
     "$(docker compose ps -q trader)"                     # must say: unless-stopped
docker compose ps                                         # one `trader`, `Up`
```

Positions survive all of it because they live on the `./data` volume
(`/data/trader_state.json`), which compose re-attaches on every recreate — and because the
file is written atomically *before* a stake is signed (see "The state file" above), a
`reboot` in the middle of a send is the one case that errs toward "hold", never toward a
second stake.

One variable decides whether a deployment is a hunt or a rehearsal, and its absence is the
rehearsal: `DRY_RUN` (`1`/unset signs nothing, `0` signs). Check the other one before leaving
the box unattended — `FORCE_TRADE` must be **empty**. It is a test tool that lets an
`aave-v3` entry be supplied with no verdict behind it, and a 24/7 loop that leaves it on is a
loop buying on nothing:

```bash
grep -E '^(DRY_RUN|FORCE_TRADE)=' .env    # → DRY_RUN=0 and no FORCE_TRADE line
docker compose logs --tail 40 trader      # the last cycles say which mode they were in
```

The `.env` from step 3 is the file compose reads: the same
`TESTNET_PROFILE=arbitrum`, the same key, and the same `DRY_RUN`, handed to the
container's environment at start — never copied into the image, never a build
argument. **`DRY_RUN=1` until the logs have been read**: `docker compose logs -f
trader` prints one block per cycle, and the block says which mode it was in.


The image carries the suite and the `dev` extra so it can be *measured*, not
because a deploy needs them. Measured on this repository's own build
(`python:3.12-slim`, 589 MB, uid `10001`):

* **built and run with no network**: one cycle with the node and the feed both
  unreachable printed the two warnings (the book could not be read, the feed could
  not be reached), exited `0`, and wrote `/data/trader_state.json` **and**
  `/data/trader_cycle.json` — a container that dies when its inputs are missing is a
  container that hides a failure behind a restart loop;
* **the health check** answers `1` with no state file and `0` with a fresh one;
* **`pytest -q` inside the image** collects 699 and passes 623. The 76 that do not run
  there need `sklearn` (the `organism` extra) or a workspace sibling — the
  `entrain_infer` core, `axis_core` — and they say so by name rather than passing
  quietly. The trading path's own tests do run in the image:
  `docker compose exec trader pytest -q tests/test_trade_loop.py
  tests/test_physics_real_testnet.py tests/test_physics_aave_v3.py` is green. All 850
  pass on a host that holds the workspace.


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
| `TESTNET_PROTOCOLS_FILE` / `TESTNET_PROTOCOLS` | the operator's address book — the whitelist. An entry's `kind` says what it is: `vault` (default) or `aave-v3` | empty book (nothing stakeable) |
| `TESTNET_HISTORY` | where a record comes from: `chain` (the vault's own `YieldDistributed` logs), `feed` (DeFiLlama's daily APY series for the entry's `name` or pinned `pool` uuid — no node read) or `auto` (log first, feed when the log is silent) | `chain`; `.env.example` ships `auto` |
| `TESTNET_DECIMALS` | credits per token unit | `6` |
| `TESTNET_LOOKBACK` | blocks scanned for a record | `5000` |
| `TESTNET_RECEIPT_TIMEOUT` | seconds to await a receipt | `120` |
| `TESTNET_BLOCK_TIMEOUT` | seconds to await a block | `30` |
| `TESTNET_ALLOW_MAINNET` | `1` to let a real-money chain through (with `allow_mainnet=True` in code) | unset → refused |
| `TESTNET_ABI_FILE` | override the built-in vault ABI | built in |
| `DRY_RUN` | `0/false/no/off` makes the trader sign; anything else (including unset, and a typo) computes and logs only | unset → **dry run** |
| `SCAN_LIMIT` | pools the funnel reads each cycle (`--limit` overrides it) | `200` (= the feed's `POOLS_LIMIT`) |
| `MEASURE_LIMIT` | approved pools the hunter measures each cycle (`--hunt` overrides it) | `50` |
| `MAX_TRADE_SIZE_USD` | the most one transaction may commit | `0` → no USD ceiling of its own; `.env.example` ships `10.0` |
| `MAX_TRADE_FRACTION` | a share of the wallet, read from the chain on the same cycle | `0.20` |
| `ETH_USD_PRICE` | how the native balance is valued; without it the fraction applies to the stable credits only | unset → ETH unpriced |
| `GAS_RESERVE_ETH` | below this, a stake is refused instead of sent | `0.0005` |
| `TEST_STAKE_AMOUNT_USD` | the ceiling on one transaction while the operator is still deciding — "send one dollar first". Never lifted by a panic; on a top-up venue it caps the *step* towards the target | `0` → no micro cap; `.env.example` leaves it empty |
| `FORCE_TRADE` | `1` lets an `aave-v3` entry be supplied while the market has no verdict for it (`--force-trade` says the same). A test tool: no ceiling lifted, no conviction outranked, nothing signed in a dry run | unset → the market keeps its veto |
| `TRADE_INTERVAL` | seconds between trading cycles (compose) | `300` |
| `TRADER_STATE` | where the trader remembers its positions | `data/trader_state.json` |
| `TRADE_MARKET` | `real` (DeFiLlama), `synthetic` (drawn, no egress) or `off` | `real` |
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

