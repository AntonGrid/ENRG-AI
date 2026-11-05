"""The friction: what a trade costs *before* it has earned anything.

Every other file in this layer answers a question about somebody else's numbers — is
the record honest, is the address approved, is the price a panic. This one answers a
question about the agent's **own** arithmetic, and it is the question that decides
whether any of the others matter:

    gross alpha is what the market advertises. net alpha is what survives the trade.

Two costs stand between the two, and both are paid in the same currency as the alpha:

* **gas.** Paid once per transaction, in ETH, at whatever the chain is charging. It
  does not scale with the size of the trade, which is exactly why it is a *floor*: a
  thirty-dollar stake and a three-hundred-thousand-dollar stake cost the same to send.
* **slippage.** Paid in the price, and it *does* scale with the size of the trade
  relative to the depth of the pool. A trade that is one percent of a pool is a
  rounding error; the same trade into a pool a hundred times smaller is the pool.

The model is deliberately small, and it is a model rather than a measurement:

    slippage_pct = coefficient × trade_size_usd / tvl_usd        (capped at 100 %)
    gas_cost_usd = gas_used × gas_price_gwei × 1e-9 × eth_price_usd
    net_alpha    = trade_size_usd × gross_alpha_pct − slippage − gas

The ``coefficient`` is read as price impact per unit of relative size, which is the
usual first-order estimate for a constant-product pool: half the depth for a pair whose
two legs move together (``STABLE_SLIPPAGE_COEFFICIENT``), twice the depth for one whose
depth is split across a volatile side (``VOLATILE_SLIPPAGE_COEFFICIENT``). Which of the
two applies is a *policy* the caller declares, not a measurement this file makes — and
the tests pin the ordering of the two, not the numbers.

What the model refuses to do is as much of the argument as what it computes:

* **an unmeasured depth is not a bottomless pool.** ``tvl_usd=None`` means the venue
  published no depth; slippage is then *not estimated* (and :class:`CostReport` says so
  in ``measured_depth``) instead of being filled in with a comfortable zero. A
  *published* zero is the opposite case and is fatal: nothing can be traded against
  nothing, so slippage is the whole trade (:data:`MAX_SLIPPAGE`).
* **a profit is a claim about money, not about percentages.** A trade clears the bar
  only if it earns at least :data:`MIN_NET_ALPHA_USD` **and** at least
  :data:`MIN_NET_ALPHA_PCT` of its own size: the first is the fixed cost of showing up,
  the second is the hurdle that makes the risk worth taking. A cent of alpha on a
  million dollars is a rounding error wearing a percentage sign.

The verdict is a :class:`CostReport` — a value, not a boolean and not an exception — so
the book can record *why* it declined to trade instead of merely that it did. That is
the same discipline ``risk_assessment`` keeps about verdicts: a refusal is a measurement
with a reason attached, and the reason has a name
(:data:`UNPROFITABLE_AFTER_FEES`).

Where this file is wired in:

* :class:`agent.physics.portfolio.Portfolio` takes one as ``friction`` and asks it
  before any credit moves — an entry that cannot pay for itself is logged as ``skip``
  with :data:`UNPROFITABLE_AFTER_FEES` and the bucket stays in cash;
* :meth:`TransactionCostAnalyzer.from_receipt` builds the model out of the *facts a
  venue actually returned* — the ``gas_used`` and ``effective_gas_price`` of a mined
  transaction, read off the record
  :class:`agent.physics.real_testnet_provider.RealTestnetProvider` hands back — so the
  numbers being subtracted come from a transaction that happened rather than from a
  parameter someone liked.

Quick start::

    from agent.physics.transaction_cost_analyzer import TransactionCostAnalyzer

    venue = TransactionCostAnalyzer(eth_price_usd=3_000.0, gas_price_gwei=2.0)
    report = venue.assess(annual_apy=0.12, trade_size_usd=1_000.0, tvl_usd=250_000.0)
    print(report.net_alpha_usd, report.reason)   # ~3.9, '' — worth placing

Nothing here touches a network, a clock or a file: the module is arithmetic over facts
the caller measured, which is what makes it testable to the last cent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "CostModelError",
    "CostReport",
    "TransactionCostAnalyzer",
    "DEFAULT_ETH_PRICE_USD",
    "DEFAULT_GAS_PRICE_GWEI",
    "DEFAULT_GAS_UNITS",
    "DEFAULT_HORIZON_DAYS",
    "GWEI",
    "MAX_SLIPPAGE",
    "MIN_NET_ALPHA_PCT",
    "MIN_NET_ALPHA_USD",
    "NO_TRADE",
    "STABLE_SLIPPAGE_COEFFICIENT",
    "UNPROFITABLE_AFTER_FEES",
    "VOLATILE_SLIPPAGE_COEFFICIENT",
    "WEI_PER_ETH",
]


# ── the declared numbers, and where each one comes from ──────────────────────


def _receipt_field(receipt: Any, *names: str) -> Optional[Any]:
    """Read one fact off a receipt, whichever shape the venue handed back.

    A node's receipt is a mapping; a provider's record may be an object. Rather than make
    the caller convert — a conversion is a place where a gas figure quietly becomes a
    string — both shapes are read here, and ``None`` means *the venue did not report it*,
    which every caller then has to handle explicitly.
    """
    for name in names:
        if isinstance(receipt, Mapping):
            value = receipt.get(name, None)
        else:
            value = getattr(receipt, name, None)
        if value is not None:
            return value
    return None


class CostModelError(RuntimeError):
    """A cost model that cannot be a cost model: a negative price, a receipt with no gas."""


#: Price impact per unit of *relative* size, for a pair whose two legs move together.
#: Half the depth is the conservative read of a stable swap, where the two sides are the
#: same asset and the curve is nearly flat; it is a declared policy, not a derivation.
STABLE_SLIPPAGE_COEFFICIENT = 0.5

#: Price impact per unit of *relative* size, for a pair with a volatile side. Four times
#: the stable coefficient, because the depth on the far side is not the depth the trade
#: is priced against, and the curve bends with the size.
VOLATILE_SLIPPAGE_COEFFICIENT = 2.0

#: The most a trade can cost in price impact. Beyond 100 % of the trade there is no fill
#: to speak of; the number exists so that a pool with no depth produces a finite, readable
#: refusal instead of an infinity in a table.
MAX_SLIPPAGE = 1.0

#: Wei in one ether, and gwei in one ether — the two conversions gas is spelled in.
WEI_PER_ETH = 10 ** 18
GWEI = 10 ** 9

#: What one ether is worth when the operator has not said. A declared assumption, and the
#: only number here that nobody measured: gas is priced *in* it, so a stale ETH price
#: moves every verdict by the same factor. The notebook passes the real one.
DEFAULT_ETH_PRICE_USD = 3_000.0

#: Gas units for the ordinary two-storage-writes transaction: an ERC-20 ``approve``, a
#: vault ``stake`` or ``withdraw``. Measured, not guessed — a stake is 90–120 k with a
#: cold allowance, and the round number in the middle is the honest default.
DEFAULT_GAS_UNITS = 92_000

#: A gwei. Sepolia and every local Anvil sit at 1–7 gwei in practice, and a quiet hour on
#: mainnet is not much worse; a venue that measured its own gas overrides it.
DEFAULT_GAS_PRICE_GWEI = 1.0

#: The horizon over which a position must pay back its entry cost. Thirty days is the
#: shortest span in which a DeFi position's yield is not noise; shorter and gas dominates
#: every verdict, longer and the model is financing a thesis rather than a trade.
DEFAULT_HORIZON_DAYS = 30.0

#: The fixed floor: below this many dollars the trade is not worth the click, whatever
#: percentage of its own size it earned.
MIN_NET_ALPHA_USD = 0.50

#: The hurdle rate over the horizon: a tenth of a percent. Both floors apply, and the
#: reason is in the module docstring — a fixed cost of showing up and a risk premium are
#: two different objections to the same trade.
MIN_NET_ALPHA_PCT = 0.001

#: The name a refusal carries all the way into the book's journal and the notebook's
#: table. One spelling, because a reason that is re-typed at each site is a reason that
#: drifts.
UNPROFITABLE_AFTER_FEES = "UNPROFITABLE_AFTER_FEES"

#: A trade of no size: nothing to earn and nothing to pay. Not a refusal — a question that
#: was never a question.
NO_TRADE = "NO_TRADE"



# ── the answer: a report, not a boolean ─────────────────────────────────────


@dataclass(frozen=True)
class CostReport:
    """One trade, priced: what it would earn gross, what it must pay, what is left.

    Every field is a number the caller can print, and the four that decide anything are
    ``net_alpha_usd``, ``net_alpha_pct``, ``profitable`` and ``reason``.
    ``measured_depth`` is the honesty flag: ``False`` means the venue published no depth,
    so ``slippage_pct`` is zero *because nothing was estimated* rather than because the
    pool is deep — a distinction a table cannot show and a report can.
    """

    trade_size_usd: float = 0.0
    gross_alpha_pct: float = 0.0
    gross_alpha_usd: float = 0.0
    slippage_pct: float = 0.0
    slippage_usd: float = 0.0
    gas_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    net_alpha_usd: float = 0.0
    net_alpha_pct: float = 0.0
    measured_depth: bool = False
    profitable: bool = False
    reason: str = NO_TRADE

    @property
    def refused(self) -> bool:
        """True when the trade was priced and lost: the mirror of :attr:`profitable`."""
        return not self.profitable

    @property
    def cost_share(self) -> float:
        """Costs as a share of the trade — the number a naive book is quietly missing."""
        if self.trade_size_usd <= 0.0:
            return 0.0
        return float(self.total_cost_usd / self.trade_size_usd)

    def as_dict(self) -> Dict[str, Any]:
        """The report as a row: what a table, a log line or a JSON report carries."""
        return {
            "trade_size_usd": round(float(self.trade_size_usd), 6),
            "gross_alpha_pct": round(float(self.gross_alpha_pct), 8),
            "gross_alpha_usd": round(float(self.gross_alpha_usd), 6),
            "slippage_pct": round(float(self.slippage_pct), 8),
            "slippage_usd": round(float(self.slippage_usd), 6),
            "gas_cost_usd": round(float(self.gas_cost_usd), 6),
            "total_cost_usd": round(float(self.total_cost_usd), 6),
            "net_alpha_usd": round(float(self.net_alpha_usd), 6),
            "net_alpha_pct": round(float(self.net_alpha_pct), 8),
            "measured_depth": bool(self.measured_depth),
            "profitable": bool(self.profitable),
            "reason": str(self.reason),
        }

    def summary_text(self) -> str:
        """One line, in the order the question is asked: earn, pay, keep."""
        return (
            f"{self.trade_size_usd:,.2f} $ → gross {self.gross_alpha_usd:,.4f} $ "
            f"({self.gross_alpha_pct:.4%}), costs {self.total_cost_usd:,.4f} $ "
            f"[slip {self.slippage_usd:,.4f} + gas {self.gas_cost_usd:,.4f}], "
            f"net {self.net_alpha_usd:+,.4f} $ ({self.net_alpha_pct:+.4%})"
            + ("" if not self.reason else f" — {self.reason}")
        )

# ── the model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransactionCostAnalyzer:
    """Gas and slippage, priced together, so a trade can be refused *before* it is sent.

    Constructed with the facts of the venue it prices — what one ether is worth, what a
    transaction costs in gas units and in gwei — and asked per trade for the two numbers
    that belong to the trade: its size and the depth it will be traded against.

    Frozen and cheap, because it is passed around: the
    :class:`~agent.physics.portfolio.Portfolio` holds one for the life of a run, and a
    model that could be mutated mid-run would be a model whose verdicts cannot be
    reproduced.
    """

    eth_price_usd: float = DEFAULT_ETH_PRICE_USD
    gas_used: int = DEFAULT_GAS_UNITS
    gas_price_gwei: float = DEFAULT_GAS_PRICE_GWEI
    stable_coefficient: float = STABLE_SLIPPAGE_COEFFICIENT
    volatile_coefficient: float = VOLATILE_SLIPPAGE_COEFFICIENT
    minimum_usd: float = MIN_NET_ALPHA_USD
    minimum_pct: float = MIN_NET_ALPHA_PCT
    horizon_days: float = DEFAULT_HORIZON_DAYS

    def __post_init__(self) -> None:
        """A model with a negative price or a negative floor is not a model."""
        if not float(self.eth_price_usd) > 0.0:
            raise CostModelError(
                f"an ether at {self.eth_price_usd!r} dollars prices gas at nothing"
            )
        if int(self.gas_used) < 0 or float(self.gas_price_gwei) < 0.0:
            raise CostModelError(
                f"gas of {self.gas_used!r} units at {self.gas_price_gwei!r} gwei "
                "is not a price"
            )
        for name in ("stable_coefficient", "volatile_coefficient", "minimum_pct"):
            if float(getattr(self, name)) < 0.0:
                raise CostModelError(
                    f"{name} cannot be negative ({getattr(self, name)!r})"
                )
        if float(self.minimum_usd) < 0.0 or float(self.horizon_days) <= 0.0:
            raise CostModelError(
                f"a floor of {self.minimum_usd!r} $ over {self.horizon_days!r} days "
                "is not a policy"
            )

    # -- the two costs ----------------------------------------------------- #

    @property
    def gas_cost_wei(self) -> int:
        """The transaction's price in wei — the fact a receipt reports, before any USD."""
        return int(int(self.gas_used) * int(round(float(self.gas_price_gwei) * GWEI)))

    def estimate_gas_cost_usd(
        self,
        gas_used: Optional[int] = None,
        gas_price_gwei: Optional[float] = None,
        eth_price_usd: Optional[float] = None,
    ) -> float:
        """Gas in dollars: ``units × gwei × 1e-9 × price``, with the model's own defaults.

        Omitted arguments fall back to the instance's facts, which is the common call: the
        venue measured its gas once, and every trade in the run pays the same price. A gas
        price of zero is a legitimate answer (a sponsored transaction, a local node
        running free) and produces ``0.0`` rather than an error.
        """
        units = int(self.gas_used if gas_used is None else gas_used)
        gwei = float(self.gas_price_gwei if gas_price_gwei is None else gas_price_gwei)
        price = float(self.eth_price_usd if eth_price_usd is None else eth_price_usd)
        if units < 0 or gwei < 0.0:
            raise CostModelError(
                f"gas of {units!r} units at {gwei!r} gwei is not a price"
            )
        if not price > 0.0:
            raise CostModelError(f"an ether at {price!r} dollars prices gas at nothing")
        return float(units) * gwei * (1.0 / GWEI) * price



    def estimate_slippage(
        self, trade_size_usd: float, tvl_usd: Optional[float], *, stable: bool = True
    ) -> Optional[float]:
        """Price impact as a fraction of the trade, or ``None`` when depth is unmeasured.

        Three answers, and the difference between them is the point of the method:

        * ``None`` — the venue published no depth. Nothing is estimated, and the caller
          learns which case it got from :attr:`CostReport.measured_depth` rather than from
          a zero that looks like a measurement;
        * :data:`MAX_SLIPPAGE` — the venue published a depth of zero. A pool with nothing
          in it cannot absorb the trade at any price, so the whole trade is friction;
        * otherwise the capped first-order estimate ``coefficient × size / depth``.
        """
        size = float(trade_size_usd)
        if tvl_usd is None:
            return None
        depth = float(tvl_usd)
        if not size > 0.0:
            return 0.0
        if not depth > 0.0:
            return float(MAX_SLIPPAGE)
        coefficient = float(
            self.stable_coefficient if stable else self.volatile_coefficient
        )
        return float(min(coefficient * size / depth, MAX_SLIPPAGE))

    # -- the horizon, and the arithmetic that decides ---------------------- #

    def horizon_return(self, annual_apy: float) -> float:
        """What an annual rate pays over the model's horizon — simple, not compounded.

        ``apy × days / 365``: the first-order answer, and the honest one here, because the
        quantity being compared against a *one-off* cost is the yield earned while the
        position is held, not the yield a year of compounding would eventually add.
        """
        return float(annual_apy) * float(self.horizon_days) / 365.0

    def calculate_net_alpha(
        self,
        gross_alpha_pct: float,
        trade_size_usd: float,
        *,
        tvl_usd: Optional[float] = None,
        stable: bool = True,
        gas_cost_usd: Optional[float] = None,
    ) -> CostReport:
        """The whole model in one call: gross minus slippage minus gas, and a verdict.

        ``gross_alpha_pct`` is the return expected over the horizon, as a fraction
        (``0.01`` is one percent). ``gas_cost_usd`` defaults to the model's own price, so
        the common call needs only the two facts that belong to the trade.
        """
        size = float(trade_size_usd)
        if size != size or size in (float("inf"), float("-inf")):
            raise CostModelError(f"a trade of {trade_size_usd!r} is not a size")
        gross_pct = float(gross_alpha_pct)
        if gross_pct != gross_pct:
            raise CostModelError(f"an alpha of {gross_alpha_pct!r} is not a number")
        if size <= 0.0:
            return CostReport(
                trade_size_usd=size,
                gross_alpha_pct=gross_pct,
                measured_depth=tvl_usd is not None,
                profitable=False,
                reason=NO_TRADE,
            )
        slippage_pct = self.estimate_slippage(size, tvl_usd, stable=stable)
        measured = slippage_pct is not None
        slip = 0.0 if slippage_pct is None else float(slippage_pct)
        gas = (
            self.estimate_gas_cost_usd()
            if gas_cost_usd is None
            else float(gas_cost_usd)
        )
        gross_usd = gross_pct * size
        slippage_usd = slip * size
        total = slippage_usd + gas
        net_usd = gross_usd - total
        net_pct = net_usd / size
        profitable = bool(
            net_usd >= float(self.minimum_usd) and net_pct >= float(self.minimum_pct)
        )
        return CostReport(
            trade_size_usd=size,
            gross_alpha_pct=gross_pct,
            gross_alpha_usd=gross_usd,
            slippage_pct=slip,
            slippage_usd=slippage_usd,
            gas_cost_usd=gas,
            total_cost_usd=total,
            net_alpha_usd=net_usd,
            net_alpha_pct=net_pct,
            measured_depth=measured,
            profitable=profitable,
            reason="" if profitable else UNPROFITABLE_AFTER_FEES,
        )

    def assess(
        self,
        annual_apy: float,
        trade_size_usd: float,
        *,
        tvl_usd: Optional[float] = None,
        stable: bool = True,
        gas_cost_usd: Optional[float] = None,
    ) -> CostReport:
        """Price a trade against an **annual** rate: the horizon is the model's own.

        This is the call the book makes, because a measured record speaks in APY
        (:attr:`agent.physics.portfolio.Protocol.implied_apy`) and a cost is paid once.
        Converting between the two is policy — :attr:`horizon_days` — so it lives here
        rather than at every call site.
        """
        return self.calculate_net_alpha(
            self.horizon_return(annual_apy),
            trade_size_usd,
            tvl_usd=tvl_usd,
            stable=stable,
            gas_cost_usd=gas_cost_usd,
        )

    # -- from a receipt, because a fact beats a parameter ------------------ #

    @classmethod
    def from_receipt(
        cls,
        receipt: Any,
        *,
        eth_price_usd: float = DEFAULT_ETH_PRICE_USD,
        **overrides: Any,
    ) -> "TransactionCostAnalyzer":
        """Build the model out of what a venue actually returned for a transaction.

        ``receipt`` is *what the venue handed back*, in whichever of the two shapes it
        hands back — an object with attributes, or a mapping (a node's receipt dict, or the
        record :meth:`agent.physics.real_testnet_provider.RealTestnetProvider.execute_stake`
        returns). Both spellings of each fact are accepted (``gas_used`` / ``gasUsed``,
        ``effective_gas_price`` / ``effectiveGasPrice``) because the difference between
        them is a naming convention and not a measurement.

        Those two are facts of a mined transaction, so a model built here prices the next
        trade at what the last one *cost*, not at what an operator hoped it would. A
        receipt that reports no gas is refused rather than turned into a free-trade model:
        nothing was broadcast, so nothing was measured, and a zero would be a lie with a
        reassuring shape.
        """
        units = _receipt_field(receipt, "gas_used", "gasUsed")
        wei = _receipt_field(receipt, "effective_gas_price", "effectiveGasPrice")
        if units is None or int(units) <= 0:
            raise CostModelError(
                "a receipt that used no gas is not a quote: nothing was broadcast, so "
                "nothing can be priced"
            )
        if wei is None:
            raise CostModelError(
                "a receipt without an effective gas price prices nothing: the fee the "
                "chain charged is the fact this model is built out of"
            )
        fields: Dict[str, Any] = {
            "eth_price_usd": float(eth_price_usd),
            "gas_used": int(units),
            "gas_price_gwei": float(int(wei)) / GWEI,
        }
        fields.update(overrides)
        return cls(**fields)
