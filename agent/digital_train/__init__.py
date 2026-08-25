"""Digital self-training (domain-agnostic, digital world layer).

- ``model``    — ``TimeSeriesModel``: trends + anomalies + cross-domain links
                 on any normalized numeric series (numpy-only);
- ``data``     — build a training matrix from collected feed observations;
- ``state``    — persistent state (history + model snapshot + metrics);
- ``pipeline`` — autonomous cycle: collect → train → update → signed
                 federated contribution (import via ``agent.digital_train.pipeline``).

Runs headlessly: ``python -m agent.digital_train.pipeline --loop``.
"""

from agent.digital_train.model import TimeSeriesModel
from agent.digital_train.state import DigitalState, load, save

__all__ = [
    "DigitalState",
    "TimeSeriesModel",
    "load",
    "save",
]
