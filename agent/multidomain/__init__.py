"""Multi-domain learning — shared backbone + per-domain heads.

Implements GLOBAL_AI_ARCHITECTURE §9.3: one backbone learns world-level
regularities across all domains; each domain has an independent head, so a
new domain starts from a pretrained backbone (transfer, few-shot) and a
broken feed does not corrupt the others (failure isolation).

- ``model`` — ``MultiDomainModel`` (numpy SGD, tanh hidden layer).
"""

from agent.multidomain.model import MultiDomainModel

__all__ = ["MultiDomainModel"]
