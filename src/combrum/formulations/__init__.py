"""Row-generation formulations behind the generic solve-method contract.

Both methods are model-agnostic (feature map injected at construction) and
master-based (master built on the owner rank, default 0, and passed via the
fit context).
"""

from combrum.formulations.nslack import NSlack
from combrum.formulations.oneslack import (
    FeatureMap,
    OneSlack,
)

__all__ = [
    "FeatureMap",
    "NSlack",
    "OneSlack",
]
