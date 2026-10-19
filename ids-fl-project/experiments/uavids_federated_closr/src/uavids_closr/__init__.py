"""CPU-compatible centralized and federated CLOSR research baseline."""

from .model import CLOSRModel
from .loss import CLOSRLoss

__all__ = ["CLOSRLoss", "CLOSRModel"]
