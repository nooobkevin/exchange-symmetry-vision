"""Exchange-symmetry bilinear interaction library for vision experiments."""

from .adapter import (
    RMSNorm2d,
    ExchangeBilinearCore,
    GroupedWedgeCore,
    SpatialExchangeAdapter,
    VALID_MODES,
)
from .models import (
    ConvNeXtWithAdapter,
    load_convnext_atto_ema_weights,
    FrozenPairHead,
    ViTWithAdapter,
)

__all__ = [
    "RMSNorm2d",
    "ExchangeBilinearCore",
    "GroupedWedgeCore",
    "SpatialExchangeAdapter",
    "VALID_MODES",
    "ConvNeXtWithAdapter",
    "load_convnext_atto_ema_weights",
    "FrozenPairHead",
    "ViTWithAdapter",
]
