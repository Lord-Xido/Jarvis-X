"""Compatibility surface for the operational DM-vOmegaXi / Deep Distiller law.

The historical name ``dm_vo_xi_operational`` now resolves to the product-facing
Dr Moagi Deep Distiller implementation. New code should import directly from
``jarvisx.dr_moagi_deep_distiller``.
"""

from .dr_moagi_deep_distiller import (
    DeepDistiller,
    DeepDistillerCandidate,
    DeepDistillerConfig,
    DeepDistillerReport,
    DeepDistillerTheta,
)

__all__ = [
    "DeepDistiller",
    "DeepDistillerCandidate",
    "DeepDistillerConfig",
    "DeepDistillerReport",
    "DeepDistillerTheta",
]
