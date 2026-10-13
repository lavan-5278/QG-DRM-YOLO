"""Read-only diagnostics for the PVR-Q research gate."""

from .phase0_metrics import compute_competition, gate_g0, match_instances

__all__ = ("compute_competition", "gate_g0", "match_instances")
