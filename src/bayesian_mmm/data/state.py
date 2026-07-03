"""Persist simulation state across incremental daily refreshes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bayesian_mmm import config as cfg


@dataclass
class SimulationState:
    """Carryover state for AR(1) spends, adstock, and RNG between refreshes."""

    rng_seed: int = cfg.RNG_SEED
    day_index: int = 0
    last_daily_date: str | None = None
    spend_log_state: dict[str, float] = field(default_factory=dict)
    google_trend_last: float = cfg.GOOGLE_TREND_BASE
    adstock_state: dict[str, float] = field(default_factory=dict)
    adstock_scale_max: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fresh(cls, seed: int | None = None) -> SimulationState:
        """Initialize empty state for a new bootstrap run."""
        seed = cfg.RNG_SEED if seed is None else seed
        return cls(
            rng_seed=seed,
            day_index=0,
            last_daily_date=None,
            spend_log_state={},
            google_trend_last=cfg.GOOGLE_TREND_BASE,
            adstock_state={ch: 0.0 for ch in cfg.MEDIA_CHANNELS},
            adstock_scale_max={ch: 1e-12 for ch in cfg.MEDIA_CHANNELS},
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SimulationState:
        return cls(
            rng_seed=data.get("rng_seed", cfg.RNG_SEED),
            day_index=data.get("day_index", 0),
            last_daily_date=data.get("last_daily_date"),
            spend_log_state=data.get("spend_log_state", {}),
            google_trend_last=data.get("google_trend_last", cfg.GOOGLE_TREND_BASE),
            adstock_state=data.get("adstock_state", {}),
            adstock_scale_max=data.get("adstock_scale_max", {}),
        )


def load_state(path: Path | None = None) -> SimulationState | None:
    """Load state from JSON; return None if file does not exist."""
    path = cfg.SIMULATION_STATE_JSON if path is None else path
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return SimulationState.from_dict(json.load(f))


def save_state(state: SimulationState, path: Path | None = None) -> None:
    """Write simulation state to JSON."""
    path = cfg.SIMULATION_STATE_JSON if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
