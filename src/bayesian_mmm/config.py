"""
Global configuration for the Bayesian MMM portfolio project.

Step 1 (synthetic data) uses the constants and ground-truth DGP parameters here.
Later steps (PyMC model, optimization, reporting) will extend this module.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility & calendar
# ---------------------------------------------------------------------------
RNG_SEED: int = 42
N_WEEKS: int = 260
START_DATE: str = "2020-01-06"

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
DAILY_MMM_CSV = DATA_SYNTHETIC_DIR / "daily_mmm.csv"
WEEKLY_MMM_CSV = DATA_SYNTHETIC_DIR / "weekly_mmm.csv"
SIMULATION_STATE_JSON = DATA_SYNTHETIC_DIR / "simulation_state.json"
TRUE_PARAMS_JSON = DATA_SYNTHETIC_DIR / "true_params.json"

# ---------------------------------------------------------------------------
# Bootstrap & refresh defaults
# ---------------------------------------------------------------------------
BOOTSTRAP_YEARS: int = 5
BOOTSTRAP_END_DATE: str | None = None  # None → yesterday at run time

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_WEEKLY_TABLE: str = "marketing_spend_weekly"
# If True, max(date) in Supabase is week-start Monday → last covered day = max + 6
DATE_IS_WEEK_START: bool = True

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
MEDIA_CHANNELS: list[str] = [
    "PPC_Brand",
    "PPC_Generic",
    "Display",
    "Social",
    "TV",
    "OOH",
    "Meta",
    "Yahoo",
]

SPEND_COLUMNS: list[str] = [f"{ch}_Spend" for ch in MEDIA_CHANNELS]

CONTROL_COLUMNS: list[str] = [
    "competitor_spend",
    "google_trend_competitor",
]

KPI_COLUMNS: list[str] = ["opportunities", "sales"]

DATE_COLUMN: str = "date"

# ---------------------------------------------------------------------------
# Ground-truth media parameters (DGP)
# Narrative:
#   PPC_Brand — saturated (high adstock, low half_sat vs typical spend scale)
#   Display   — under-saturated (high half_sat)
#   TV / OOH  — long carryover (high adstock alpha)
# ---------------------------------------------------------------------------
CHANNEL_ADSTOCK_ALPHA: dict[str, float] = {
    "PPC_Brand": 0.55,
    "PPC_Generic": 0.35,
    "Display": 0.25,
    "Social": 0.30,
    "TV": 0.75,
    "OOH": 0.70,
    "Meta": 0.28,
    "Yahoo": 0.20,
}

CHANNEL_HILL_SLOPE: dict[str, float] = {
    "PPC_Brand": 1.8,
    "PPC_Generic": 1.4,
    "Display": 1.2,
    "Social": 1.3,
    "TV": 1.5,
    "OOH": 1.4,
    "Meta": 1.3,
    "Yahoo": 1.1,
}

# half_sat is on scaled adstocked spend in [0, 1] after / max(adstocked)
CHANNEL_HILL_HALF_SAT: dict[str, float] = {
    "PPC_Brand": 0.35,  # low → saturates quickly
    "PPC_Generic": 0.55,
    "Display": 0.90,  # high → room to grow
    "Social": 0.60,
    "TV": 0.50,
    "OOH": 0.55,
    "Meta": 0.58,
    "Yahoo": 0.65,
}

CHANNEL_BETA: dict[str, float] = {
    "PPC_Brand": 420.0,
    "PPC_Generic": 280.0,
    "Display": 350.0,
    "Social": 220.0,
    "TV": 180.0,
    "OOH": 120.0,
    "Meta": 200.0,
    "Yahoo": 80.0,
}

# ---------------------------------------------------------------------------
# Baseline opportunities (Fourier seasonality + Q4 boost)
# ---------------------------------------------------------------------------
BASELINE_INTERCEPT: float = 800.0
BASELINE_FOURIER_ORDER: int = 2
BASELINE_FOURIER_COEFS: list[tuple[float, float]] = [
    (120.0, 0.0),  # cos/sin pair 1 (annual)
    (60.0, 40.0),  # cos/sin pair 2 (semi-annual)
]
Q4_BOOST: float = 180.0  # additive lift in Oct–Dec weeks

# ---------------------------------------------------------------------------
# Controls & funnel
# ---------------------------------------------------------------------------
CONTROL_GAMMA: dict[str, float] = {
    "competitor_spend": -0.0008,
    "google_trend_competitor": 15.0,
}

TRUE_CLOSE_RATE: float = 0.12

# ---------------------------------------------------------------------------
# Noise (opportunities & sales)
# ---------------------------------------------------------------------------
OPPORTUNITIES_NOISE_STD: float = 45.0
SALES_NOISE_STD: float = 8.0

# ---------------------------------------------------------------------------
# Spend & control simulation
# ---------------------------------------------------------------------------
SPEND_BASE_LEVEL: dict[str, float] = {
    "PPC_Brand": 18_000.0,
    "PPC_Generic": 12_000.0,
    "Display": 8_000.0,
    "Social": 6_000.0,
    "TV": 25_000.0,
    "OOH": 10_000.0,
    "Meta": 7_000.0,
    "Yahoo": 2_500.0,
}

SPEND_LOG_SIGMA: float = 0.12
SPEND_SEASONAL_AMPLITUDE: float = 0.15
SPEND_AR_PHI: float = 0.65
SPEND_PULSE_PROB: float = 0.04
SPEND_PULSE_SCALE: float = 1.8

COMPETITOR_SPEND_BASE: float = 45_000.0
COMPETITOR_SPEND_SIGMA: float = 0.10
GOOGLE_TREND_BASE: float = 50.0
GOOGLE_TREND_SIGMA: float = 8.0
GOOGLE_TREND_AR: float = 0.80

# ---------------------------------------------------------------------------
# Daily-scaled DGP helpers (weekly params → daily equivalents)
# ---------------------------------------------------------------------------
DAYS_PER_WEEK: float = 7.0
DAYS_PER_YEAR: float = 365.25


def daily_spend_base(channel: str) -> float:
    """Average daily spend level from weekly base."""
    return SPEND_BASE_LEVEL[channel] / DAYS_PER_WEEK


def daily_beta(channel: str) -> float:
    return CHANNEL_BETA[channel] / DAYS_PER_WEEK


def daily_adstock_alpha(channel: str) -> float:
    """Daily decay equivalent to weekly CHANNEL_ADSTOCK_ALPHA."""
    alpha_week = CHANNEL_ADSTOCK_ALPHA[channel]
    return alpha_week ** (1.0 / DAYS_PER_WEEK)


def daily_baseline_intercept() -> float:
    return BASELINE_INTERCEPT / DAYS_PER_WEEK


def daily_fourier_coefs() -> list[tuple[float, float]]:
    return [(c / DAYS_PER_WEEK, s / DAYS_PER_WEEK) for c, s in BASELINE_FOURIER_COEFS]


def daily_q4_boost() -> float:
    return Q4_BOOST / DAYS_PER_WEEK


def daily_opportunities_noise_std() -> float:
    return OPPORTUNITIES_NOISE_STD / math.sqrt(DAYS_PER_WEEK)


def daily_sales_noise_std() -> float:
    return SALES_NOISE_STD / math.sqrt(DAYS_PER_WEEK)


def daily_competitor_spend_base() -> float:
    return COMPETITOR_SPEND_BASE / DAYS_PER_WEEK


def daily_pulse_prob() -> float:
    """Per-day pulse probability preserving expected weekly rate."""
    return 1.0 - (1.0 - SPEND_PULSE_PROB) ** (1.0 / DAYS_PER_WEEK)
