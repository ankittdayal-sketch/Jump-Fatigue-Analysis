"""
Volleyball Jump Fatigue & Injury Prevention Simulator
Core logic module — DNN-inspired rule engine + NLP sentiment analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DailyInput:
    """All data captured for a single training day."""

    date: date

    # Physiological
    hrv_today: float          # ms
    hrv_baseline_30d: float   # ms — rolling 30-day average
    deep_sleep_min: float     # minutes of deep sleep last night
    water_liters: float       # total water intake (L)
    protein_g: float          # total protein intake (g)
    body_weight_kg: float     # used for protein/kg calculation

    # Training load
    jumps_today: int          # jump reps in today's session
    rpe: float                # rate of perceived exertion 1–10
    acute_load_7d: int        # total jumps over past 7 days (incl. today)
    chronic_load_weekly_avg: int  # average weekly jumps over past 28 days

    # Asymmetry / stiffness
    morning_stiffness_days: int   # consecutive days of morning stiffness
    one_sided_soreness_hrs: float # hours one side has been sore

    # Journal
    journal_entry: str = ""
    mood_quote: str = ""


@dataclass
class Flag:
    level: str          # "danger" | "warning" | "info"
    code: str           # machine-readable identifier
    title: str
    body: str


@dataclass
class SimulatorResult:
    date: date
    readiness_score: float      # 0–100
    injury_risk_pct: float      # 0–100
    zone: str                   # "green" | "yellow" | "red"
    zone_label: str
    zone_description: str
    flags: list[Flag]
    ac_ratio: float
    protein_per_kg: float
    hrv_drop_sd: float          # how many SDs today's HRV is below baseline
    sentiment_zone: str         # "green" | "yellow" | "red"
    sentiment_keywords: list[str]
    quote_of_the_day: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# NLP keyword lists
# ---------------------------------------------------------------------------

RED_KEYWORDS = {
    "dread", "dreading", "exhausted", "heavy", "hate", "quit", "done",
    "burnout", "broken", "aching", "clicking", "pop", "sharp", "hopeless",
    "can't go on", "giving up", "no energy", "empty",
}

YELLOW_KEYWORDS = {
    "tired", "sore", "stiff", "slow", "dull", "weak", "groggy", "sluggish",
    "achy", "fatigue", "fatigued", "dragging", "rough", "off", "meh",
}


def analyze_sentiment(text: str) -> tuple[str, list[str]]:
    """
    Basic keyword-based NLP sentiment analysis.

    Returns
    -------
    zone : "green" | "yellow" | "red"
    matched_keywords : list of flagged words found in text
    """
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    red_found = sorted(words & RED_KEYWORDS)
    yellow_found = sorted(words & YELLOW_KEYWORDS)

    if red_found:
        return "red", red_found
    if yellow_found:
        return "yellow", yellow_found
    return "green", []


# ---------------------------------------------------------------------------
# Injury flag engine
# ---------------------------------------------------------------------------

def compute_hrv_sd_drop(hrv_today: float, baseline: float) -> float:
    """
    Estimate how many standard deviations today's HRV is below baseline.
    We approximate 1 SD ≈ 12% of the baseline value (common population heuristic).
    """
    if baseline <= 0:
        return 0.0
    approx_sd = baseline * 0.12
    drop = baseline - hrv_today
    return drop / approx_sd if approx_sd else 0.0


def evaluate_flags(inp: DailyInput, sentiment_zone: str) -> list[Flag]:
    flags: list[Flag] = []

    # 1. Acute:Chronic workload ratio (>1.5x = overreach)
    ac_ratio = (inp.acute_load_7d / inp.chronic_load_weekly_avg
                if inp.chronic_load_weekly_avg > 0 else 0.0)
    if ac_ratio > 1.5:
        flags.append(Flag(
            level="danger",
            code="OVERREACH",
            title="High strain warning — acute overreach",
            body=(
                f"Your 7-day load is {ac_ratio:.2f}× your chronic baseline "
                f"(threshold: 1.5×). Soft-tissue injury risk is significantly "
                f"elevated. Reduce jump volume by 30–40% for 2–3 days."
            ),
        ))
    elif ac_ratio > 1.2:
        flags.append(Flag(
            level="warning",
            code="OVERREACH_MILD",
            title="Elevated workload — borderline overreach",
            body=(
                f"A:C ratio of {ac_ratio:.2f}× is approaching the 1.5× danger "
                f"threshold. Monitor load carefully and avoid adding extra sessions."
            ),
        ))

    # 2. CNS fatigue — HRV drop ≥ 2 SD
    hrv_drop = compute_hrv_sd_drop(inp.hrv_today, inp.hrv_baseline_30d)
    if hrv_drop >= 2.0:
        flags.append(Flag(
            level="danger",
            code="CNS_FATIGUE",
            title="CNS fatigue — HRV critically low",
            body=(
                f"Today's HRV ({inp.hrv_today:.0f} ms) has dropped "
                f"{hrv_drop:.1f} SD below your 30-day baseline "
                f"({inp.hrv_baseline_30d:.0f} ms). Central nervous system "
                f"recovery is compromised. Prioritise sleep and reduce intensity."
            ),
        ))

    # 3. Tendon desiccation risk
    if inp.morning_stiffness_days >= 3 and inp.water_liters < 3.0:
        flags.append(Flag(
            level="danger",
            code="TENDON_DESICCATION",
            title="Tendon desiccation risk",
            body=(
                f"Morning stiffness reported for {inp.morning_stiffness_days} "
                f"consecutive days with low hydration ({inp.water_liters:.1f} L). "
                f"Inadequate fluid intake reduces tendon elasticity. "
                f"Increase water intake to 3 L+ immediately."
            ),
        ))

    # 4. Biomechanical compensation
    if inp.one_sided_soreness_hrs > 48:
        flags.append(Flag(
            level="warning",
            code="BIOMECH_COMPENSATION",
            title="Biomechanical compensation risk",
            body=(
                f"One-sided soreness persisting {inp.one_sided_soreness_hrs:.0f} hrs "
                f"suggests compensation loading on the contralateral limb, "
                f"which significantly increases injury risk. Consider a movement "
                f"screen or physiotherapy assessment."
            ),
        ))

    # 5. Deep sleep — growth hormone repair flag
    if inp.deep_sleep_min < 90:
        flags.append(Flag(
            level="warning",
            code="LOW_DEEP_SLEEP",
            title="Reduced repair hormone flag — low deep sleep",
            body=(
                f"Deep sleep recorded at {inp.deep_sleep_min:.0f} min "
                f"(threshold: 90 min). Growth hormone secretion — critical for "
                f"connective tissue repair — occurs primarily during deep sleep."
            ),
        ))

    # 6. Protein insufficiency during high-load phase
    protein_per_kg = inp.protein_g / inp.body_weight_kg if inp.body_weight_kg > 0 else 0
    if protein_per_kg < 1.6 and inp.rpe >= 7:
        flags.append(Flag(
            level="warning",
            code="PROTEIN_DEFICIT",
            title="Mechanical risk — insufficient protein for load",
            body=(
                f"Protein at {protein_per_kg:.2f} g/kg body weight "
                f"(minimum: 1.6 g/kg during high-intensity phases). "
                f"Muscle and tendon repair is compromised at current intake."
            ),
        ))

    # 7. Sentiment / journal flags
    if sentiment_zone == "red":
        flags.append(Flag(
            level="danger",
            code="BURNOUT_SENTIMENT",
            title="Burnout sentiment detected in journal",
            body=(
                "Keywords associated with mental burnout or physical distress "
                "were detected in your journal. Mandatory rest is recommended. "
                "Mental fatigue reliably precedes physical injury."
            ),
        ))
    elif sentiment_zone == "yellow":
        flags.append(Flag(
            level="warning",
            code="FATIGUE_SENTIMENT",
            title="Fatigue language in journal entry",
            body=(
                "Your journal entry contains words associated with physical "
                "fatigue. Consider a reduced-intensity session and prioritise "
                "nutrition and hydration."
            ),
        ))

    return flags


# ---------------------------------------------------------------------------
# Readiness score & risk calculation
# ---------------------------------------------------------------------------

def compute_scores(inp: DailyInput, flags: list[Flag]) -> tuple[float, float]:
    """
    Returns (readiness_score 0-100, injury_risk_pct 0-100).
    Risk drives score: readiness ≈ 100 - risk, adjusted by positive factors.
    """
    risk = 5.0  # baseline noise

    flag_weights = {
        "OVERREACH": 25,
        "OVERREACH_MILD": 12,
        "CNS_FATIGUE": 15,
        "TENDON_DESICCATION": 15,
        "BIOMECH_COMPENSATION": 12,
        "LOW_DEEP_SLEEP": 10,
        "PROTEIN_DEFICIT": 8,
        "BURNOUT_SENTIMENT": 15,
        "FATIGUE_SENTIMENT": 7,
    }

    for flag in flags:
        risk += flag_weights.get(flag.code, 5)

    risk = min(risk, 99.0)

    # Positive modifiers
    score = 100.0 - risk
    if inp.deep_sleep_min > 110:
        score = min(100, score + 5)
    if inp.water_liters >= 3.0:
        score = min(100, score + 3)
    if inp.hrv_today > inp.hrv_baseline_30d:
        score = min(100, score + 4)

    score = max(0.0, min(100.0, round(score, 1)))
    risk = round(risk, 1)
    return score, risk


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def build_recommendations(flags: list[Flag], score: float) -> list[str]:
    recs = []
    codes = {f.code for f in flags}

    if "OVERREACH" in codes:
        recs.append("Reduce jump volume by 30–40% for the next 2–3 days.")
    if "CNS_FATIGUE" in codes:
        recs.append("Replace high-intensity training with technical or low-load work.")
    if "TENDON_DESICCATION" in codes:
        recs.append("Increase water intake to at least 3 L today and tomorrow.")
    if "BIOMECH_COMPENSATION" in codes:
        recs.append("Book a physiotherapy or movement screen within 48 hours.")
    if "LOW_DEEP_SLEEP" in codes:
        recs.append("Target 7–9 hours total sleep; avoid screens 1 hour before bed.")
    if "PROTEIN_DEFICIT" in codes:
        recs.append("Add a protein source to your next two meals (target ≥1.6 g/kg).")
    if "BURNOUT_SENTIMENT" in codes:
        recs.append("Take a full rest day. Mental recovery is a performance priority.")
    if not recs and score >= 75:
        recs.append("All systems go — train at full intensity. Stay dialled in.")
    return recs


# ---------------------------------------------------------------------------
# Quote of the day
# ---------------------------------------------------------------------------

QUOTES = {
    "green": (
        '"The only way to prove you are a good sport is to lose." '
        "— Competitive spirit is high. Harness it."
    ),
    "yellow": (
        '"The spirit is willing, but the flesh is weak." '
        "— Nutritional and recovery intervention recommended."
    ),
    "red": (
        '"I am tired of being tired." '
        "— Your mind is signalling the body needs a full rest day. "
        "Trust the process; rest is training."
    ),
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_simulation(inp: DailyInput) -> SimulatorResult:
    """Run the full simulation pipeline and return a result object."""

    sentiment_zone, sentiment_keywords = analyze_sentiment(
        inp.journal_entry + " " + inp.mood_quote
    )

    flags = evaluate_flags(inp, sentiment_zone)
    score, risk = compute_scores(inp, flags)
    recs = build_recommendations(flags, score)

    ac_ratio = (inp.acute_load_7d / inp.chronic_load_weekly_avg
                if inp.chronic_load_weekly_avg > 0 else 0.0)
    protein_per_kg = inp.protein_g / inp.body_weight_kg if inp.body_weight_kg > 0 else 0.0
    hrv_drop_sd = compute_hrv_sd_drop(inp.hrv_today, inp.hrv_baseline_30d)

    # Zone
    if score >= 75:
        zone, zone_label, zone_desc = (
            "green",
            "Green zone — ready to train",
            "Recovery markers are strong. Full intensity training is well-supported.",
        )
    elif score >= 50:
        zone, zone_label, zone_desc = (
            "yellow",
            "Yellow zone — proceed with caution",
            "Some recovery metrics are suboptimal. Reduce peak intensity by 20–30%.",
        )
    else:
        zone, zone_label, zone_desc = (
            "red",
            "Red zone — rest day recommended",
            "Multiple injury risk flags are active. Active recovery or full rest is strongly advised.",
        )

    # Override quote if risk > 20%
    if risk > 20:
        quote = QUOTES.get(sentiment_zone if sentiment_zone != "green" else zone, QUOTES["yellow"])
    else:
        quote = QUOTES["green"]

    return SimulatorResult(
        date=inp.date,
        readiness_score=score,
        injury_risk_pct=risk,
        zone=zone,
        zone_label=zone_label,
        zone_description=zone_desc,
        flags=flags,
        ac_ratio=round(ac_ratio, 2),
        protein_per_kg=round(protein_per_kg, 2),
        hrv_drop_sd=round(hrv_drop_sd, 2),
        sentiment_zone=sentiment_zone,
        sentiment_keywords=sentiment_keywords,
        quote_of_the_day=quote,
        recommendations=recs,
    )
