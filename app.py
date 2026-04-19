"""
Volleyball Injury Prevention Simulator — FastAPI app
Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations
from datetime import date
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from simulator import DailyInput, Flag, SimulatorResult, run_simulation


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    date: date = Field(default_factory=date.today)

    hrv_today: float = Field(..., ge=0, le=250)
    hrv_baseline_30d: float = Field(..., ge=0, le=250)
    deep_sleep_min: float = Field(..., ge=0, le=300)
    water_liters: float = Field(..., ge=0, le=10)
    protein_g: float = Field(..., ge=0, le=500)
    body_weight_kg: float = Field(..., ge=30, le=200)

    jumps_today: int = Field(..., ge=0, le=1000)
    rpe: float = Field(..., ge=1, le=10)
    acute_load_7d: int = Field(..., ge=0)
    chronic_load_weekly_avg: int = Field(..., ge=0)

    morning_stiffness_days: int = Field(..., ge=0, le=30)
    one_sided_soreness_hrs: float = Field(..., ge=0, le=200)

    journal_entry: str = Field(default="")
    mood_quote: str = Field(default="")

    model_config = {
        "json_schema_extra": {
            "example": {
                "hrv_today": 58,
                "hrv_baseline_30d": 72,
                "deep_sleep_min": 85,
                "water_liters": 2.2,
                "protein_g": 110,
                "body_weight_kg": 75,
                "jumps_today": 140,
                "rpe": 8,
                "acute_load_7d": 820,
                "chronic_load_weekly_avg": 500,
                "morning_stiffness_days": 2,
                "one_sided_soreness_hrs": 36,
                "journal_entry": "Feeling a bit sluggish today, knees are achy.",
            }
        }
    }


class FlagOut(BaseModel):
    level: str
    code: str
    title: str
    body: str


class SimulateResponse(BaseModel):
    date: date
    readiness_score: float
    injury_risk_pct: float
    zone: str
    zone_label: str
    zone_description: str
    flags: List[FlagOut]
    ac_ratio: float
    protein_per_kg: float
    hrv_drop_sd: float
    sentiment_zone: str
    sentiment_keywords: List[str]
    quote_of_the_day: str
    recommendations: List[str]


def result_to_response(r: SimulatorResult) -> SimulateResponse:
    return SimulateResponse(
        date=r.date,
        readiness_score=r.readiness_score,
        injury_risk_pct=r.injury_risk_pct,
        zone=r.zone,
        zone_label=r.zone_label,
        zone_description=r.zone_description,
        flags=[FlagOut(level=f.level, code=f.code, title=f.title, body=f.body)
               for f in r.flags],
        ac_ratio=r.ac_ratio,
        protein_per_kg=r.protein_per_kg,
        hrv_drop_sd=r.hrv_drop_sd,
        sentiment_zone=r.sentiment_zone,
        sentiment_keywords=r.sentiment_keywords,
        quote_of_the_day=r.quote_of_the_day,
        recommendations=r.recommendations,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Volleyball Injury Prevention Simulator",
    description=(
        "DNN-inspired daily readiness and injury risk engine for volleyball athletes. "
        "Submit daily physiological, training-load, and journal data to receive a "
        "readiness score, injury risk %, active flags, and personalised recommendations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "volleyball-injury-prevention-simulator"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/simulate", response_model=SimulateResponse, tags=["simulation"])
def simulate(body: SimulateRequest):
    """
    Run the injury-prevention simulation for a single training day.
    Returns a readiness score (0-100), injury risk %, active flags,
    and personalised recommendations.
    """
    inp = DailyInput(
        date=body.date,
        hrv_today=body.hrv_today,
        hrv_baseline_30d=body.hrv_baseline_30d,
        deep_sleep_min=body.deep_sleep_min,
        water_liters=body.water_liters,
        protein_g=body.protein_g,
        body_weight_kg=body.body_weight_kg,
        jumps_today=body.jumps_today,
        rpe=body.rpe,
        acute_load_7d=body.acute_load_7d,
        chronic_load_weekly_avg=body.chronic_load_weekly_avg,
        morning_stiffness_days=body.morning_stiffness_days,
        one_sided_soreness_hrs=body.one_sided_soreness_hrs,
        journal_entry=body.journal_entry,
        mood_quote=body.mood_quote,
    )
    result = run_simulation(inp)
    return result_to_response(result)


@app.get("/flags/definitions", tags=["reference"])
def flag_definitions():
    """Return all possible injury flag codes and their descriptions."""
    return {
        "OVERREACH": {
            "level": "danger",
            "trigger": "Acute:Chronic workload ratio > 1.5x",
            "description": "Leading predictor of soft-tissue injury in jump-sport athletes.",
        },
        "OVERREACH_MILD": {
            "level": "warning",
            "trigger": "A:C ratio between 1.2-1.5x",
            "description": "Borderline overreach — monitor carefully.",
        },
        "CNS_FATIGUE": {
            "level": "danger",
            "trigger": "HRV drops >= 2 SD below 30-day baseline",
            "description": "Central nervous system recovery is compromised.",
        },
        "TENDON_DESICCATION": {
            "level": "danger",
            "trigger": "Morning stiffness >= 3 consecutive days AND water < 3 L",
            "description": "Inadequate hydration reduces tendon elasticity and lubrication.",
        },
        "BIOMECH_COMPENSATION": {
            "level": "warning",
            "trigger": "One-sided soreness persisting > 48 hours",
            "description": "Overcompensation significantly raises contralateral injury risk.",
        },
        "LOW_DEEP_SLEEP": {
            "level": "warning",
            "trigger": "Deep sleep < 90 minutes",
            "description": "Growth hormone secretion for tissue repair is insufficient.",
        },
        "PROTEIN_DEFICIT": {
            "level": "warning",
            "trigger": "Protein < 1.6 g/kg AND RPE >= 7",
            "description": "Insufficient protein for muscle/tendon repair during high-load phases.",
        },
        "BURNOUT_SENTIMENT": {
            "level": "danger",
            "trigger": "Red-zone keywords detected in journal",
            "description": "Mental burnout language reliably precedes physical injury.",
        },
        "FATIGUE_SENTIMENT": {
            "level": "warning",
            "trigger": "Yellow-zone keywords detected in journal",
            "description": "Physical fatigue signals detected via NLP.",
        },
    }
