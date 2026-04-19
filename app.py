from datetime import date
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simulator import DailyInput, run_simulation

app = FastAPI(title="Volleyball Injury Prevention Simulator")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SimulateRequest(BaseModel):
    entry_date: str = ""
    hrv_today: float = 65
    hrv_baseline_30d: float = 72
    deep_sleep_min: float = 95
    water_liters: float = 2.5
    protein_g: float = 130
    body_weight_kg: float = 75
    jumps_today: int = 120
    rpe: float = 6
    acute_load_7d: int = 680
    chronic_load_weekly_avg: int = 500
    morning_stiffness_days: int = 0
    one_sided_soreness_hrs: float = 0
    journal_entry: str = ""
    mood_quote: str = ""


@app.get("/")
def root():
    return {"status": "ok", "message": "Volleyball Injury Prevention Simulator is running. Go to /docs to use it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def simulate(body: SimulateRequest):
    inp = DailyInput(
        date=date.today(),
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
    r = run_simulation(inp)
    return {
        "date": str(r.date),
        "readiness_score": r.readiness_score,
        "injury_risk_pct": r.injury_risk_pct,
        "zone": r.zone,
        "zone_label": r.zone_label,
        "zone_description": r.zone_description,
        "flags": [{"level": f.level, "code": f.code, "title": f.title, "body": f.body} for f in r.flags],
        "ac_ratio": r.ac_ratio,
        "protein_per_kg": r.protein_per_kg,
        "hrv_drop_sd": r.hrv_drop_sd,
        "sentiment_zone": r.sentiment_zone,
        "sentiment_keywords": r.sentiment_keywords,
        "quote_of_the_day": r.quote_of_the_day,
        "recommendations": r.recommendations,
    }


@app.get("/flags/definitions")
def flag_definitions():
    return {
        "OVERREACH": "A:C workload ratio > 1.5x — leading predictor of soft-tissue injury.",
        "OVERREACH_MILD": "A:C ratio 1.2-1.5x — borderline overreach, monitor carefully.",
        "CNS_FATIGUE": "HRV drops >= 2 SD below 30-day baseline.",
        "TENDON_DESICCATION": "Morning stiffness >= 3 days AND water < 3L.",
        "BIOMECH_COMPENSATION": "One-sided soreness > 48 hours.",
        "LOW_DEEP_SLEEP": "Deep sleep < 90 minutes.",
        "PROTEIN_DEFICIT": "Protein < 1.6g/kg AND RPE >= 7.",
        "BURNOUT_SENTIMENT": "Red-zone keywords detected in journal.",
        "FATIGUE_SENTIMENT": "Yellow-zone keywords detected in journal.",
    }
