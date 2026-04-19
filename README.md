# Volleyball Injury Prevention Simulator

DNN-inspired daily readiness and injury risk engine for volleyball athletes.

## Files

| File | Purpose |
|---|---|
| `simulator.py` | Core logic — all flag rules, scoring, NLP sentiment, data models |
| `app.py` | FastAPI REST API — wrap `simulator.py` as a web service |
| `cli.py` | Interactive command-line interface |
| `requirements.txt` | Python dependencies |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the REST API

```bash
uvicorn app:app --reload
```

- Interactive API docs: http://127.0.0.1:8000/docs
- Redoc: http://127.0.0.1:8000/redoc

### 3. Run the CLI

```bash
python cli.py
# or for raw JSON output:
python cli.py --json
```

---

## API Usage

### POST /simulate

Send a JSON body with daily metrics and receive a full readiness assessment.

**Example request:**
```json
{
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
  "journal_entry": "Feeling sluggish, knees are achy."
}
```

**Example response:**
```json
{
  "readiness_score": 47.0,
  "injury_risk_pct": 53.0,
  "zone": "red",
  "zone_label": "Red zone — rest day recommended",
  "flags": [
    {
      "level": "danger",
      "code": "OVERREACH",
      "title": "High strain warning — acute overreach",
      "body": "Your 7-day load is 1.64× your chronic baseline..."
    }
  ],
  "recommendations": ["Reduce jump volume by 30–40% for the next 2–3 days."],
  "quote_of_the_day": "\"I am tired of being tired.\" — Your mind is signalling..."
}
```

### GET /flags/definitions

Returns all possible flag codes, their trigger conditions, and descriptions.

---

## Flag Reference

| Code | Level | Trigger |
|---|---|---|
| `OVERREACH` | danger | A:C workload ratio > 1.5× |
| `OVERREACH_MILD` | warning | A:C ratio 1.2–1.5× |
| `CNS_FATIGUE` | danger | HRV drops ≥ 2 SD below 30-day baseline |
| `TENDON_DESICCATION` | danger | Stiffness ≥ 3 days + water < 3 L |
| `BIOMECH_COMPENSATION` | warning | One-sided soreness > 48 hours |
| `LOW_DEEP_SLEEP` | warning | Deep sleep < 90 minutes |
| `PROTEIN_DEFICIT` | warning | Protein < 1.6 g/kg + RPE ≥ 7 |
| `BURNOUT_SENTIMENT` | danger | Red-zone keywords in journal |
| `FATIGUE_SENTIMENT` | warning | Yellow-zone keywords in journal |

---

## Extending the App

- **Database logging**: Add SQLite or PostgreSQL via SQLAlchemy to store daily entries and compute the rolling 30-day HRV baseline automatically.
- **Mobile frontend**: The FastAPI endpoint is CORS-enabled — connect React Native, Flutter, or Swift/Kotlin directly.
- **Real DNN**: Replace the rule engine in `simulator.py` with a trained scikit-learn or PyTorch model using historical athlete data.
- **Push notifications**: Integrate with Firebase or APNs to send flag alerts to athletes each morning.

---

> **Medical disclaimer**: This tool is a simulator and informational aid, not a medical device. If an athlete experiences sharp pain or a popping sensation, they should stop activity and consult a sports medicine professional immediately.
