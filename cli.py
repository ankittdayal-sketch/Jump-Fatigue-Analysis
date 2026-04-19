#!/usr/bin/env python3
"""
Volleyball Injury Prevention Simulator — CLI
Usage:  python cli.py
        python cli.py --json  (outputs raw JSON)
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import date

from simulator import DailyInput, run_simulation


RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"


def prompt(label: str, default=None, cast=float):
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{hint}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            return cast(raw)
        except ValueError:
            print(f"  Please enter a valid {cast.__name__}.")


def collect_inputs() -> DailyInput:
    print(f"\n{BOLD}{CYAN}=== Volleyball Injury Prevention Simulator ==={RESET}")
    print(f"{GRAY}Press Enter to accept the default value shown in brackets.{RESET}\n")

    print(f"{BOLD}— Physiological —{RESET}")
    hrv_today       = prompt("HRV today (ms)", 65)
    hrv_base        = prompt("HRV 30-day baseline (ms)", 72)
    deep_sleep      = prompt("Deep sleep last night (min)", 95)
    water           = prompt("Water intake today (L)", 2.5)
    protein         = prompt("Protein intake today (g)", 130)
    bw              = prompt("Body weight (kg)", 75)

    print(f"\n{BOLD}— Training load —{RESET}")
    jumps           = prompt("Jumps today (reps)", 120, cast=int)
    rpe             = prompt("RPE 1–10", 6)
    acute           = prompt("7-day total jump load (acute)", 680, cast=int)
    chronic         = prompt("28-day weekly avg jumps (chronic)", 500, cast=int)

    print(f"\n{BOLD}— Asymmetry & stiffness —{RESET}")
    stiff           = prompt("Morning stiffness (consecutive days)", 0, cast=int)
    asymm           = prompt("One-sided soreness (hours)", 0)

    print(f"\n{BOLD}— Journal —{RESET}")
    journal         = input("  How are you feeling today? (free text): ").strip()

    return DailyInput(
        date=date.today(),
        hrv_today=hrv_today,
        hrv_baseline_30d=hrv_base,
        deep_sleep_min=deep_sleep,
        water_liters=water,
        protein_g=protein,
        body_weight_kg=bw,
        jumps_today=jumps,
        rpe=rpe,
        acute_load_7d=acute,
        chronic_load_weekly_avg=chronic,
        morning_stiffness_days=stiff,
        one_sided_soreness_hrs=asymm,
        journal_entry=journal,
    )


def color_zone(zone: str) -> str:
    return {"green": GREEN, "yellow": YELLOW, "red": RED}.get(zone, RESET)


def print_result(result) -> None:
    c = color_zone(result.zone)
    print(f"\n{BOLD}{c}{'─'*50}{RESET}")
    print(f"{BOLD}{c} {result.zone_label.upper()}{RESET}")
    print(f"{GRAY} {result.zone_description}{RESET}")
    print(f"{BOLD}{c}{'─'*50}{RESET}\n")

    print(f"  Readiness score : {BOLD}{result.readiness_score:.0f} / 100{RESET}")
    riskc = RED if result.injury_risk_pct >= 40 else YELLOW if result.injury_risk_pct >= 20 else GREEN
    print(f"  Injury risk     : {BOLD}{riskc}{result.injury_risk_pct:.0f}%{RESET}")
    print(f"  A:C ratio       : {result.ac_ratio:.2f}×")
    print(f"  Protein/kg      : {result.protein_per_kg:.2f} g/kg")
    print(f"  HRV drop        : {result.hrv_drop_sd:.1f} SD\n")

    if result.flags:
        print(f"{BOLD}Active flags:{RESET}")
        for flag in result.flags:
            fc = RED if flag.level == "danger" else YELLOW
            marker = "●" if flag.level == "danger" else "○"
            print(f"  {fc}{marker} {flag.title}{RESET}")
            print(f"    {GRAY}{flag.body}{RESET}")
        print()

    if result.recommendations:
        print(f"{BOLD}Recommendations:{RESET}")
        for rec in result.recommendations:
            print(f"  • {rec}")
        print()

    print(f"{BOLD}Quote of the day:{RESET}")
    print(f"  {CYAN}{result.quote_of_the_day}{RESET}\n")

    if result.sentiment_keywords:
        print(f"{GRAY}Sentiment keywords detected: {', '.join(result.sentiment_keywords)}{RESET}\n")

    print(f"{GRAY}⚠  This simulator is an informational tool, not a medical device.")
    print(f"   If you experience sharp pain or a popping sensation, stop activity")
    print(f"   and consult a sports medicine professional immediately.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Volleyball Injury Prevention Simulator")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    try:
        inp = collect_inputs()
        result = run_simulation(inp)

        if args.json:
            import dataclasses
            def default(o):
                if hasattr(o, "__dataclass_fields__"):
                    return dataclasses.asdict(o)
                if isinstance(o, date):
                    return o.isoformat()
                raise TypeError
            print(json.dumps(dataclasses.asdict(result), default=default, indent=2))
        else:
            print_result(result)

    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
