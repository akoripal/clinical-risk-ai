"""
LLM Evaluation Harness — Clinical Risk Agent
20 synthetic patient profiles with expected risk tiers and narrative
quality checks. Tests that the agent produces grounded, safe, correctly
structured outputs across a diverse patient population.
"""

import json
import sys
import pickle
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.clinical_agent import run_clinical_agent

MODEL_PATH = Path("data/processed/logistic_regression.pkl")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

FEATURES = [
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "medication_burden", "diagnostic_complexity",
    "is_emergency", "age_numeric"
]

# ── Synthetic test profiles ───────────────────────────────────────────────────

TEST_PROFILES = [
    # High risk profiles
    {
        "id": "HIGH_01",
        "expected_tier": "High",
        "description": "Elderly emergency, high complexity",
        "features": {"num_lab_procedures": 80, "num_procedures": 5,
                     "num_medications": 25, "number_outpatient": 0,
                     "number_emergency": 5, "number_inpatient": 6,
                     "number_diagnoses": 9, "medication_burden": 9,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 85}
    },
    {
        "id": "HIGH_02",
        "expected_tier": "High",
        "description": "Frequent utilizer, high med burden",
        "features": {"num_lab_procedures": 75, "num_procedures": 4,
                     "num_medications": 22, "number_outpatient": 1,
                     "number_emergency": 4, "number_inpatient": 5,
                     "number_diagnoses": 8, "medication_burden": 8,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 78}
    },
    {
        "id": "HIGH_03",
        "expected_tier": "High",
        "description": "Max complexity, elderly",
        "features": {"num_lab_procedures": 90, "num_procedures": 6,
                     "num_medications": 28, "number_outpatient": 0,
                     "number_emergency": 6, "number_inpatient": 7,
                     "number_diagnoses": 10, "medication_burden": 10,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 90}
    },
    {
        "id": "HIGH_04",
        "expected_tier": "High",
        "description": "Polypharmacy, emergency, multiple inpatient",
        "features": {"num_lab_procedures": 70, "num_procedures": 4,
                     "num_medications": 20, "number_outpatient": 0,
                     "number_emergency": 3, "number_inpatient": 5,
                     "number_diagnoses": 9, "medication_burden": 9,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 80}
    },
    {
        "id": "HIGH_05",
        "expected_tier": "High",
        "description": "High inpatient history, complex diagnoses",
        "features": {"num_lab_procedures": 85, "num_procedures": 5,
                     "num_medications": 24, "number_outpatient": 0,
                     "number_emergency": 5, "number_inpatient": 8,
                     "number_diagnoses": 9, "medication_burden": 8,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 75}
    },
    # Moderate risk profiles
    {
        "id": "MOD_01",
        "expected_tier": "Moderate",
        "description": "Middle-aged emergency, moderate complexity",
        "features": {"num_lab_procedures": 55, "num_procedures": 3,
                     "num_medications": 15, "number_outpatient": 1,
                     "number_emergency": 2, "number_inpatient": 2,
                     "number_diagnoses": 6, "medication_burden": 5,
                     "diagnostic_complexity": 2, "is_emergency": 1,
                     "age_numeric": 58}
    },
    {
        "id": "MOD_02",
        "expected_tier": "Moderate",
        "description": "Prior inpatient history, non-emergency",
        "features": {"num_lab_procedures": 50, "num_procedures": 2,
                     "num_medications": 14, "number_outpatient": 2,
                     "number_emergency": 1, "number_inpatient": 3,
                     "number_diagnoses": 6, "medication_burden": 4,
                     "diagnostic_complexity": 2, "is_emergency": 0,
                     "age_numeric": 62}
    },
    {
        "id": "MOD_03",
        "expected_tier": "Moderate",
        "description": "Elevated medications, moderate age",
        "features": {"num_lab_procedures": 60, "num_procedures": 3,
                     "num_medications": 18, "number_outpatient": 1,
                     "number_emergency": 2, "number_inpatient": 2,
                     "number_diagnoses": 7, "medication_burden": 6,
                     "diagnostic_complexity": 2, "is_emergency": 1,
                     "age_numeric": 65}
    },
    {
        "id": "MOD_04",
        "expected_tier": "Moderate",
        "description": "Multiple diagnoses, moderate utilization",
        "features": {"num_lab_procedures": 58, "num_procedures": 3,
                     "num_medications": 16, "number_outpatient": 2,
                     "number_emergency": 2, "number_inpatient": 2,
                     "number_diagnoses": 7, "medication_burden": 5,
                     "diagnostic_complexity": 2, "is_emergency": 0,
                     "age_numeric": 60}
    },
    {
        "id": "MOD_05",
        "expected_tier": "Moderate",
        "description": "Borderline high, older non-emergency",
        "features": {"num_lab_procedures": 65, "num_procedures": 3,
                     "num_medications": 17, "number_outpatient": 1,
                     "number_emergency": 2, "number_inpatient": 3,
                     "number_diagnoses": 7, "medication_burden": 6,
                     "diagnostic_complexity": 2, "is_emergency": 0,
                     "age_numeric": 70}
    },
    # Low risk profiles
    {
        "id": "LOW_01",
        "expected_tier": "Low",
        "description": "Young, routine admission, low complexity",
        "features": {"num_lab_procedures": 20, "num_procedures": 1,
                     "num_medications": 5, "number_outpatient": 0,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 2, "medication_burden": 1,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 35}
    },
    {
        "id": "LOW_02",
        "expected_tier": "Low",
        "description": "Middle-aged, elective, minimal history",
        "features": {"num_lab_procedures": 25, "num_procedures": 1,
                     "num_medications": 7, "number_outpatient": 1,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 3, "medication_burden": 2,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 45}
    },
    {
        "id": "LOW_03",
        "expected_tier": "Low",
        "description": "No prior utilization, simple presentation",
        "features": {"num_lab_procedures": 18, "num_procedures": 1,
                     "num_medications": 4, "number_outpatient": 0,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 2, "medication_burden": 1,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 40}
    },
    {
        "id": "LOW_04",
        "expected_tier": "Low",
        "description": "Older but simple, elective, no history",
        "features": {"num_lab_procedures": 30, "num_procedures": 1,
                     "num_medications": 8, "number_outpatient": 1,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 3, "medication_burden": 2,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 65}
    },
    {
        "id": "LOW_05",
        "expected_tier": "Low",
        "description": "Minimal complexity across all dimensions",
        "features": {"num_lab_procedures": 15, "num_procedures": 0,
                     "num_medications": 3, "number_outpatient": 0,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 1, "medication_burden": 0,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 30}
    },
    # Edge cases
    {
        "id": "EDGE_01",
        "expected_tier": "Low",
        "description": "Elderly but elective, low utilization",
        "features": {"num_lab_procedures": 35, "num_procedures": 1,
                     "num_medications": 10, "number_outpatient": 2,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 4, "medication_burden": 2,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 80}
    },
    {
        "id": "EDGE_02",
        "expected_tier": "High",
        "description": "Young but catastrophically complex",
        "features": {"num_lab_procedures": 95, "num_procedures": 6,
                     "num_medications": 30, "number_outpatient": 0,
                     "number_emergency": 7, "number_inpatient": 9,
                     "number_diagnoses": 10, "medication_burden": 10,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 35}
    },
    {
        "id": "EDGE_03",
        "expected_tier": "Moderate",
        "description": "Borderline — moderate on all dimensions",
        "features": {"num_lab_procedures": 45, "num_procedures": 2,
                     "num_medications": 13, "number_outpatient": 1,
                     "number_emergency": 1, "number_inpatient": 1,
                     "number_diagnoses": 5, "medication_burden": 4,
                     "diagnostic_complexity": 2, "is_emergency": 1,
                     "age_numeric": 55}
    },
    {
        "id": "EDGE_04",
        "expected_tier": "Low",
        "description": "High age, zero utilization history",
        "features": {"num_lab_procedures": 28, "num_procedures": 1,
                     "num_medications": 9, "number_outpatient": 0,
                     "number_emergency": 0, "number_inpatient": 0,
                     "number_diagnoses": 3, "medication_burden": 2,
                     "diagnostic_complexity": 1, "is_emergency": 0,
                     "age_numeric": 75}
    },
    {
        "id": "EDGE_05",
        "expected_tier": "High",
        "description": "Max everything except age",
        "features": {"num_lab_procedures": 100, "num_procedures": 6,
                     "num_medications": 30, "number_outpatient": 0,
                     "number_emergency": 8, "number_inpatient": 10,
                     "number_diagnoses": 9, "medication_burden": 10,
                     "diagnostic_complexity": 3, "is_emergency": 1,
                     "age_numeric": 50}
    },
]

# ── Quality checks on narrative ───────────────────────────────────────────────

FORBIDDEN_PHRASES = [
    "i cannot", "i am unable", "as an ai", "i don't have access",
    "consult a doctor", "i recommend you see", "disclaimer",
    "specific medications", "dosage", "prescribe"
]

def check_narrative_quality(narrative: str, profile: dict) -> list[str]:
    failures = []
    text = narrative.lower()

    # Must be 2–5 sentences
    sentences = [s.strip() for s in narrative.split(".") if len(s.strip()) > 10]
    if len(sentences) < 2:
        failures.append(f"TOO_SHORT: Only {len(sentences)} sentence(s)")
    if len(sentences) > 6:
        failures.append(f"TOO_LONG: {len(sentences)} sentences (max 5)")

    # Must not contain forbidden phrases
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            failures.append(f"FORBIDDEN_PHRASE: '{phrase}'")

    # Must not be empty
    if len(narrative.strip()) < 50:
        failures.append("EMPTY_OR_TOO_SHORT: Narrative under 50 chars")

    return failures

def assign_risk_tier(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Moderate"
    else:
        return "High"

# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation():
    results = []
    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print("CLINICAL RISK AGENT — LLM EVALUATION HARNESS")
    print(f"{'='*60}\n")

    for profile in TEST_PROFILES:
        pid = profile["id"]
        features = profile["features"]
        expected = profile["expected_tier"]

        input_df = pd.DataFrame([features])[FEATURES]
        prob = float(MODEL.predict_proba(input_df)[0][1])
        actual_tier = assign_risk_tier(prob)

        agent_result = run_clinical_agent(
            patient_features=features,
            risk_prob=prob,
            risk_tier=actual_tier
        )
        narrative = agent_result["narrative"]
        tools_called = [t["tool"] for t in agent_result["tool_calls"]]
        quality_failures = check_narrative_quality(narrative, profile)

        tier_match = actual_tier == expected
        tools_ok = ("retrieve_similar_cases" in tools_called and
                    "check_high_risk_flags" in tools_called)
        quality_ok = len(quality_failures) == 0

        overall_pass = tier_match and tools_ok and quality_ok

        if overall_pass:
            passed += 1
            status = "✓ PASS"
        else:
            failed += 1
            status = "✗ FAIL"

        print(f"[{status}] {pid} — {profile['description']}")
        print(f"         Tier: expected={expected}, actual={actual_tier} "
              f"{'✓' if tier_match else '✗'}")
        print(f"         Tools called: {tools_called} "
              f"{'✓' if tools_ok else '✗'}")
        if quality_failures:
            for qf in quality_failures:
                print(f"         Quality: ✗ {qf}")
        print()

        results.append({
            "id": pid,
            "description": profile["description"],
            "expected_tier": expected,
            "actual_tier": actual_tier,
            "tier_match": tier_match,
            "tools_called": tools_called,
            "tools_ok": tools_ok,
            "quality_failures": quality_failures,
            "passed": overall_pass,
            "narrative": narrative
        })

    total = len(TEST_PROFILES)
    print(f"{'='*60}")
    print(f"RESULTS: {passed}/{total} passed ({100*passed//total}%)")
    print(f"  Tier accuracy:    "
          f"{sum(r['tier_match'] for r in results)}/{total}")
    print(f"  Tools compliance: "
          f"{sum(r['tools_ok'] for r in results)}/{total}")
    print(f"  Narrative quality:{sum(r['quality_failures']==[] for r in results)}/{total}")
    print(f"{'='*60}\n")

    return results

if __name__ == "__main__":
    run_evaluation()
    