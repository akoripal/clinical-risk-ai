import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PREDICTIONS_PATH = Path("data/processed/test_predictions.csv")

# ── Tool 1: retrieve similar cases from historical data ──────────────────────

def retrieve_similar_cases(features: dict, n: int = 5) -> str:
    """Find n most similar historical patients and their actual outcomes."""
    try:
        df = pd.read_csv(PREDICTIONS_PATH)
        feature_cols = [
            "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient",
            "number_diagnoses", "medication_burden", "diagnostic_complexity",
            "is_emergency", "age_numeric"
        ]
        available = [c for c in feature_cols if c in df.columns]
        query = np.array([features.get(c, 0) for c in available], dtype=float)

        matrix = df[available].fillna(0).values.astype(float)
        norms = np.linalg.norm(matrix - query, axis=1)
        top_idx = np.argsort(norms)[:n]
        similar = df.iloc[top_idx][available + ["actual", "lr_prob", "risk_tier"]].copy()
        similar["distance"] = norms[top_idx].round(3)

        prolonged_rate = similar["actual"].mean()
        cases = similar.to_dict(orient="records")

        return json.dumps({
            "similar_case_count": n,
            "prolonged_stay_rate_in_similar": round(prolonged_rate, 3),
            "cases": cases
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool 2: rule-based clinical risk flags ───────────────────────────────────

def check_high_risk_flags(features: dict) -> str:
    """Apply deterministic clinical rules to flag specific risk patterns."""
    flags = []

    if features.get("age_numeric", 0) >= 75 and features.get("is_emergency") == 1:
        flags.append("ELDERLY_EMERGENCY: Patient is 75+ and admitted via emergency — "
                     "elevated discharge planning complexity.")

    if features.get("number_inpatient", 0) >= 3:
        flags.append("FREQUENT_INPATIENT: 3+ prior inpatient visits — "
                     "pattern suggests chronic instability or care gaps.")

    if features.get("medication_burden", 0) >= 7:
        flags.append("HIGH_MED_BURDEN: 7+ active medication changes — "
                     "polypharmacy risk, pharmacist review recommended.")

    if (features.get("number_emergency", 0) >= 2 and
            features.get("number_inpatient", 0) >= 2):
        flags.append("HIGH_UTILIZER: Multiple prior emergency + inpatient visits — "
                     "consider care management enrollment.")

    if features.get("number_diagnoses", 0) >= 8:
        flags.append("HIGH_COMPLEXITY: 8+ diagnoses — "
                     "multidisciplinary team review strongly recommended.")

    if not flags:
        flags.append("NO_CRITICAL_FLAGS: No high-risk rule triggers detected.")

    return json.dumps({"flags": flags, "flag_count": len(flags)})


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar_cases",
            "description": (
                "Retrieve the most similar historical diabetic inpatients "
                "from the training dataset and return their actual prolonged "
                "stay outcomes. Use this to ground your assessment in real data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Patient feature dictionary"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of similar cases to retrieve",
                        "default": 5
                    }
                },
                "required": ["features"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_high_risk_flags",
            "description": (
                "Apply deterministic clinical rules to check for specific "
                "high-risk patterns such as elderly emergency admissions, "
                "frequent utilization, or polypharmacy. Always call this "
                "before producing your final recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Patient feature dictionary"
                    }
                },
                "required": ["features"]
            }
        }
    }
]

TOOL_MAP = {
    "retrieve_similar_cases": retrieve_similar_cases,
    "check_high_risk_flags": check_high_risk_flags,
}

SYSTEM_PROMPT = """You are a clinical decision support agent assisting care 
coordinators in a hospital setting. You have access to two tools:

1. retrieve_similar_cases — finds historically similar patients and their 
   actual outcomes from a validated dataset.
2. check_high_risk_flags — applies deterministic clinical rules to detect 
   specific risk patterns.

When assessing a patient:
- Always call both tools before producing your final response.
- Ground your narrative in the tool outputs — do not invent clinical details.
- Be concise and direct. Care coordinators are busy.
- Never recommend specific medications or dosages.
- Format your final response as three sentences:
  Sentence 1: Risk level + key drivers from tool results.
  Sentence 2: Operational implication for the care team.
  Sentence 3: One specific, actionable next step."""


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_clinical_agent(patient_features: dict, risk_prob: float,
                       risk_tier: str) -> dict:
    """
    Run the agentic loop: LLM reasons, calls tools, reasons again,
    produces a grounded clinical narrative.
    """
    client = Groq()

    user_message = f"""Please assess this diabetic inpatient for prolonged stay risk.

Patient features: {json.dumps(patient_features, indent=2)}

ML model output:
- Prolonged stay probability: {risk_prob:.1%}
- Risk tier: {risk_tier}

Use your tools to retrieve similar cases and check clinical flags before 
producing your final assessment."""

    messages = [{"role": "user", "content": user_message}]
    tool_calls_made = []

    # Agentic loop — runs until model stops calling tools
    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # If the model wants to call tools
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            # Execute each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = TOOL_MAP[fn_name](**fn_args)
                tool_calls_made.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result": json.loads(result)
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        else:
            # Model produced final response — exit loop
            final_narrative = msg.content
            return {
                "narrative": final_narrative,
                "tool_calls": tool_calls_made,
                "turns": len([m for m in messages if m["role"] == "assistant"])
            }
            