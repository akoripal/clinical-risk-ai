import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agent.clinical_agent import run_clinical_agent

load_dotenv()

app = FastAPI(title="Clinical Risk API")

MODEL_PATH = Path("data/processed/logistic_regression.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "medication_burden", "diagnostic_complexity",
    "is_emergency", "age_numeric"
]

class PatientInput(BaseModel):
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    medication_burden: int
    diagnostic_complexity: int
    is_emergency: int
    age_numeric: float

def assign_risk_tier(prob: float) -> str:
    if prob < 0.2:
        return "Low"
    elif prob < 0.45:
        return "Moderate"
    else:
        return "High"

@app.post("/predict")
def predict(patient: PatientInput):
    input_df = pd.DataFrame([patient.model_dump()])[FEATURES]
    prob = float(model.predict_proba(input_df)[0][1])
    tier = assign_risk_tier(prob)

    agent_result = run_clinical_agent(
        patient_features=patient.model_dump(),
        risk_prob=prob,
        risk_tier=tier
    )

    return {
        "risk_probability": round(prob, 4),
        "risk_tier": tier,
        "clinical_narrative": agent_result["narrative"],
        "agent_tool_calls": agent_result["tool_calls"],
        "agent_turns": agent_result["turns"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}
