import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

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
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Moderate"
    else:
        return "High"

def generate_clinical_narrative(patient: PatientInput, prob: float, tier: str) -> str:
    client = Groq()
    prompt = f"""You are a clinical decision support assistant. A logistic regression model has
assessed a diabetic inpatient and produced the following risk profile.

Patient data:
- Age: {patient.age_numeric}
- Lab procedures: {patient.num_lab_procedures}
- Medications prescribed: {patient.num_medications}
- Medication burden (active changes): {patient.medication_burden}
- Number of diagnoses: {patient.number_diagnoses}
- Diagnostic complexity score: {patient.diagnostic_complexity}
- Emergency admission: {"Yes" if patient.is_emergency else "No"}
- Prior inpatient visits: {patient.number_inpatient}
- Prior emergency visits: {patient.number_emergency}

Model output:
- Prolonged stay probability: {prob:.1%}
- Risk tier: {tier}

Write a concise 3-sentence clinical narrative for a care coordinator.
Sentence 1: State the risk level and the 2-3 most influential factors driving it.
Sentence 2: Identify what this means operationally for the care team.
Sentence 3: Recommend one specific, actionable next step.
Do not use bullet points. Be direct and clinical in tone."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@app.post("/predict")
def predict(patient: PatientInput):
    input_df = pd.DataFrame([patient.model_dump()])[FEATURES]
    prob = float(model.predict_proba(input_df)[0][1])
    tier = assign_risk_tier(prob)
    narrative = generate_clinical_narrative(patient, prob, tier)
    return {
        "risk_probability": round(prob, 4),
        "risk_tier": tier,
        "clinical_narrative": narrative,
        "input_features": patient.model_dump()
    }

@app.get("/health")
def health():
    return {"status": "ok"}