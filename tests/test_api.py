"""
API integration tests — run with: pytest tests/test_api.py -v
These test the full predict endpoint without requiring a running server.
"""
import sys
import pickle
import pytest
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.main import app

client = TestClient(app)

VALID_PATIENT = {
    "num_lab_procedures": 52,
    "num_procedures": 3,
    "num_medications": 18,
    "number_outpatient": 0,
    "number_emergency": 2,
    "number_inpatient": 3,
    "number_diagnoses": 7,
    "medication_burden": 5,
    "diagnostic_complexity": 3,
    "is_emergency": 1,
    "age_numeric": 75.0
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_200():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200

def test_predict_response_schema():
    response = client.post("/predict", json=VALID_PATIENT)
    data = response.json()
    assert "risk_probability" in data
    assert "risk_tier" in data
    assert "clinical_narrative" in data
    assert "agent_tool_calls" in data

def test_risk_probability_range():
    response = client.post("/predict", json=VALID_PATIENT)
    prob = response.json()["risk_probability"]
    assert 0.0 <= prob <= 1.0

def test_risk_tier_valid_values():
    response = client.post("/predict", json=VALID_PATIENT)
    tier = response.json()["risk_tier"]
    assert tier in ["Low", "Moderate", "High"]

def test_agent_calls_both_tools():
    response = client.post("/predict", json=VALID_PATIENT)
    tool_calls = response.json()["agent_tool_calls"]
    tool_names = [t["tool"] for t in tool_calls]
    assert "retrieve_similar_cases" in tool_names
    assert "check_high_risk_flags" in tool_names

def test_narrative_not_empty():
    response = client.post("/predict", json=VALID_PATIENT)
    narrative = response.json()["clinical_narrative"]
    assert len(narrative.strip()) > 50

def test_low_risk_patient():
    low_risk = {**VALID_PATIENT,
                "number_inpatient": 0, "number_emergency": 0,
                "num_medications": 3, "medication_burden": 0,
                "age_numeric": 35.0, "is_emergency": 0}
    response = client.post("/predict", json=low_risk)
    assert response.status_code == 200
    assert response.json()["risk_tier"] == "Low"

def test_missing_field_returns_422():
    bad_payload = {k: v for k, v in VALID_PATIENT.items()
                   if k != "age_numeric"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    