import streamlit as st
import requests
import json
import os

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000/predict"))

st.set_page_config(
    page_title="Clinical Risk Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Risk Assistant")
st.caption("AI-powered prolonged stay risk assessment for diabetic inpatients")

# Sidebar — patient input form
with st.sidebar:
    st.header("Patient Data")
    st.divider()

    age = st.slider("Patient age", 5, 95, 65, step=10)
    is_emergency = st.selectbox("Admission type", ["Emergency", "Non-Emergency"])
    num_medications = st.number_input("Medications prescribed", 1, 80, 18)
    medication_burden = st.number_input("Medication burden (active changes)", 0, 20, 5)
    num_lab_procedures = st.number_input("Lab procedures", 1, 130, 52)
    num_procedures = st.number_input("Other procedures", 0, 6, 3)
    number_diagnoses = st.number_input("Number of diagnoses", 1, 16, 7)
    diagnostic_complexity = st.number_input("Diagnostic complexity", 1, 3, 3)
    number_inpatient = st.number_input("Prior inpatient visits", 0, 20, 3)
    number_emergency = st.number_input("Prior emergency visits", 0, 20, 2)
    number_outpatient = st.number_input("Prior outpatient visits", 0, 40, 0)

    assess_btn = st.button("Assess Risk", type="primary", use_container_width=True)

# Main area — chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello. I'm your clinical risk assistant. Configure a patient profile in the sidebar and click **Assess Risk** to generate a prolonged stay risk assessment."
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle assessment
if assess_btn:
    payload = {
        "num_lab_procedures": int(num_lab_procedures),
        "num_procedures": int(num_procedures),
        "num_medications": int(num_medications),
        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),
        "number_diagnoses": int(number_diagnoses),
        "medication_burden": int(medication_burden),
        "diagnostic_complexity": int(diagnostic_complexity),
        "is_emergency": 1 if is_emergency == "Emergency" else 0,
        "age_numeric": float(age)
    }

    with st.chat_message("user"):
        st.markdown(f"Assess patient: age {age}, {is_emergency.lower()} admission, "
                    f"{num_medications} medications, {number_diagnoses} diagnoses.")
    st.session_state.messages.append({
        "role": "user",
        "content": f"Assess patient: age {age}, {is_emergency.lower()} admission, "
                   f"{num_medications} medications, {number_diagnoses} diagnoses."
    })

    with st.chat_message("assistant"):
        with st.spinner("Running risk model and generating narrative..."):
            try:
                res = requests.post(API_URL, json=payload)
                data = res.json()

                prob = data["risk_probability"]
                tier = data["risk_tier"]
                narrative = data["clinical_narrative"]

                tier_color = {"Low": "green", "Moderate": "orange", "High": "red"}[tier]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Probability", f"{prob:.1%}")
                with col2:
                    st.markdown(f"**Risk Tier:** :{tier_color}[{tier}]")

                st.divider()
                st.markdown("**Clinical Narrative**")
                st.markdown(narrative)

                response_text = (f"**Risk:** :{tier_color}[{tier}] ({prob:.1%})\n\n"
                                 f"{narrative}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })

            except Exception as e:
                st.error(f"API error: {e}")

# Follow-up chat
if prompt := st.chat_input("Ask a follow-up question about this patient..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.markdown("Follow-up reasoning coming in Phase 5 extension — "
                    "LangGraph agent will handle this.")
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Follow-up reasoning coming in Phase 5 extension."
    })