import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("cdss_model.pkl")

st.title("🏥 AI-Based Clinical Decision Support System")

st.warning("⚠️ This system is for educational purposes only and not a substitute for professional medical advice.")

st.header("Patient Symptom Entry Form")
st.subheader("Select severity (0=None, 1=Mild, 2=Moderate, 3=Severe)")

fever = st.slider("Fever", 0, 3, 0)
cough = st.slider("Cough", 0, 3, 0)
headache = st.slider("Headache", 0, 3, 0)
vomiting = st.slider("Vomiting", 0, 3, 0)
fatigue = st.slider("Fatigue", 0, 3, 0)
short_breath = st.slider("Shortness of Breath", 0, 3, 0)
chest_pain = st.slider("Chest Pain", 0, 3, 0)
sore_throat = st.slider("Sore Throat", 0, 3, 0)
runny_nose = st.slider("Runny Nose", 0, 3, 0)
nausea = st.slider("Nausea", 0, 3, 0)
diarrhea = st.slider("Diarrhea", 0, 3, 0)
joint_pain = st.slider("Joint Pain", 0, 3, 0)
skin_rash = st.slider("Skin Rash", 0, 3, 0)
dizziness = st.slider("Dizziness", 0, 3, 0)

age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0

if st.button("Predict Condition"):
    input_data = np.array([[fever, cough, headache, vomiting, fatigue, short_breath,
                            chest_pain, sore_throat, runny_nose, nausea, diarrhea,
                            joint_pain, skin_rash, dizziness, age, gender_val]])

    prediction = model.predict(input_data)[0]

    # Risk logic
    severe_count = sum([1 for x in [fever, cough, headache, vomiting, fatigue,
                                    short_breath, chest_pain] if x >= 2])

    if severe_count >= 4:
        risk = "🔴 High Risk – Consult a doctor immediately"
    elif severe_count >= 2:
        risk = "🟡 Medium Risk – Medical checkup recommended"
    else:
        risk = "🟢 Low Risk"

    st.success(f"Predicted Condition: {prediction}")
    st.info(f"Risk Level: {risk}")
