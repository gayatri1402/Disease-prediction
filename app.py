import streamlit as st
import joblib
import numpy as np

model = joblib.load("disease_model.pkl")

st.title("🏥 Disease Prediction System")
st.write("Select your symptoms:")

fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
vomiting = st.checkbox("Vomiting")
fatigue = st.checkbox("Fatigue")
cold = st.checkbox("Cold")
body_pain = st.checkbox("Body Pain")

if st.button("Predict Disease"):
    input_data = np.array([[int(fever), int(cough), int(headache),
                            int(vomiting), int(fatigue), int(cold), int(body_pain)]])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Disease: {prediction[0]}")
