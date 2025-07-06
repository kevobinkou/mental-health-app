import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("mental_health_model.pkl")

# Set page config
st.set_page_config(page_title="Student Mental Health Prediction", layout="centered")

# Custom header
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='color:#4B8BBE;'>🧠 Student Mental Health Prediction</h1>
        <p style='font-size:18px;'>Predict whether a student may need mental health support using a trained ML model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input form
with st.form("mental_health_form"):
    gender = st.selectbox("Choose your gender", ["Female", "Male"])
    age = st.number_input("Enter your age", min_value=16, max_value=40, step=1)
    course = st.selectbox("What is your course?", ["Engineering", "Business", "Arts", "Science", "Other"])
    year = st.selectbox("Current year of study", [1, 2, 3, 4])
    cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=4.0, step=0.1)
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Other"])
    anxiety = st.radio("Do you have Anxiety?", ["Yes", "No"])
    panic_attack = st.radio("Do you have Panic attack?", ["Yes", "No"])
    specialist = st.radio("Did you seek any specialist for a treatment?", ["Yes", "No"])
    submit = st.form_submit_button("🔍 Predict")

# Process after submit
if submit:
    gender_map = {"Female": 0, "Male": 1}
    marital_map = {"Single": 0, "Married": 1, "Other": 2}
    course_map = {"Engineering": 0, "Business": 1, "Arts": 2, "Science": 3, "Other": 4}
    binary_map = {"Yes": 1, "No": 0}

    input_data = np.array([[
        gender_map[gender],
        age,
        course_map[course],
        year,
        cgpa,
        marital_map[marital_status],
        binary_map[anxiety],
        binary_map[panic_attack],
        binary_map[specialist]
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The student **may need mental health support**. Please consider counseling.")
    else:
        st.success("✅ The student **does not currently show strong indicators** of needing counseling.")

# Custom footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size: 14px;'>
    Made with ❤️ by Kelvin Maina | <a href="https://github.com/kevobinkou" target="_blank">GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)
