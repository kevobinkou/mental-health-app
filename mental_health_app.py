import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Load Model ---
model = joblib.load("mental_health_model.pkl")

# --- Page Config ---
st.set_page_config(
    page_title="Student Mental Health Prediction",
    layout="centered",
    page_icon="🧠"
)

# --- Display Logo ---
logo = Image.open("student_grade_predictor.jpg")
st.image(logo, width=180)

# --- LOGIN AUTHENTICATION ---
users = {
    "student1": {"password": "stud123", "role": "student"},
    "admin1": {"password": "admin123", "role": "admin"}
}

def login():
    st.sidebar.header("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = users.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
        else:
            st.sidebar.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# --- Welcome Message ---
st.success(f"Welcome {st.session_state.username}!")

# --- STUDENT ACCESS SECTION ---
if st.session_state.role == "student":
    st.info("You are logged in as a **student**.")

    # --- Custom Header ---
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color:#4B8BBE;'>🧠 Student Mental Health Prediction</h1>
            <p style='font-size:18px;'>Predict whether a student may need mental health support using a trained ML model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Input Form ---
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

    # --- Process Input ---
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
        proba = model.predict_proba(input_data)[0]
        confidence = round(np.max(proba) * 100, 2)

        # --- Show Result ---
        if prediction == 1:
            st.error("⚠️ The student **may need mental health support**. Please consider counseling.")
        else:
            st.success("✅ The student **does not currently show strong indicators** of needing counseling.")
        
        st.info(f"📊 Model confidence: **{confidence}%**")

        # --- Log Submission ---
        log_submission = {
            "user": st.session_state.username,
            "age": age,
            "cgpa": cgpa,
            "year": year,
            "prediction": prediction,
            "confidence": confidence
        }
        log_file = "submission_log.csv"
        pd.DataFrame([log_submission]).to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

        # --- Visualizations ---
        st.subheader("📈 Submitted Input Summary")

        # Bar Chart: Academic Factors
        st.markdown("### Academic Info")
        fig, ax = plt.subplots()
        ax.bar(["Age", "CGPA", "Study Year"], [age, cgpa, year], color="#4B8BBE")
        ax.set_ylabel("Value")
        st.pyplot(fig)

        # Pie Chart: Mental Health Indicators
        st.markdown("### Reported Mental Health Indicators")
        mh_labels = ["Anxiety", "Panic Attack", "Sought Specialist"]
        mh_values = [
            binary_map[anxiety],
            binary_map[panic_attack],
            binary_map[specialist]
        ]
        fig2, ax2 = plt.subplots()
        ax2.pie(mh_values, labels=mh_labels, autopct=lambda p: f'{p:.0f}%' if p > 0 else '', colors=['#FF9999','#66B2FF','#99FF99'])
        ax2.set_title("Mental Health Responses")
        st.pyplot(fig2)

# --- ADMIN ACCESS SECTION ---
elif st.session_state.role == "admin":
    st.info("You are logged in as an **admin**.")
    st.subheader("📋 Admin Dashboard")

    log_file = "submission_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        st.dataframe(df)
    else:
        st.warning("No student submissions found yet.")

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size: 14px;'>
    Made with ❤️ by Kelvin Maina | <a href="https://github.com/kevobinkou" target="_blank">GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)
