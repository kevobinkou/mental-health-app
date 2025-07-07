# ----------------- Page configuration -----------------
import streamlit as st
st.set_page_config(
    page_title="Student Mental Health Prediction",
    layout="centered",
    page_icon="🧠"
)

# ----------------- Imports -----------------
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import mysql.connector

# ----------------- MySQL Connection Setup -----------------
@st.cache_resource
def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["DB_HOST"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_NAME"],
        port=3306
    )

conn = get_db_connection()
cursor = conn.cursor()

# ----------------- Load trained model -----------------
model = joblib.load("mental_health_model.pkl")

# ----------------- Logo -----------------
logo = Image.open("student_grade_predictor.jpg")
st.image(logo, width=180)

# ----------------- Login Authentication -----------------
users = {
    "admin": {"password": "adminpass", "role": "admin"},
    "student": {"password": "studpass", "role": "student"}
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = ""
    st.session_state.role = ""

if not st.session_state.authenticated:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.user = username
            st.session_state.role = users[username]["role"]
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ----------------- Logout Option -----------------
st.sidebar.markdown(f"👤 Logged in as: `{st.session_state.user}`")
st.sidebar.markdown("---")

if "logout_confirm" not in st.session_state:
    st.session_state.logout_confirm = False

if st.sidebar.button("🚪 Logout"):
    st.session_state.logout_confirm = True

if st.session_state.logout_confirm:
    st.sidebar.markdown(
        """
        <div style="background-color: #FFF3CD; padding: 15px; border-radius: 10px; border: 1px solid #FFEEBA;">
            <strong>⚠️ Are you sure you want to logout?</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Yes"):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("❌ No"):
            st.session_state.logout_confirm = False

# ----------------- Student View -----------------
if st.session_state.role == "student":
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color:#4B8BBE;'>🧠 Student Mental Health Prediction</h1>
            <p style='font-size:18px;'>Predict whether a student may need mental health support using a trained ML model.</p>
        </div>
    """, unsafe_allow_html=True)

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

        if prediction == 1:
            st.error("⚠️ The student **may need mental health support**. Please consider counseling.")
        else:
            st.success("✅ The student **does not currently show strong indicators** of needing counseling.")

        st.info(f"📊 Model confidence: **{confidence}%**")

        insert_query = """
            INSERT INTO submissions (
                username, gender, age, course, year, cgpa, marital_status,
                anxiety, panic_attack, specialist, prediction_result, confidence_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        result_label = "Needs Support" if prediction == 1 else "No Strong Indicators"

        record = (
            st.session_state.user,
            gender, age, course, year, cgpa, marital_status,
            binary_map[anxiety], binary_map[panic_attack], binary_map[specialist],
            result_label, confidence
        )
        cursor.execute(insert_query, record)
        conn.commit()

        st.subheader("📈 Submitted Input Summary")

        st.markdown("### Academic Info")
        fig, ax = plt.subplots()
        ax.bar(["Age", "CGPA", "Study Year"], [age, cgpa, year], color="#4B8BBE")
        ax.set_ylabel("Value")
        st.pyplot(fig)

        st.markdown("### Reported Mental Health Indicators")
        mh_labels = ["Anxiety", "Panic Attack", "Sought Specialist"]
        mh_values = [binary_map[anxiety], binary_map[panic_attack], binary_map[specialist]]
        fig2, ax2 = plt.subplots()
        ax2.pie(mh_values, labels=mh_labels, autopct=lambda p: f'{p:.0f}%' if p > 0 else '',
                colors=['#FF9999','#66B2FF','#99FF99'])
        ax2.set_title("Mental Health Responses")
        st.pyplot(fig2)

# ----------------- Admin View -----------------
elif st.session_state.role == "admin":
    st.subheader("📋 Admin Dashboard")
    cursor.execute("SELECT * FROM submissions ORDER BY submission_time DESC")
    data = cursor.fetchall()
    if data:
        st.dataframe(data, use_container_width=True)
    else:
        st.info("No submissions yet.")

# ----------------- Footer -----------------
st.markdown("""
    <hr>
    <p style='text-align:center; font-size: 14px;'>
    Made with ❤️ by Kelvin Maina | <a href="https://github.com/kevobinkou" target="_blank">GitHub</a>
    </p>
""", unsafe_allow_html=True)
