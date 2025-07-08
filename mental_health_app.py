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
import time

# ----------------- Load trained model -----------------
model = joblib.load("mental_health_model.pkl")

# ----------------- Logo -----------------
logo = Image.open("student_grade_predictor.jpg")
st.image(logo, width=180)

# ----------------- MySQL Connection Setup with Retry -----------------
@st.cache_resource
def get_db_connection():
    max_retries = 3
    delay_seconds = 3
    attempt = 0

    while attempt < max_retries:
        try:
            conn = mysql.connector.connect(
                host=st.secrets["DB_HOST"],
                user=st.secrets["DB_USER"],
                password=st.secrets["DB_PASSWORD"],
                database=st.secrets["DB_NAME"],
                port=3306,
                connection_timeout=10
            )
            if conn.is_connected():
                return conn
        except mysql.connector.Error as err:
            attempt += 1
            st.warning(f"🔄 Retry {attempt}/{max_retries} – Database connection failed: {err}")
            time.sleep(delay_seconds)

    st.error("❌ Could not connect to the database after multiple attempts. Please try again later.")
    return None

conn = get_db_connection()
if conn is None:
    st.stop()

try:
    cursor = conn.cursor()
except mysql.connector.Error as err:
    st.error(f"❌ Failed to create database cursor: {err}")
    st.stop()

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
    st.markdown("<h2 style='color:#4B8BBE;'>🔐 Student Mental Health Login</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.user = username
            st.session_state.role = users[username]["role"]
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
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
            <p style='font-size:18px; color:#555;'>Predict whether a student may need mental health support using a trained ML model.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("mental_health_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=16, max_value=40, step=1)
            course = st.selectbox("Course", ["Engineering", "Business", "Arts", "Science", "Other"])
            year = st.selectbox("Year of Study", [1, 2, 3, 4])
        with col2:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, step=0.1)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"])
            anxiety = st.radio("Do you have Anxiety?", ["Yes", "No"])
            panic_attack = st.radio("Do you have Panic Attacks?", ["Yes", "No"])
            specialist = st.radio("Have you seen a specialist?", ["Yes", "No"])
        submit = st.form_submit_button("🔍 Predict")

    if submit:
        gender_map = {"Female": 0, "Male": 1}
        marital_map = {"Single": 0, "Married": 1, "Other": 2}
        course_map = {"Engineering": 0, "Business": 1, "Arts": 2, "Science": 3, "Other": 4}
        binary_map = {"Yes": 1, "No": 0}

        input_data = np.array([[ 
            gender_map[gender],
            int(age),
            course_map[course],
            int(year),
            float(cgpa),
            marital_map[marital_status],
            binary_map[anxiety],
            binary_map[panic_attack],
            binary_map[specialist]
        ]])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = round(float(np.max(proba)) * 100, 2)  # ✅ Native float

        result_label = "Needs Support" if prediction == 1 else "No Strong Indicators"

        if prediction == 1:
            st.error("⚠️ The student may need mental health support. Please consider counseling.")
        else:
            st.success("✅ No strong indicators of needing mental health support.")

        st.info(f"🔍 Model Confidence: **{confidence}%**")

        # ✅ Safe casting of all fields for MySQL insert
        record = (
            str(st.session_state.user),
            int(gender_map[gender]),
            int(age),
            int(course_map[course]),
            int(year),
            float(cgpa),
            int(marital_map[marital_status]),
            int(binary_map[anxiety]),
            int(binary_map[panic_attack]),
            int(binary_map[specialist]),
            result_label,
            float(confidence)
        )

        insert_query = """
            INSERT INTO submissions (
                username, gender, age, course, year, cgpa, marital_status,
                anxiety, panic_attack, specialist, prediction_result, confidence_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            cursor.execute(insert_query, record)
            conn.commit()
        except mysql.connector.Error as err:
            st.error(f"❌ Database insert error: {err}")

        # ---------- Visualization ----------
        st.subheader("📈 Your Submission Summary")

        st.markdown("### 🎓 Academic Profile")
        fig, ax = plt.subplots()
        ax.bar(["Age", "CGPA", "Year"], [age, cgpa, year], color=["#4B8BBE", "#79D2DE", "#FFD166"])
        ax.set_ylabel("Values")
        st.pyplot(fig)

        st.markdown("### 🧠 Mental Health Indicators")
        mh_labels = ["Anxiety", "Panic Attack", "Specialist Visit"]
        mh_values = [binary_map[anxiety], binary_map[panic_attack], binary_map[specialist]]
        fig2, ax2 = plt.subplots()
        ax2.pie(mh_values, labels=mh_labels, autopct=lambda p: f'{p:.0f}%' if p > 0 else '',
                colors=['#FF9999','#66B2FF','#99FF99'])
        ax2.set_title("Reported Symptoms")
        st.pyplot(fig2)

# ----------------- Admin View -----------------
elif st.session_state.role == "admin":
    st.subheader("📋 Admin Dashboard")
    try:
        cursor.execute("SELECT * FROM submissions ORDER BY submission_time DESC")
        data = cursor.fetchall()
        if data:
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No submissions yet.")
    except mysql.connector.Error as err:
        st.error(f"⚠️ Failed to load submissions: {err}")

# ----------------- Footer -----------------
st.markdown("""
    <hr>
    <p style='text-align:center; font-size: 14px; color: #888;'>
    Made with ❤️ by Kelvin Maina |
    <a href="https://github.com/kevobinkou" target="_blank">GitHub</a>
    </p>
""", unsafe_allow_html=True)
