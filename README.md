# 🧠 Student Mental Health Prediction App

A youth-focused, mobile-friendly Streamlit web app that predicts whether a student may need **mental health support**, powered by a trained machine learning model and connected to a **Clever Cloud MySQL database** for secure submission storage and admin analysis.

---

## 🚀 Live Demo

👉 [https://kevobinkou-mh-app.streamlit.app](https://kevobinkou-mh-app.streamlit.app)  
*(Click the link to access the live app!)*

---

## 🧩 Features

✅ Student & Admin login authentication  
✅ Predicts mental health support needs using ML  
✅ Model confidence score output  
✅ Stores all submissions in MySQL database  
✅ Admin dashboard to view all data  
✅ Pie & bar chart visualizations  
✅ Fully mobile-friendly with colorful, calm UI  
✅ Hosted on Streamlit Cloud + Clever Cloud DB

---

## 📸 App Preview

| Mobile View                              | Admin Dashboard                          |
|------------------------------------------|-------------------------------------------|
| ![Mobile Screenshot](screenshots/mobile.png) | ![Admin Screenshot](screenshots/admin.png) |

> *(Make sure to add actual screenshots in the `screenshots/` folder!)*

---

## 🔐 Login Credentials (Test)

| Role    | Username  | Password   |
|---------|-----------|------------|
| Student | `student` | `studpass` |
| Admin   | `admin`   | `adminpass` |

> Only the **admin** can view all submissions and monitor prediction history.

---

## 🧬 Model Inputs

- Gender
- Age
- Course
- Year of Study
- CGPA
- Marital Status
- Mental Health indicators:
  - Anxiety
  - Panic Attacks
  - Sought Specialist Help

---

## 🎯 Output

- **Prediction**: Whether the student needs mental health support
- **Confidence Score**: Model's certainty in prediction
- **Visual Summary**: Bar and pie charts of input and symptoms

---

## 💻 Tech Stack

- `Python`
- `Streamlit`
- `scikit-learn`
- `joblib`
- `MySQL (Clever Cloud)`
- `matplotlib`, `Pillow (PIL)`
- `GitHub + Streamlit Cloud`

---

## ⚙️ Run Locally

```bash
# Clone the project
git clone https://github.com/kevobinkou/mental-health-app.git
cd mental-health-app

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Streamlit secrets
mkdir .streamlit
touch .streamlit/secrets.toml
Paste your database secrets in .streamlit/secrets.toml like this:

toml
Copy
Edit
DB_HOST = "your_db_host"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"
DB_NAME = "your_db_name"
Then run:

bash
Copy
Edit
streamlit run mental_health_app.py
📁 Dataset Info
Used a cleaned version of student_mental_health.csv
→ Trained a model saved as mental_health_model.pkl

👨‍💻 Author
Kelvin Maina Irungu
🎓 Dedan Kimathi University of Technology
📧 irungu.maina22@students.dkut.ac.ke
📧 kelmaina4837@gmail.com
🔗 GitHub: kevobinkou

💡 Built with passion to promote mental health awareness among students and leverage technology for early intervention.