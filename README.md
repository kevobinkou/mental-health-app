 🧠 Student Mental Health Prediction App

A machine learning-powered Streamlit app that predicts whether a student may need mental health support based on survey inputs such as CGPA, anxiety levels, and year of study.

 🚀 Demo
[👉 Live App on Streamlit ](https://github.com/kevobinkou/mental-health-app.git)

 📦 Features
- Predicts likelihood of mental health issues in students
- User-friendly interface built with Streamlit
- Uses Random Forest model trained on real student data

 🧠 Technologies Used
- Python
- scikit-learn
- pandas
- numpy
- joblib
- Streamlit

📁 Project Structure
mental-health-app/
├── mental_health_app.py # Streamlit web app
├── train_and_save_model.py # Model training script
├── student_mental_health.csv # Input dataset
├── mental_health_model.pkl # Saved trained model
└── requirements.txt # Python dependencies

## 🛠 Setup Instructions

1. Clone this repo  
```bash
git clone https://github.com/kevobinkou/mental-health-app.git
cd mental-health-app
pip install -r requirements.txt
streamlit run mental_health_app.py
✍️ Author
Kelvin Maina - @kevobinkou
