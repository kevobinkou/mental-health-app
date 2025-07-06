import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Show current directory and files
print("📁 Current Directory:", os.getcwd())
print("📦 Files:", os.listdir())

# Load dataset
df = pd.read_csv("student_mental_health.csv")

# Drop Timestamp if present
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# Extract numeric year from "Your current year of Study"
df["Your current year of Study"] = df["Your current year of Study"].str.extract(r"(\d+)")
df["Your current year of Study"] = pd.to_numeric(df["Your current year of Study"], errors="coerce")

# Convert CGPA to numeric
df["What is your CGPA?"] = pd.to_numeric(df["What is your CGPA?"], errors="coerce")

# Binary columns encoding (Yes/No to 1/0)
binary_cols = [
    "Do you have Depression?", 
    "Do you have Anxiety?", 
    "Do you have Panic attack?", 
    "Did you seek any specialist for a treatment?"
]
for col in binary_cols:
    df[col] = df[col].str.strip().map({"Yes": 1, "No": 0})

# Encode categorical variables
df["Choose your gender"] = df["Choose your gender"].astype("category").cat.codes
df["Marital status"] = df["Marital status"].astype("category").cat.codes
df["What is your course?"] = df["What is your course?"].astype("category").cat.codes

# Show cleaned preview
print("\n📊 Cleaned Preview:")
print(df.head())

# Show where missing data exists
print("\n🧪 Checking for missing values column-wise:")
print(df.isnull().sum())

# Fill missing CGPA and year with the median value
df["What is your CGPA?"] = df["What is your CGPA?"].fillna(df["What is your CGPA?"].median())
df["Your current year of Study"] = df["Your current year of Study"].fillna(df["Your current year of Study"].median())


# Optional: drop rows with missing target
df = df.dropna(subset=["Do you have Depression?"])

print("\n✅ Final Cleaned Data (after handling missing values):")
print(df.head())
print("🧮 Total Rows Used for Training:", len(df))

# Features and target
X = df.drop("Do you have Depression?", axis=1)
y = df["Do you have Depression?"]

# Ensure there's enough data to train
if len(df) < 5:
    print("🚫 Not enough data to train a model. Please check your dataset.")
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "mental_health_model.pkl")
    print("✅ Model trained and saved as 'mental_health_model.pkl'")
