import pandas as pd
import os

# Confirm working directory and files
print("📁 Current Directory:", os.getcwd())
print("📦 Files:", os.listdir())

# Load dataset
df = pd.read_csv("student_mental_health.csv")

print("\n🔍 Raw Data Sample:")
print(df.head())

# Drop timestamp if it exists
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# Extract year of study as number
df["Your current year of Study"] = df["Your current year of Study"].str.extract(r"(\d+)").astype(float)

# Convert CGPA to float
df["What is your CGPA?"] = pd.to_numeric(df["What is your CGPA?"], errors="coerce")

# Binary map
binary_map = {"Yes": 1, "No": 0, "Yes ": 1, "No ": 0}
for col in [
    "Do you have Depression?", "Do you have Anxiety?",
    "Do you have Panic attack?", "Did you seek any specialist for a treatment?"
]:
    df[col] = df[col].astype(str).str.strip().map(binary_map)

# Encode categorical features
df["Choose your gender"] = df["Choose your gender"].astype("category").cat.codes
df["Marital status"] = df["Marital status"].astype("category").cat.codes
df["What is your course?"] = df["What is your course?"].astype("category").cat.codes

# Print preview
print("\n📊 Cleaned Preview (before fixing missing data):")
print(df.head(10))

# Check missing
print("\n🔎 Missing values BEFORE filling:")
print(df.isnull().sum())

# Fix missing values smartly
df["Age"] = df["Age"].fillna(df["Age"].median())
df["What is your CGPA?"] = df["What is your CGPA?"].fillna(df["What is your CGPA?"].mean())
df["Your current year of Study"] = df["Your current year of Study"].fillna(2)

# Final check
print("\n🧪 Missing values AFTER filling:")
print(df.isnull().sum())

# Drop any remaining incomplete records (precaution)
df = df.dropna()

print("\n✅ Final Cleaned Data:")
print(df.head())
print("\n🧮 Total Rows After Cleaning:", len(df))
