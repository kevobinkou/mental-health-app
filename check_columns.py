import pandas as pd

df = pd.read_csv("student_mental_health.csv")
print("📋 Columns in your CSV:")
print(df.columns.tolist())
