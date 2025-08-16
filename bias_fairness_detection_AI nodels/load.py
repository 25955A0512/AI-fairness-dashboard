import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)

# Assign column names
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Replace missing values marked as '?'
df = df.replace(" ?", pd.NA).dropna()

# Encode target
df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# Preserve raw sensitive feature for fairness analysis
df["sex_original"] = df["sex"]

# One-hot encode categorical features (excluding preserved column)
df_encoded = pd.get_dummies(df.drop(columns=["sex_original"]), drop_first=True)

# Add preserved column back
df_encoded["sex_original"] = df["sex_original"]

# Save cleaned data
df_encoded.to_csv("data/adult_cleaned.csv", index=False)
print("✅ Cleaned data saved to data/adult_cleaned.csv")

