# models/train_models.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# 📥 Load cleaned dataset
df = pd.read_csv("data/adult_cleaned.csv")

# 🎯 Define features, labels, and sensitive attribute
X = df.drop(columns=["income", "sex_original"])
y = df["income"]
sensitive_feature = df["sex_original"]

# 📊 Scale features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train original logistic regression model
original_model = LogisticRegression(max_iter=2000)
original_model.fit(X_scaled, y)
joblib.dump(original_model, "models/logistic_model.pkl")
print("✅ Original model saved.")

# ✅ Train fairness-mitigated model using ExponentiatedGradient
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=2000),
    constraints=DemographicParity()
)
mitigator.fit(X_scaled, y, sensitive_features=sensitive_feature)
joblib.dump(mitigator, "models/mitigated_model.pkl")
print("✅ Mitigated model saved.")

# 💾 Save the scaler for consistent preprocessing during inference
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Scaler saved.")
