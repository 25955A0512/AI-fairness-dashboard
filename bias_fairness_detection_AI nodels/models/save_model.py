# save_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load cleaned data
data = pd.read_csv("data/adult_cleaned.csv")
X = data.drop("income", axis=1)
y = data["income"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "models/logistic_model.pkl")
print("✅ Model saved to models/logistic_model.pkl")
