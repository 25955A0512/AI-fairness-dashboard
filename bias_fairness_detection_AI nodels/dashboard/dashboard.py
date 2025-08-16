import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

# 📍 Paths
original_path = "../models/original_model.pkl"
mitigated_path = "../models/mitigated_model.pkl"
scaler_path = "../models/scaler.pkl"
data_path = "../data/cleaned_data.csv"

# 🧠 Load models
original_model = joblib.load(original_path)
mitigated_model = joblib.load(mitigated_path)

# 🔐 Load scaler safely
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("❌ Scaler file not found. Please check 'models/scaler.pkl'")
    st.stop()

# 📥 Load cleaned dataset
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error("❌ Cleaned dataset not found. Please check 'data/cleaned_data.csv'")
    st.stop()

# 🎯 Features and target
features = ['age', 'education-num', 'hours-per-week']
target = 'income'
sensitive_feature = 'sex'

X = df[features]
y = df[target]
X_scaled = scaler.transform(X)

# 🌐 Streamlit UI
st.title("🔍 Bias & Fairness Detection Dashboard")
st.markdown("Analyze fairness metrics across models using demographic parity and equalized odds.")

# 🎛️ Model selector
model_choice = st.radio("Choose a model to evaluate:", ["Original", "Mitigated"])
model = original_model if model_choice == "Original" else mitigated_model

# 🔮 Predictions
y_pred = model.predict(X_scaled)

# 📊 Fairness metrics
metric_frame = MetricFrame(
    metrics={
        "Selection Rate": selection_rate,
        "Accuracy": accuracy_score
    },
    y_true=y,
    y_pred=y_pred,
    sensitive_features=df[sensitive_feature]
)

# 📈 Plot selection rates
fig = px.bar(
    metric_frame.by_group,
    x=metric_frame.by_group.index,
    y="Selection Rate",
    title=f"Selection Rate by {sensitive_feature} ({model_choice} Model)",
    labels={"index": sensitive_feature, "Selection Rate": "Rate"},
    color="Selection Rate"
)
st.plotly_chart(fig)

# 📋 Show metrics
st.subheader("📊 Fairness Metrics")
st.write("**Demographic Parity Difference:**", demographic_parity_difference(y, y_pred, sensitive_features=df[sensitive_feature]))
st.write("**Equalized Odds Difference:**", equalized_odds_difference(y, y_pred, sensitive_features=df[sensitive_feature]))

# 📌 Show full metric frame
with st.expander("Show full metric breakdown"):
    st.dataframe(metric_frame.by_group)

