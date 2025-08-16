# dashboard.py
import os
import joblib
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score

# 📦 Load models and scaler


# Dynamically resolve the path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_model.pkl')
original_model = joblib.load(model_path)


# Build the correct path to the model file
mitigated_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mitigated_model.pkl')

# Load the model
mitigated_model = joblib.load(mitigated_path)

scaler = joblib.load("models/scaler.pkl")


# 📥 Load cleaned dataset for evaluation
df = pd.read_csv("data/adult_cleaned.csv")
X = df.drop(columns=["income", "sex_original"])
y = df["income"]
sensitive_feature = df["sex_original"]

# ⚙️ Scale features
X_scaled = scaler.transform(X)

# 🧠 Model selector
st.title("Bias & Fairness Detection Dashboard")
model_choice = st.radio("Choose model to evaluate:", ["Original", "Mitigated"])

if model_choice == "Original":
    model = original_model
    st.success("✅ Using Original Logistic Regression Model")
else:
    model = mitigated_model
    st.success("✅ Using Fairness-Mitigated Model")

# 📊 Make predictions
y_pred = model.predict(X_scaled)

# 🗂️ Tabbed layout
tab1, tab2 = st.tabs(["📈 Metrics", "📊 Charts"])

with tab1:
    # 🎯 Accuracy
    acc = accuracy_score(y, y_pred)
    st.metric("Model Accuracy", f"{acc:.3f}")

    # ⚖️ Fairness metrics
    selection_metrics = MetricFrame(
        metrics={"Selection Rate": selection_rate},
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    dp_diff = demographic_parity_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    st.subheader("Fairness Metrics by Group")
    st.write(selection_metrics.by_group)

    st.subheader("Overall Fairness")
    st.write("📉 Demographic Parity Difference:", round(dp_diff, 3))

    # 📥 Export Fairness Report
    if st.button("📤 Export Fairness Report as CSV"):
        report_df = pd.DataFrame({
            "Group": selection_metrics.by_group.index,
            "Selection Rate": selection_metrics.by_group.values
        })
        report_df["Model"] = model_choice
        report_df["Demographic Parity Difference"] = dp_diff

        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "fairness_report.csv", "text/csv")

with tab2:
    # 📊 Interactive Selection Rate Chart
    st.subheader("📊 Selection Rate by Group (Interactive)")
    selection_df = selection_metrics.by_group.reset_index()
    selection_df.columns = ["Group", "Selection Rate"]

    fig = px.bar(selection_df, x="Group", y="Selection Rate", color="Group",
                 title="Selection Rate by Sensitive Group", text="Selection Rate")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig)

    # 📉 Demographic Parity Comparison Chart
    st.subheader("📉 Compare Demographic Parity Difference")
    y_pred_orig = original_model.predict(X_scaled)
    y_pred_mitigated = mitigated_model.predict(X_scaled)

    dp_orig = demographic_parity_difference(y, y_pred_orig, sensitive_features=sensitive_feature)
    dp_mitigated = demographic_parity_difference(y, y_pred_mitigated, sensitive_features=sensitive_feature)

    dp_df = pd.DataFrame({
        "Model": ["Original", "Mitigated"],
        "Demographic Parity Difference": [dp_orig, dp_mitigated]
    })

    fig2 = px.bar(dp_df, x="Model", y="Demographic Parity Difference", color="Model",
                  title="Fairness Comparison", text="Demographic Parity Difference")
    fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig2)

# 📤 Optional: Upload new data for prediction
st.subheader("🔍 Try with Your Own Data")
uploaded_file = st.file_uploader("Upload a CSV file with same columns as training data")

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    try:
        user_scaled = scaler.transform(user_df)
        user_pred = model.predict(user_scaled)
        st.write("Predictions:", user_pred)
    except Exception as e:
        st.error(f"❌ Error processing uploaded data: {e}")
