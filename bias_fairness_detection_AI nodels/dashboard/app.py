import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
data = pd.read_csv("data/adult_cleaned.csv")

# Use preserved raw sensitive feature
sensitive_feature = data["sex_original"]
target = data["income"]

# Calculate selection rates
selection_rates = target.groupby(sensitive_feature).mean()

# Display dashboard
st.title("Bias & Fairness Detection lax Dashboard")
st.write("### Selection Rates by Sex")
st.write(selection_rates)

# Plot selection rates
fig, ax = plt.subplots()
selection_rates.plot(kind="bar", ax=ax, color=["skyblue", "salmon"])
ax.set_ylabel("Selection Rate (Income > 50K)")
ax.set_xlabel("Sex")
ax.set_title("Income Selection Rate by Sex")
st.pyplot(fig)
