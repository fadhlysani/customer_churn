import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load Resources
def load_resources():
    with open('best_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

try:
    model, scaler, encoders = load_resources()
except FileNotFoundError:
    st.error("Resource files (model, scaler, or encoders) not found. Please ensure they are exported correctly.")
    st.stop()

# 2. App Interface
st.title("Customer Churn Prediction App")
st.markdown("""
Input customer details below to predict the likelihood of them leaving the service.
""")

# Sidebar Inputs
st.sidebar.header("Customer Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure (Months)", 1, 60, 12)
usage_freq = st.sidebar.slider("Usage Frequency (Monthly)", 1, 30, 15)
support_calls = st.sidebar.slider("Support Calls (Last 6 Months)", 0, 10, 5)
payment_delay = st.sidebar.slider("Payment Delay (Days)", 0, 30, 5)
sub_type = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

# 3. Data Preparation
input_data = pd.DataFrame({
    'Gender': [gender],
    'Tenure': [tenure],
    'Usage Frequency': [usage_freq],
    'Support Calls': [support_calls],
    'Payment Delay': [payment_delay],
    'Subscription Type': [sub_type],
    'Contract Length': [contract]
})

# Preprocessing
input_processed = input_data.copy()
for col in ['Gender', 'Subscription Type', 'Contract Length']:
    le = encoders[col]
    input_processed[col] = le.transform(input_processed[col])

# Scaling
input_scaled = scaler.transform(input_processed)

# 4. Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1][0]

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"Target Status: CHURN")
    else:
        st.success(f"Target Status: NOT CHURN")
    
    st.write(f"**Churn Probability:** {probability:.2%}")
    
    # Display input summary
    st.info("User input data processed successfully.")
