import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
from geminill import (
    agent_router, agent_financial_advice, agent_credit_score_explanation,
    agent_financial_health_qna, agent_credit_score_improvement, agent_explain_financial_terms
)

# --- Load ML Models ---
try:
    lgbm_model = joblib.load('lightgbm_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
except FileNotFoundError:
    st.error("Model files not found. Ensure 'lightgbm_model.pkl' and 'xgboost_model.pkl' are in the directory.")
    st.stop()

# --- Session State Initialization ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(datetime.now())
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None

# --- ML Prediction Function ---
def predict_credit_score_movement(model, input_data):
    try:
        prediction = model.predict([input_data])[0]
        confidence = model.predict_proba([input_data])[0]
        return prediction, confidence
    except Exception as e:
        return None, f"Prediction error: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Score Prediction & Financial Advisor", layout="wide", page_icon="🔮")
st.title("🔮 Credit Score Prediction & Financial Advisor")
st.markdown(""" 
    Predict your credit score movement and get personalized financial advice from our AI assistant. 
    Enter your details below and interact with the advisor for insights on finance, credit, and more.
""")

# Tabs for Prediction and Chatbot
tab1, tab2 = st.tabs(["📊 Credit Score Prediction", "💬 Financial Advisor Chat"])

# --- Tab 1: Credit Score Prediction ---
with tab1:
    st.header("Enter Your Financial Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=30, key="age")
        monthly_income = st.number_input("Monthly Income (INR)", min_value=20000, max_value=300000, value=50000, key="income")
        monthly_emi_outflow = st.number_input("Monthly EMI Outflow (INR)", min_value=3000, max_value=60000, value=10000, key="emi")
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="utilization")
        repayment_history_score = st.number_input("Repayment History Score (0-100)", min_value=0, max_value=100, value=80, key="repayment")
        recent_credit_card_usage = st.number_input("Recent Credit Card Usage (Last 3 Months)", min_value=0, max_value=1, value=0, key="cc_usage")

    with col2:
        current_outstanding = st.number_input("Current Outstanding (INR)", min_value=5000, max_value=150000, value=10000, key="outstanding")
        num_open_loans = st.number_input("Number of Open Loans", min_value=0, max_value=10, value=1, key="loans")
        dpd_last_3_months = st.number_input("Days Past Due (Last 3 Months)", min_value=0, max_value=90, value=0, key="dpd")
        num_hard_inquiries_last_6m = st.number_input("Number of Hard Inquiries (Last 6 Months)", min_value=0, max_value=10, value=1, key="inquiries")
        recent_loan_disbursed_amount = st.number_input("Recent Loan Disbursed Amount (INR)", min_value=0, max_value=500000, value=100000, key="loan_amount")
        total_credit_limit = st.number_input("Total Credit Limit (INR)", min_value=5000, max_value=100000, value=50000, key="credit_limit")
        months_since_last_default = st.number_input("Months Since Last Default", min_value=0, max_value=24, value=6, key="default")

    gender = st.selectbox("Gender", ["Male", "Female"], index=0, key="gender")
    model_choice = st.selectbox("Select Model for Prediction", ["LightGBM", "XGBoost"], index=0, key="model_choice")
    model_mapping = {"LightGBM": lgbm_model, "XGBoost": xgb_model}

    # Store user data as dictionary for Gemini
    user_data = {
        "Age": age,
        "Monthly Income (INR)": monthly_income,
        "Monthly EMI Outflow (INR)": monthly_emi_outflow,
        "Credit Utilization Ratio": credit_utilization_ratio,
        "Repayment History Score": repayment_history_score,
        "Recent Credit Card Usage": recent_credit_card_usage,
        "Current Outstanding (INR)": current_outstanding,
        "Number of Open Loans": num_open_loans,
        "Days Past Due (Last 3 Months)": dpd_last_3_months,
        "Number of Hard Inquiries (Last 6 Months)": num_hard_inquiries_last_6m,
        "Recent Loan Disbursed Amount (INR)": recent_loan_disbursed_amount,
        "Total Credit Limit (INR)": total_credit_limit,
        "Months Since Last Default": months_since_last_default,
        "Gender": gender
    }
    st.session_state.user_data = user_data

    if st.button("Predict", key="predict_button"):
        input_data = [
            age, monthly_income, monthly_emi_outflow, current_outstanding, credit_utilization_ratio,
            num_open_loans, repayment_history_score, dpd_last_3_months, num_hard_inquiries_last_6m,
            recent_credit_card_usage, recent_loan_disbursed_amount, total_credit_limit, months_since_last_default
        ]
        selected_model = model_mapping[model_choice]
        prediction, confidence = predict_credit_score_movement(selected_model, input_data)

        if isinstance(confidence, str):  # Error case
            st.error(confidence)
        else:
            # Map prediction to label
            prediction_label = {0: "Decrease", 1: "Remain Stable", 2: "Increase"}.get(prediction, "Unknown")
            st.session_state.prediction_label = prediction_label

            # Display results
            st.subheader(f"Prediction Result: {prediction_label}")
            st.write(f"Your credit score is predicted to **{prediction_label}** in the next 3 months.")

            # Confidence visualization using Streamlit bar chart
            st.subheader("Prediction Confidence")
            confidence_data = {
                "Decrease": confidence[0],
                "Remain Stable": confidence[1],
                "Increase": confidence[2]
            }
            st.bar_chart(confidence_data)
