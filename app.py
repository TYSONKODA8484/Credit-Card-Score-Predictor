import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the models
lgbm_model = joblib.load('lightgbm_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Function to predict credit score movement
def predict_credit_score_movement(model, input_data):
    prediction = model.predict([input_data])  
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Credit Score Prediction", layout="wide", page_icon="🔮")

# Header Section
st.title("🔮 Credit Score Prediction Tool")
st.markdown("""
    This tool predicts whether a customer's credit score will **increase**, **decrease**, or **remain stable** based on the details you provide.
    Fill out the details below, and click **Predict** to see the result.
""")

# Input Fields Section
with st.container():
    st.header("Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=30)
        monthly_income = st.number_input("Monthly Income (INR)", min_value=20000, max_value=300000, value=50000)
        monthly_emi_outflow = st.number_input("Monthly EMI Outflow (INR)", min_value=3000, max_value=60000, value=10000)
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.5)
        repayment_history_score = st.number_input("Repayment History Score (0-100)", min_value=0, max_value=100, value=80)
        recent_credit_card_usage = st.number_input("Recent Credit Card Usage (Last 3 Months)", min_value=0, max_value=1, value=0)

    with col2:
        current_outstanding = st.number_input("Current Outstanding (INR)", min_value=5000, max_value=150000, value=10000)
        num_open_loans = st.number_input("Number of Open Loans", min_value=0, max_value=10, value=1)
        dpd_last_3_months = st.number_input("Days Past Due (Last 3 Months)", min_value=0, max_value=90, value=0)
        num_hard_inquiries_last_6m = st.number_input("Number of Hard Inquiries (Last 6 Months)", min_value=0, max_value=10, value=1)
        recent_loan_disbursed_amount = st.number_input("Recent Loan Disbursed Amount (INR)", min_value=0, max_value=500000, value=100000)
        total_credit_limit = st.number_input("Total Credit Limit (INR)", min_value=5000, max_value=100000, value=50000)
        months_since_last_default = st.number_input("Months Since Last Default", min_value=0, max_value=24, value=6)

# Gender input field
gender = st.selectbox("Gender", ["Male", "Female"], index=0, key="gender", help="Select gender from the options")

# Create a feature vector from user input
input_data = [
    age, monthly_income, monthly_emi_outflow, current_outstanding, credit_utilization_ratio,
    num_open_loans, repayment_history_score, dpd_last_3_months, num_hard_inquiries_last_6m,
    recent_credit_card_usage, recent_loan_disbursed_amount, total_credit_limit, months_since_last_default
]

# Model Selection Dropdown
model_choice = st.selectbox("Select Model for Prediction", ["LightGBM", "XGBoost"], index=0)

# Map models to their respective variables
model_mapping = {
    "LightGBM": lgbm_model,
    "XGBoost": xgb_model
}

if st.button("Predict"):
    # Get the selected model
    selected_model = model_mapping[model_choice]

    # Get the prediction from the model
    prediction = predict_credit_score_movement(selected_model, input_data)

    # Map the prediction to a meaningful label
    if prediction == 0:
        prediction_label = "Decrease"
    elif prediction == 1:
        prediction_label = "Remain Stable"
    else:
        prediction_label = "Increase"

    # Results
    st.subheader(f"Prediction Result: {prediction_label}")
    st.write("""
        Based on the provided details, our model predicts the customer's credit score will **{}** in the next 3 months.
        
        Please note, this prediction is based on randomly generated data. 
        It's not a guarantee but a prediction based on patterns observed in the data.
    """.format(prediction_label))

    # Visualization 
    st.subheader("Prediction Confidence")
    model_confidence = selected_model.predict_proba([input_data])[0]
    classes = ["Decrease", "Remain Stable", "Increase"]

    # Confidence levels for each class
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(classes, model_confidence, color=["red", "yellow", "green"])
    ax.set_ylabel("Prediction Probability", fontsize=12)
    ax.set_title(f"Confidence of Prediction using {model_choice}", fontsize=14)
    st.pyplot(fig)

    # Adding a nice note with styling
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 10px 24px;
        }
        </style>
    """, unsafe_allow_html=True)
