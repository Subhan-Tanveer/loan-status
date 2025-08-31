import numpy as np
import pandas as pd
import pickle
import streamlit as st

# --- Configuration and Model Loading ---

# Load the trained model
# IMPORTANT: Ensure 'trained_model.sav' is in the same directory as this Streamlit app,
# or provide the correct absolute path if it's located elsewhere.
try:
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Error: 'trained_model.sav' not found. Please ensure the model file is in the correct directory.")
    st.stop() # Stop the app if the model isn't found

# Define the expected order of columns for the model after one-hot encoding.
# This list MUST match the columns that your model was trained on after preprocessing.
# Based on common preprocessing (get_dummies with drop_first=True), these are the likely columns.
# If your training script used different encoding or dropped/added columns, adjust this list.
MODEL_FEATURES = [
    "Gender", "Married",	"Dependents",	"Education",	"Self_Employed",	"ApplicantIncome",	"CoapplicantIncome",	"LoanAmount",	"Loan_Amount_Term",	"Credit_History",	"Property_Area"
]


# --- Prediction Function ---

def predict_loan_status(input_data_dict):
    """
    Predicts the loan status (approved/not approved) based on input data.

    Args:
        input_data_dict (dict): A dictionary of user inputs for each feature.
                                Example: {'Gender': 'Male', 'Married': 'Yes', ...}

    Returns:
        str: A message indicating whether the loan will be approved or not.
    """
    # 1. Create a DataFrame from the input dictionary
    # Ensure the input is treated as a single row DataFrame
    input_df = pd.DataFrame([input_data_dict])

    # 2. Preprocessing Steps (MUST match training preprocessing)
    # Convert categorical variables using one-hot encoding (get_dummies)
    # drop_first=True is crucial if your model was trained with it to avoid multicollinearity.
    # We must ensure all expected dummy columns are present, even if they are zero for this specific input.

    # Apply get_dummies
    input_df_processed = pd.get_dummies(input_df, drop_first=True)

    # Reindex the DataFrame to ensure all model features are present and in the correct order.
    # Any features not present in input_df_processed will be filled with 0 (which is correct for dummy variables).
    # This step is critical to match the column order and presence from training.
    for col in MODEL_FEATURES:
        if col not in input_df_processed.columns:
            input_df_processed[col] = 0 # Add missing dummy columns with 0

    # Ensure all columns are in the exact order as `MODEL_FEATURES` and drop any extra columns
    input_final = input_df_processed[MODEL_FEATURES]

    try:
        # Perform prediction
        prediction = loaded_model.predict(input_final)

        # Interpret the prediction
        if prediction[0] == 0: # Assuming 0 maps to 'N' (Not Approved) based on common encoding
            return 'The loan will not be approved. ❌'
        else: # Assuming 1 maps to 'Y' (Approved)
            return 'The loan will be approved! ✅'
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Prediction failed."


# --- Streamlit App Layout ---

def main():
    st.set_page_config(page_title="Loan Status Predictor", page_icon="🏦", layout="centered")
    st.title('Loan Status Prediction App 🏦')
    st.markdown("---")
    st.write("Enter the applicant's details below to predict their loan approval status.")

    st.header("Applicant Details")

    # Input fields for parameters with corrected mappings based on typical get_dummies output
    # Gender
    gender_input = st.selectbox('Gender', ['Male', 'Female'])

    # Married
    married_input = st.selectbox('Married', ['Yes', 'No'])

    # Dependents
    dependents_input = st.selectbox('Number of Dependents', ['0', '1', '2', '3+'])

    # Education
    education_input = st.selectbox('Education', ['Graduate', 'Not Graduate'])

    # Self_Employed
    self_employed_input = st.selectbox('Self Employed', ['Yes', 'No'])

    # ApplicantIncome
    applicant_income = st.number_input('Applicant Income', min_value=0.0, value=5000.0, step=100.0)

    # CoapplicantIncome
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0.0, value=0.0, step=100.0)

    # LoanAmount
    # Note: LoanAmount might be in actual currency or thousands depending on training data.
    # If your model expects actual amount, ensure this input is consistent.
    loan_amount = st.number_input('Loan Amount', min_value=0.0, value=150000.0, step=1000.0)

    # Loan_Amount_Term
    loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=12, max_value=480, value=360)

    # Credit_History
    # Assuming 1.0 for Yes, 0.0 for No, and NaN for missing,
    # but since it's a selectbox, we map to 1.0 or 0.0 directly.
    credit_history_input = st.selectbox('Credit History (1.0 for Yes, 0.0 for No)', ['1.0 (Yes)', '0.0 (No)'])
    credit_history = float(credit_history_input.split(' ')[0]) # Extract 1.0 or 0.0

    # Property_Area
    property_area_input = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])

    # Create a button for prediction
    st.markdown("---")
    if st.button('Predict Loan Status', help="Click to get the loan approval prediction."):
        # Collect all inputs into a dictionary
        input_data = {
            'Gender': gender_input,
            'Married': married_input,
            'Dependents': dependents_input,
            'Education': education_input,
            'Self_Employed': self_employed_input,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area_input
        }

        # Call the prediction function with the raw input data
        result = predict_loan_status(input_data)
        st.success(result)
        st.balloons() # Visual feedback for a successful prediction

# Run the app
if __name__ == '__main__':
    main()
