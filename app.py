import streamlit as st
import pandas as pd
import joblib

# App settings
st.set_page_config(page_title="Employee Turnover Predictor", layout="centered")
st.title("Employee Turnover Prediction App")
st.markdown("Fill in the employee's details to predict if they are likely to leave the company.")

# Model selection
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Choose Prediction Model", ["Random Forest", "Decision Tree"])

# Load the selected model
model_filename = "rf_model.pkl" if model_choice == "Random Forest" else "dt_model.pkl"
st.sidebar.write(f"Current model file: `{model_filename}`")  # للتأكد من إنه بيتغير

try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Please make sure it's in the same directory.")
    st.stop()

# Employee Input Section
st.header("Employee Information")

col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    time_spend_company = st.number_input("Years at Company", min_value=0, max_value=20, value=3)
    work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
    promotion_last_5years = st.selectbox("Got Promotion in Last 5 Years?", ["No", "Yes"])

with col2:
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=0, max_value=400, value=160)
    department = st.selectbox("Department", [
        "sales", "technical", "support", "IT", "hr", "accounting",
        "management", "marketing", "product_mng", "RandD"
    ])
    salary = st.selectbox("Salary Level", ["low", "medium", "high"])

# Prepare input data
work_accident_val = 1 if work_accident == "Yes" else 0
promotion_val = 1 if promotion_last_5years == "Yes" else 0

input_data = pd.DataFrame({
    "satisfaction_level": [satisfaction_level],
    "last_evaluation": [last_evaluation],
    "number_project": [number_project],
    "average_montly_hours": [average_montly_hours],
    "time_spend_company": [time_spend_company],
    "Work_accident": [work_accident_val],
    "promotion_last_5years": [promotion_val],
    "department": [department],
    "salary": [salary]
})

# One-hot encoding for categorical variables
input_data = pd.get_dummies(input_data, columns=["department", "salary"], drop_first=True)

# Match training feature columns
training_columns = model.feature_names_in_
missing_cols = set(training_columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[training_columns]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"This employee is likely to leave the company (probability: {prediction_proba[0]*100:.2f}%)")
    else:
        st.success(f"This employee is likely to stay (probability: {(1 - prediction_proba[0])*100:.2f}%)")