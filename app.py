import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

if callable(expected_columns):
    try:
        expected_columns = expected_columns()
    except TypeError:
        expected_columns = []

if isinstance(expected_columns, pd.Index):
    expected_columns = expected_columns.tolist()
elif not isinstance(expected_columns, (list, tuple)):
    try:
        expected_columns = list(expected_columns)
    except Exception:
        expected_columns = []

st.title("Heart Stroke Prediction by a Logistic Regression Model")
st.markdown("Provide the following details to check your heart stroke risk:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'MaxHR': max_hr,
        'Sex_M': 1 if sex == 'M' else 0,
        'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
        'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
        'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Up': 1 if st_slope == 'Up' else 0
    }

    input_df = pd.DataFrame([raw_input])


    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])


    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error(" High Risk of Heart Disease")
    else:
        st.success(" Low Risk of Heart Disease")