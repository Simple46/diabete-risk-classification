
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# PATHS AND MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Trained model not found. Please ensure "
        "models/diabetes_model.joblib exists."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("Patient Diabetes Assessment")

st.markdown(
    """
    Enter the patient's information and health measurements below.
    The trained Logistic Regression model will use these values
    to generate a prediction.
    """
)

st.warning(
    """
    This application is for educational and research purposes only.
    It is not a medical diagnostic tool. Predictions should not replace
    professional medical advice, laboratory testing or clinical assessment.
    """
)


# --------------------------------------------------
# PATIENT INPUT FORM
# --------------------------------------------------

st.subheader("Patient Information")

with st.form("diabetes_prediction_form"):

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient's full name"
    )

    st.divider()

    st.subheader("Health Measurements")

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            step=1.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )

    with col2:

        insulin = st.number_input(
            "Insulin Level",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            step=1.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    st.divider()

    submitted = st.form_submit_button(
        "Generate Prediction",
        use_container_width=True,
        type="primary"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if not patient_name.strip():

        st.error("Please enter the patient's name.")

    else:

        # Prepare patient measurements
        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=FEATURE_COLUMNS
        )

        # Apply the same zero handling used during training
        input_data[INVALID_ZERO_COLUMNS] = (
            input_data[INVALID_ZERO_COLUMNS].replace(0, np.nan)
        )

        try:

            # Generate prediction
            prediction = model.predict(input_data)[0]

            # Get model-estimated class probabilities
            probabilities = model.predict_proba(input_data)[0]

            # Locate the probabilities by their actual class labels
            class_probabilities = dict(
                zip(model.classes_, probabilities)
            )

            non_diabetes_probability = (
                class_probabilities.get(0, 0) * 100
            )

            diabetes_probability = (
                class_probabilities.get(1, 0) * 100
            )

            st.divider()

            st.subheader("Prediction Result")

            st.write(
                f"Patient: **{patient_name.strip()}**"
            )

            if prediction == 1:

                st.error(
                    f"""
                    Based on the model's prediction,
                    **{patient_name.strip()} is likely to have diabetes.**
                    """
                )

            else:

                st.success(
                    f"""
                    Based on the model's prediction,
                    **{patient_name.strip()} is not classified as
                    likely to have diabetes.**
                    """
                )

            # Probability metrics
            st.subheader("Model Prediction Probabilities")

            col1, col2 = st.columns(2)

            col1.metric(
                "Diabetes Outcome",
                f"{diabetes_probability:.2f}%"
            )

            col2.metric(
                "Non-Diabetes Outcome",
                f"{non_diabetes_probability:.2f}%"
            )

            # Probability chart
            probability_data = pd.DataFrame({
                "Outcome": [
                    "No Diabetes Outcome",
                    "Diabetes Outcome"
                ],
                "Probability": [
                    non_diabetes_probability,
                    diabetes_probability
                ]
            })

            fig = px.bar(
                probability_data,
                x="Outcome",
                y="Probability",
                text="Probability",
                color="Outcome",
                color_discrete_map={
                    "No Diabetes Outcome": "#D6A84F",
                    "Diabetes Outcome": "#9B4D3A"
                },
                title="Model-Estimated Class Probabilities"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Model-Estimated Probability (%)",
                xaxis_title="",
                yaxis_range=[0, 110],
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Missing measurement warning
            missing_inputs = input_data.columns[
                input_data.isnull().any()
            ].tolist()

            if missing_inputs:

                st.warning(
                    "The following zero-valued measurements were "
                    "treated as missing and handled by the model's "
                    f"median imputer: {', '.join(missing_inputs)}."
                )

            st.info(
                """
                **Important:** These percentages are estimates produced
                by the machine learning model. They are not confirmed
                clinical probabilities or a medical diagnosis.
                A healthcare professional should assess the patient
                using appropriate clinical tests.
                """
            )

        except Exception as error:

            st.error(
                f"An error occurred while generating the prediction: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Analytics | Logistic Regression | "
    "Educational machine learning project"
)
