
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PATHS AND DATA LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    df = load_data()

except FileNotFoundError:
    st.error("Dataset not found. Check your data/diabetes.csv file.")
    st.stop()


# Prepare data for analysis
analysis_data = df.copy()

analysis_data[INVALID_ZERO_COLUMNS] = (
    analysis_data[INVALID_ZERO_COLUMNS].replace(0, np.nan)
)


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("Diabetes Data Analysis")

st.markdown(
    """
    Explore the Pima Indians Diabetes dataset through
    interactive visualizations and statistical summaries.
    """
)

st.caption("Dataset: Pima Indians Diabetes Database")


# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------

total_records = len(analysis_data)

diabetes_count = int(
    (analysis_data["Outcome"] == 1).sum()
)

non_diabetes_count = int(
    (analysis_data["Outcome"] == 0).sum()
)

diabetes_percentage = (
    diabetes_count / total_records * 100
    if total_records else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Diabetes Outcome", f"{diabetes_count:,}")
col3.metric("Non-Diabetes Outcome", f"{non_diabetes_count:,}")
col4.metric("Positive Outcome", f"{diabetes_percentage:.1f}%")


st.divider()


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("Dataset Preview")

st.write(
    "A preview of the records used in this machine learning project."
)

st.dataframe(
    analysis_data.head(10),
    use_container_width=True
)


# --------------------------------------------------
# DESCRIPTIVE STATISTICS
# --------------------------------------------------

st.subheader("Descriptive Statistics")

st.write(
    "Statistical summaries of the numerical features in the dataset."
)

st.dataframe(
    analysis_data.describe().round(2),
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# OUTCOME DISTRIBUTION
# --------------------------------------------------

st.subheader("Diabetes Outcome Distribution")

outcome_counts = (
    analysis_data["Outcome"]
    .value_counts()
    .sort_index()
    .rename({
        0: "No Diabetes Outcome",
        1: "Diabetes Outcome"
    })
    .reset_index()
)

outcome_counts.columns = ["Outcome", "Count"]

fig_outcome = px.pie(
    outcome_counts,
    names="Outcome",
    values="Count",
    hole=0.45,
    title="Distribution of Diabetes Outcomes",
    color="Outcome",
    color_discrete_map={
        "No Diabetes Outcome": "#D6A84F",
        "Diabetes Outcome": "#9B4D3A"
    }
)

fig_outcome.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

st.plotly_chart(
    fig_outcome,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# FEATURE DISTRIBUTION
# --------------------------------------------------

st.subheader("Feature Distribution")

available_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

selected_feature = st.selectbox(
    "Select a feature to explore",
    available_features
)

fig_distribution = px.histogram(
    analysis_data,
    x=selected_feature,
    color="Outcome",
    marginal="box",
    barmode="overlay",
    opacity=0.7,
    color_discrete_map={
        0: "#D6A84F",
        1: "#9B4D3A"
    },
    title=f"{selected_feature} Distribution by Outcome",
    labels={
        "Outcome": "Diabetes Outcome"
    }
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# BMI COMPARISON
# --------------------------------------------------

st.subheader("BMI Comparison")

fig_bmi = px.box(
    analysis_data,
    x="Outcome",
    y="BMI",
    color="Outcome",
    color_discrete_map={
        0: "#D6A84F",
        1: "#9B4D3A"
    },
    labels={
        "Outcome": "Diabetes Outcome",
        "BMI": "Body Mass Index"
    },
    title="BMI Distribution Across Outcome Groups"
)

st.plotly_chart(
    fig_bmi,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader("Feature Correlation")

correlation = analysis_data.corr(numeric_only=True)

fig_correlation = px.imshow(
    correlation,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Correlation Between Dataset Features"
)

st.plotly_chart(
    fig_correlation,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# MISSING VALUES ANALYSIS
# --------------------------------------------------

st.subheader("Missing Values Analysis")

missing_values = (
    analysis_data.isnull()
    .sum()
    .reset_index()
)

missing_values.columns = [
    "Feature",
    "Missing Values"
]

missing_values = missing_values[
    missing_values["Missing Values"] > 0
]

if not missing_values.empty:

    fig_missing = px.bar(
        missing_values,
        x="Feature",
        y="Missing Values",
        text="Missing Values",
        title="Missing Values by Feature"
    )

    st.plotly_chart(
        fig_missing,
        use_container_width=True
    )

    st.dataframe(
        missing_values,
        use_container_width=True,
        hide_index=True
    )

else:
    st.success("No missing values found.")


# --------------------------------------------------
# DATA PREPARATION NOTE
# --------------------------------------------------

st.info(
    """
    **Data preparation note**

    Zero values in Glucose, BloodPressure, SkinThickness,
    Insulin and BMI are treated as missing measurements.

    The trained machine learning pipeline uses median
    imputation to handle missing values during prediction.
    """
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Diabetes Analytics | Educational Machine Learning Project"
)
