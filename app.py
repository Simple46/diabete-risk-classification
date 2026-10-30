
import streamlit as st

st.set_page_config(
    page_title="Diabetes Analytics",
    page_icon="📊",
    layout="wide"
)

pages = {
    "Diabetes Analytics": [
        st.Page(
            "pages/1_Data_Analysis.py",
            title="Data Analysis",
            icon="📊",
            default=True
        ),
        st.Page(
            "pages/2_AI_Prediction.py",
            title="AI Prediction",
            icon="🧠"
        )
    ]
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
