
import streamlit as st
import joblib
import pandas as pd

st.title("Starbucks Calories Predictor")
st.write("Predict calories using your trained model.")

@st.cache_resource
def load_model():
    return joblib.load("starbucks_model.pkl")

try:
    model = load_model()

    st.success("Model loaded successfully!")

    st.write("Model features:", getattr(model, "n_features_in_", "Unknown"))

    st.info(
        "This model needs the same input features used during training. "
        "Check your Kaggle notebook for the feature column names."
    )

except Exception as e:
    st.error(f"Could not load model: {e}")
