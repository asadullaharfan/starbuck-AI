import streamlit as st
import joblib
import pandas as pd

st.title("Starbucks Calories Predictor")
st.write("Nutrition values enter karein, phir Predict dabayein.")

@st.cache_resource
def load_model():
    return joblib.load("starbucks_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

# Kaggle ke feature names bilkul waise hi, spaces samet
FEATURES = [
    "Trans Fat (g) ",
    "Saturated Fat (g)",
    " Sodium (mg)",
    " Total Carbohydrates (g) ",
    "Cholesterol (mg)",
    " Dietary Fibre (g)",
    " Sugars (g)",
    " Protein (g) "
]

with st.form("calorie_form"):
    col1, col2 = st.columns(2)

    with col1:
        trans_fat = st.number_input("Trans Fat (g)", min_value=0.0, value=0.0, step=0.1)
        saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, value=0.0, step=0.1)
        sodium = st.number_input("Sodium (mg)", min_value=0.0, value=0.0, step=1.0)
        carbohydrates = st.number_input("Total Carbohydrates (g)", min_value=0.0, value=0.0, step=1.0)

    with col2:
        cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0, value=0.0, step=1.0)
        fibre = st.number_input("Dietary Fibre (g)", min_value=0.0, value=0.0, step=0.1)
        sugars = st.number_input("Sugars (g)", min_value=0.0, value=0.0, step=1.0)
        protein = st.number_input("Protein (g)", min_value=0.0, value=0.0, step=0.1)

    predict_button = st.form_submit_button("Predict Calories")

if predict_button:
    values = [
        trans_fat,
        saturated_fat,
        sodium,
        carbohydrates,
        cholesterol,
        fibre,
        sugars,
        protein
    ]

    input_data = pd.DataFrame([values], columns=FEATURES)

    # Model ke trained columns ka order use karein
    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        expected_features = list(expected_features)

        if set(expected_features) != set(FEATURES):
            st.error(f"Model ke expected feature names: {expected_features}")
            st.stop()

        input_data = input_data.reindex(columns=expected_features)

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Calories: {float(prediction):.1f} kcal")
    except Exception as e:
        st.error(f"Prediction nahi ho saki: {e}")
