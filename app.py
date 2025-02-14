import streamlit as st
import numpy as np
import pickle


# Load trained model (Ensure you have 'model.pkl' in the same directory)
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit UI
st.title("Diabetes Prediction")
st.write("Enter the following details to predict the likelihood of Diabetes.")

# Input fields with general information
Pregnancies = st.number_input(
    "Number of Pregnancies", min_value=0, max_value=17,
    help="Number of times a woman has been pregnant. Higher values may indicate a higher risk."
)
Glucose = st.number_input(
    "Glucose Level", min_value=0,
    help="Plasma glucose concentration (mg/dL). High values (>140 mg/dL) may indicate diabetes risk."
)
BloodPressure = st.number_input(
    "Blood Pressure", min_value=0,
    help="Diastolic blood pressure (mm Hg). Normal values are around 80 mm Hg. Higher values indicate risk."
)
SkinThickness = st.number_input(
    "Skin Thickness", min_value=0,
    help="Triceps skin fold thickness (mm). Used to estimate body fat percentage."
)
Insulin = st.number_input(
    "Insulin Level", min_value=0,
    help="2-Hour serum insulin (µU/mL). High values may indicate insulin resistance."
)
BMI = st.number_input(
    "Body Mass Index (BMI)", min_value=0.0, format="%.2f",
    help="Body Mass Index (weight in kg / height in m²). A value >25 is considered overweight."
)
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function (DPF)", 
    min_value=0.08, max_value=2.5, step=0.01, format="%.2f",
    help="DPF represents genetic influence on diabetes. Typical values: \n\n"
         "- No family history: 0.08 - 0.3\n"
         "- One close relative: 0.3 - 0.6\n"
         "- Multiple family members: 0.6 - 1.2\n"
         "- Strong family history: 1.2 - 2.5"
)
Age = st.number_input(
    "Age", min_value=0,
    help="Age in years. Older individuals have a higher risk of diabetes."
)

# Predict button
if st.button("Predict Diabetes"):
    # Convert input into NumPy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Get prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.error("High risk of Diabetes! Consult a doctor.")
    else:
        st.success("Low risk of Diabetes. Keep maintaining a healthy lifestyle!")
