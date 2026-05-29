import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("Breast Cancer Prediction System")

st.write("Enter patient information below:")

radius = st.number_input("Radius Mean")
texture = st.number_input("Texture Mean")
perimeter = st.number_input("Perimeter Mean")

if st.button("Predict"):

    features = np.array([[radius, texture, perimeter]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("Malignant Tumor Detected")
    else:
        st.success("Benign Tumor")