import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Fraud Detection System", 
                   page_icon="🔍",
                   layout="centered")

# Title
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it's fraudulent.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/fraud_model.h5')
    return model

model = load_model()

# Input fields
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (€)", min_value=0.0, value=100.0)
    v14 = st.number_input("V14", value=0.0, format="%.4f")
    v10 = st.number_input("V10", value=0.0, format="%.4f")
    v12 = st.number_input("V12", value=0.0, format="%.4f")
    v17 = st.number_input("V17", value=0.0, format="%.4f")

with col2:
    time = st.number_input("Time (seconds)", min_value=0.0, value=0.0)
    v4 = st.number_input("V4", value=0.0, format="%.4f")
    v11 = st.number_input("V11", value=0.0, format="%.4f")
    v16 = st.number_input("V16", value=0.0, format="%.4f")
    v7 = st.number_input("V7", value=0.0, format="%.4f")

# Predict button
if st.button("🔍 Check Transaction"):
    
    # Create input array with 30 features
    input_data = np.zeros((1, 30))
    
    # Normalize amount and time
    scaler = StandardScaler()
    input_data[0, 28] = (amount - 88.29) / 250  # normalized amount
    input_data[0, 29] = (time - 94813) / 47488   # normalized time
    
    # Fill top features
    feature_map = {9: v10, 11: v12, 13: v14, 15: v16, 16: v17,
                   3: v4, 6: v7, 10: v11}
    for idx, val in feature_map.items():
        input_data[0, idx] = val
    
    # Predict
    prediction = model.predict(input_data)[0][0]
    
    # Show result
    st.subheader("Result:")
    if prediction > 0.5:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED!")
        st.error(f"Fraud Probability: {prediction*100:.2f}%")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION")
        st.success(f"Fraud Probability: {prediction*100:.2f}%")
    
    # Probability bar
    st.subheader("Fraud Probability:")
    st.progress(float(prediction))

# Footer
st.markdown("---")
st.markdown("Built with TensorFlow + Streamlit | Credit Card Fraud Detection Project")