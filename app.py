import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Fraud Detection System",
                   page_icon="🔍",
                   layout="centered")

st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it's fraudulent.")

@st.cache_resource
def load_model():
    with open('model/lr_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

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

if st.button("🔍 Check Transaction"):
    input_data = np.zeros((1, 30))
    input_data[0, 28] = (amount - 88.29) / 250
    input_data[0, 29] = (time - 94813) / 47488
    feature_map = {9: v10, 11: v12, 13: v14, 15: v16, 16: v17,
                   3: v4, 6: v7, 10: v11}
    for idx, val in feature_map.items():
        input_data[0, idx] = val

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result:")
    if prediction == 1:
        st.error(f"🚨 FRAUDULENT TRANSACTION DETECTED!")
        st.error(f"Fraud Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION")
        st.success(f"Fraud Probability: {probability*100:.2f}%")

    st.subheader("Fraud Probability:")
    st.progress(float(probability))

st.markdown("---")
st.markdown("Built with Scikit-learn + Streamlit | Credit Card Fraud Detection")