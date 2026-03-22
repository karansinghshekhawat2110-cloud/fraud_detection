####    🔍 Credit Card Fraud Detection       ####

A machine learning system that detects fraudulent credit card transactions in real-time with explainability.

## 🚀 Live Demo
👉 [Try the app here](https://frauddetection-qmcwefcwx3yp8n4eos8dxm.streamlit.app/)

## 📊 Project Overview
Built a fraud detection system on 284,807 real credit card transactions where only 0.17% were fraudulent. The core challenge was handling this severe class imbalance while maintaining high recall to catch as many frauds as possible.

## 🧠 What I Built
- **Exploratory Data Analysis** — discovered key fraud patterns and top features
- **Data Preprocessing** — normalized features, handled class imbalance using SMOTE
- **Logistic Regression** — baseline model
- **Neural Network** — deep learning model using TensorFlow/Keras
- **SHAP Explainability** — explains WHY each transaction was flagged
- **Streamlit App** — interactive UI deployed live

## 📈 Results
| Model | Recall | Precision | AUC-ROC |
|-------|--------|-----------|---------|
| Logistic Regression | 91.8% | 5.8% | - |
| Neural Network | 86% | 72% | 0.93 |

## 🔑 Key Findings
- Fraud transactions average €122 vs €88 for normal transactions
- Top fraud indicators: V14, V10, V12, V4, V11
- No single feature separates fraud — model combines all 30 features simultaneously
- SMOTE improved fraud detection by balancing 492 fraud cases against 284,315 normal ones

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, TensorFlow/Keras
- SHAP for explainability
- Streamlit for deployment

## 📁 Project Structure
```
fraud_detection/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
├── model/
│   ├── fraud_model.h5
│   └── lr_model.pkl
├── app.py
└── requirements.txt
```

## 📦 Dataset
Download from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🚀 Run Locally
```bash
git clone https://github.com/karansinghshekhawat2110-cloud/fraud_detection.git
cd fraud_detection
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Author
Karan Singh Shekhawat — AIT Pune