# DevelopersHub.co-Internship-Tasks
This repository contains the internship tasks 1-6 for DevelopersHub.co internship 2025

# Developer Hub - Period 1 Report

This repository contains summaries and insights from six data science and machine learning projects completed during Period 1 of the Developer Hub curriculum.

---

## 📊 Task 1: Iris Dataset Exploration

**Objective**: Understand data trends and distributions in the Iris dataset.

**Key Steps**:
- Loaded using `seaborn.load_dataset('iris')`
- Explored shape, data types, and summary statistics
- Visualized data with:
  - **Pairplots**: Show clear species separation (especially Setosa)
  - **Histograms & Boxplots**: Petal dimensions are more discriminative than sepal ones

---

## 📈 Task 2: Stock Price Prediction

**Objective**: Predict next-day stock closing price using historical data.

**Details**:
- **Stock**: Apple Inc. (AAPL)
- **Time Frame**: Jan 2020 – Dec 2024
- **Features Used**: Open, High, Low, Volume
- **Target**: Next day’s Close price

**Models**:
- **Linear Regression**: RMSE = \$3.11
- **Random Forest**: RMSE = \$25.72

Linear Regression showed better generalization on test data.

---

## ❤️ Task 3: Heart Disease Prediction

**Objective**: Classify whether a patient has heart disease using clinical data.

**Dataset**: UCI Heart Disease (920 records)

**Preprocessing**:
- Dropped columns with >50% missing values
- Imputed others (median/mode)
- One-hot encoded categorical features

**Model**: Logistic Regression  
**Performance**:
- Accuracy: 82.06%
- ROC AUC: 0.8987

Important features included chest pain type and patient gender.

---

## 🩺 Task 4: General Health Query Chatbot (LLM-based)

**Goal**: Build a command-line health chatbot using a Large Language Model.

**Model**: `Mistral-7B-Instruct-v0.1` via Hugging Face

**Key Features**:
- Prompt engineering for friendly tone
- Safety filters for critical queries
- Real-time response generation

**Requirements**
- pip install transformers accelerate
- Generate a token for hugging face and go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 and give your contact details to access this LLM.

**Limitations**:
- Not for medical diagnosis
- Needs high compute (≥12GB VRAM recommended)

---

## 🧠 Task 5: Mental Health Support Chatbot

**Objective**: Train a lightweight empathetic chatbot using DistilGPT2.

**Dataset**: EmpatheticDialogues  
**Model**: DistilGPT2  
**Interface**: Streamlit app

**Requirements**
- !pip install transformers datasets accelerate streamlit
- !pip install -U datasets
- !pip install fsspec==2023.9.2

**Features**:
- Fine-tuned on ~80,000 conversation samples
- Uses top-k sampling for varied, empathetic responses

**Deployment**: Run locally or deploy to Hugging Face, Heroku, etc.

---

## 🏠 Task 6: House Price Prediction

**Goal**: Predict house prices based on features like area, bedrooms, etc.

**Dataset**: Kaggle Housing.csv (545 records)

**Models**:
- Linear Regression
- Gradient Boosting Regressor

**Evaluation**:
- **Linear Regression**: MAE ₹970K, RMSE ₹1.32M
- **Gradient Boosting**: MAE ₹959K, RMSE ₹1.29M

Gradient Boosting performed more consistently, especially at high price ranges.

---

## 📌 Conclusion

Each project in this report demonstrates a complete machine learning pipeline—from data preprocessing and modeling to evaluation and insights. The skills covered span regression, classification, NLP, and LLM-based systems, providing a strong foundation for applied data science.


