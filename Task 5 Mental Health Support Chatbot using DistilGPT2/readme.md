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
