import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Empathetic Chatbot", layout="centered")
st.title("🧘 Mental Health Support Chatbot")
st.markdown("Hi there 👋 Feel free to share your thoughts. I'm here to listen.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./empathetic-distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("./empathetic-distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    st.session_state.messages.append(("You", user_input))

    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    bot_response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.messages.append(("Bot", bot_response))

for speaker, message in st.session_state.messages:
    st.markdown(f"**{speaker}**: {message}")
