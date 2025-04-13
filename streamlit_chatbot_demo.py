import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Set Streamlit page configuration
st.set_page_config(
    page_title="Emotion-Aware NLP Demo",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_id = "noor1718/final-emotion-chatbot-t5"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Generate chatbot response
def generate_response(emotion, text):
    input_text = f"respond emotionally: {emotion}: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- UI Layout ---
st.title("Emotion-Aware Text Summarization Using Deep Learning")
st.markdown("""
**Course:** Deep Learning CSC 7760  
**Group 7** – Chiranjeev Ravichandran, Gurnoor Singh Sandhu, Hari Kumaran Venkatachalam Kasiraman Jayasekaran
""")

st.markdown("---")
st.header("🎭 Emotion-Conditioned Chatbot")
st.markdown("Provide an emotion and a message. The model will generate an emotionally aware reply.")

# User inputs
emotion = st.selectbox("Choose an emotion", [
    "sadness", "joy", "anger", "love", "fear", "gratitude", "neutral"
])
user_input = st.text_area("Enter your message", "I feel completely alone.")

# Generate button
if st.button("Generate Response"):
    with st.spinner("Thinking emotionally..."):
        response = generate_response(emotion, user_input)
        st.success("Response generated!")
        st.write("**Chatbot Response:**")
        st.info(response)

# Footer
st.markdown("---")
st.caption("© 2025 Group 7 – Deep Learning Project at Wayne State University")
