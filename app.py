import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_pretrained_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device

def generate_translation(german_text, model, tokenizer, max_new_tokens=50, num_beams=4):
    prompt = f"Translate the following German text to French:\n\n{german_text}\n\nFrench:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "French:" in generated_text:
        translation = generated_text.split("French:")[-1].strip()
    else:
        translation = generated_text.strip()
    return translation

st.title("German to French Translation Interface")
st.write("This interface queries our best performing model for translation.")

with st.spinner("Loading model..."):
    base_model_name = "facebook/opt-1.3b"
    model_A, tokenizer, device = load_pretrained_model(base_model_name)
    # Load the fine-tuned best model (Model D) from the saved adapter files.
    model_D = PeftModel.from_pretrained(model_A, "./modelD")
    model_D = model_D.to(device)
st.success("Model loaded successfully!")

input_text = st.text_area("Enter German text to translate:")

if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter some German text.")
    else:
        with st.spinner("Translating..."):
            translation = generate_translation(input_text, model_D, tokenizer)
        st.markdown("**French Translation:**")
        st.write(translation)
