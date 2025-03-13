import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        torch_dtype=torch.float16,
        device_map="auto"
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter from GitHub
    model = PeftModel.from_pretrained(
        base_model,
        "Samarth7Hathwar/german-french-translation-app",
        subfolder="modelD",
        device_map="auto"
    )
    return model, tokenizer

model, tokenizer = load_model()

st.title("German ↔ French Translator")
text = st.text_area("Enter German text:", height=150)

if st.button("Translate"):
    if text.strip():
        with st.spinner("Translating..."):
            prompt = f"Translate German to French:\n\n{text}\n\nFrench:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50, num_beams=4)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).split("French:")[-1].strip()
        st.success(translation)
    else:
        st.warning("Please enter German text")
