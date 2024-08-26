import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

model_path = "./model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

st.title("Text Classification with DistilBERT")

st.write("This project leverages the DistilBERT model from Hugging Face's Transformers library to perform sentiment analysis. The model has been fine-tuned on the IMDb dataset and is used to classify input text into two categories: Positive or Negative.")

input_text = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if input_text:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class = logits.argmax().item()
        
        # Mapping the predicted class to "Positive" or "Negative"
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        st.write(f"Predicted class: {sentiment}")
