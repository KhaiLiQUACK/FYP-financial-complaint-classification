
import streamlit as st
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessing tools
model = load_model('/content/drive/MyDrive/FYP/Models/CNN_BiLSTM_Seq.h5')
with open('/content/drive/MyDrive/FYP/Features/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('/content/drive/MyDrive/FYP/Features/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('/content/drive/MyDrive/FYP/Features/background_texts.pkl', 'rb') as f:
    background_texts = pickle.load(f)

# Define your clean_text function here (assumed already written)
# from text_cleaning_module import clean_text

# Wrapper for SHAP
class CNNBiLSTMWrapper:
    def __init__(self, model, tokenizer, maxlen=100):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __call__(self, texts):
        cleaned_texts = [clean_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        return self.model.predict(padded)

# Load SHAP explainer
shap_model = CNNBiLSTMWrapper(model, tokenizer)
explainer = shap.Explainer(shap_model, background_texts)

# Prediction and explanation
def predict_and_explain(raw_text):
    cleaned = clean_text(raw_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    pred_probs = model.predict(padded)
    label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
    confidence = float(np.max(pred_probs))

    shap_values = explainer([raw_text])
    return label, confidence, shap_values[0]

# Streamlit app UI
st.title("📝 Financial Complaint Classifier with SHAP Explanation")
text_input = st.text_area("Enter a consumer complaint:", height=200)

if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence, shap_val = predict_and_explain(text_input)
        st.success(f"**Predicted Issue:** {label} ({confidence*100:.2f}%)")

        st.markdown("### Explanation (SHAP Highlighted Text)")
        fig = shap.plots.text(shap_val, display=False)
        st.pyplot(fig)
