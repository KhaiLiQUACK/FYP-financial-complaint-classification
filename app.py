import streamlit as st
import shap
import numpy as np
import contractions
import re
import nltk
import pandas as pd
import random
import pickle
import ast
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords, wordnet, words
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download NLTK corpora
corpora_packages = ["stopwords", "words", "wordnet", "omw-1.4"]
for pkg in corpora_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# Download tagger
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

# Define wrapper class for XAI
class CNNBiLSTMWrapper:
    def __init__(self, model, tokenizer, maxlen=100):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __call__(self, texts):
        cleaned_texts = [clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        return self.model.predict(padded)

# Load best_model, label encoder, tokenizer & shap_explainer
model = load_model("CNN_BiLSTM_Seq.keras")
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# Define function to recreate text cleaning pipeline in EDA
stop_words = set(stopwords.words('english'))
negation_words = {"no", "not", "never", "none", "nobody", "nothing", "nowhere", "neither"}
custom_stopwords = stop_words - negation_words
lemmatizer = WordNetLemmatizer()
english_vocab = set(w.lower() for w in words.words())

# Repetition reducer
class RepeatReplacer:
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        # Check if the word is valid in WordNet
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        # If the word was changed, continue reducing
        if repl_word != word:
            return self.replace(repl_word)
        else:  # Return the reduced word
            return repl_word

# Initialize the replacer
replacer = RepeatReplacer()

# Helper function
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun if no match
        return wordnet.NOUN

def lemmatize_tokens(tokens):
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(token.lower(), get_wordnet_pos(tag)) for token, tag in pos_tags]

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in custom_stopwords
            and (not re.fullmatch(r'[a-zA-Z]{1,2}', word) or len(set(word)) > 1)]

def remove_non_english_words(tokens):
    return [word for word in tokens if word.lower() in english_vocab]

def clean_text(text):
    # 1. Remove multiple XXXX symbols (anonymized information) + Clean extra whitespace
    text = re.sub(r'X+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Text cleaning
    # Lowercase all words
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Remove URLs starting with http/https
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove URLs starting with www
    text = re.sub(r'www\.[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+', '', text)
    # Remove html tags with the content inside "<P/>"
    text = re.sub(r'<\s*\w+\s*/>', '', text)
    # Remove words with values
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    # Remove redundant punctuations
    text = re.sub(r'([.,!?;:()\-])\1+', r'\1', text)
    # Expand contractions again
    text = contractions.fix(text)
    # Remove all punctuation to facilitate shortening of repetitive character
    text = re.sub(r'(?<!\w)-|-(?!\w)|[^\w\s-]', ' ', text)
    # Shorten repetitive letters
    text = ' '.join([replacer.replace(word) for word in text.split()])
    # Remove redundant words
    text = re.sub(r'\b(\w+)( \1)+\b', r'\1', text)

    # 3. Tokenize using NLTK
    pattern = r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[.,!?;'\-\"()[]]"
    tokens = regexp_tokenize(text, pattern)

    # 4. POS tagging + lemmatization
    tokens = lemmatize_tokens(tokens)

    # 5. Stopword removal (keep negations) + repetitive character removal
    tokens = remove_stopwords(tokens)

    # 6. Filter out non-English words
    tokens = remove_non_english_words(tokens)

    return ' '.join(tokens)

# Define predict function for best model
def predict_cnn_bilstm(raw_text, model, tokenizer, label_encoder, maxlen=100):
    # Clean the raw complaint text
    cleaned = clean_text(raw_text)
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')
    # Predict
    pred_probs = model.predict(padded)
    pred_class = pred_probs.argmax(axis=1)
    # Decode label
    label = label_encoder.inverse_transform(pred_class)[0]
    return {
        "input_cleaned": cleaned,
        "predicted_label": label,
        "confidence": round(float(pred_probs.max()) * 100, 2)
    }

def predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    probs = model.predict(padded)
    pred_class = probs.argmax(axis=1)
    label = label_encoder.inverse_transform(pred_class)[0]
    confidence = round(float(probs.max()) * 100, 2)
    return cleaned, label, confidence, padded

def explain_shap(input_text, padded):
    # SHAP expects raw text if using Text masker, else feed preprocessed input
    shap_values = explainer(padded)
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# --- Streamlit App ---
st.set_page_config(page_title="Complaint Classifier", layout="centered")
st.title("📨 Financial Complaint Issue Classifier")
st.markdown("""Enter your complaint below to predict its category and explain why the model made the prediction using SHAP.""")

input_text = st.text_area("Enter Complaint Text:", height=200)

if st.button("Classify & Explain") and input_text.strip():
    cleaned, label, confidence, padded = predict(input_text)
    st.subheader("📌 Cleaned Input:")
    st.code(cleaned)

    st.subheader("📊 Prediction Result:")
    st.markdown(f"**Predicted Category:** `{label}`")
    st.markdown(f"**Confidence:** `{confidence}%`")

    st.subheader("🔍 SHAP Word Importance (Waterfall Plot):")
    explain_shap(input_text, padded)
else:
    st.markdown("\n🚀 Enter a complaint and click the button to see predictions.")
