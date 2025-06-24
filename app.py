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
    nltk.download('averaged_perceptron_tagger_eng')

# Load best_model, label encoder & tokenizer
model = load_model("CNN_BiLSTM_Seq.keras")
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

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
    # Confidence
    confidence = round(float(pred_probs.max()) * 100, 2)
    return cleaned, label, confidence, pred_probs, int(pred_class[0])

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

# Create SHAP explainer
wrapped_model = CNNBiLSTMWrapper(model, tokenizer)
regex_masker = shap.maskers.Text(r"\w+")
explainer = shap.Explainer(wrapped_model, masker=regex_masker, output_names=label_encoder.classes_)

# SHAP explanation
def explain_shap(raw_text, predicted_class):
    shap_values = explainer([raw_text])
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0][:, predicted_class], show=False)
    st.pyplot(fig)

# --- Streamlit App ---
st.set_page_config(page_title="Complaint Classifier", layout="centered")
st.title("📨 Financial Complaint Issue Classifier")
st.markdown("""Enter your complaint below to predict its category and explain why the model made the prediction using SHAP.""")

# input_text = st.text_area("Enter Complaint Text:", height=200)

# if st.button("Classify & Explain") and input_text.strip():
#     cleaned, label, confidence, pred_probs, predicted_class = predict_cnn_bilstm(
#         input_text, model, tokenizer, label_encoder
#     )
#     st.subheader("📌 Cleaned Input:")
#     st.code(cleaned)

#     st.subheader("📊 Prediction Result:")
#     st.markdown(f"**Predicted Category:** `{label}`")
#     st.markdown(f"**Confidence:** `{confidence}%`")

#     st.subheader("🔍 SHAP Word Importance (Waterfall Plot):")
#     explain_shap(input_text, predicted_class)
# else:
#     st.markdown("\n🚀 Enter a complaint and click the button to see predictions.")


tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Explanation", "✨ Highlighting"])

with tab1:
    raw_text = st.text_area("Enter your financial complaint:")

    if st.button("Predict") and raw_text.strip():
        cleaned = clean_text(raw_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='post')

        pred_probs = model.predict(padded)
        pred_class_idx = np.argmax(pred_probs)
        predicted_class = label_encoder.inverse_transform([pred_class_idx])[0]

        st.success(f"Predicted Class: **{predicted_class}**")

        st.session_state["input"] = raw_text
        st.session_state["cleaned"] = cleaned
        st.session_state["padded"] = padded
        st.session_state["pred_probs"] = pred_probs
        st.session_state["pred_class_idx"] = pred_class_idx
        st.session_state["shap_values"] = explainer(padded)

with tab2:
    if "shap_values" in st.session_state:
        shap_values = st.session_state["shap_values"]
        pred_probs = st.session_state["pred_probs"]
        pred_class_idx = st.session_state["pred_class_idx"]

        col1, col2 = st.columns([2, 1])

        with col1:
            class_names = label_encoder.classes_
            selected_class = st.selectbox("Select class to explain:", options=class_names, index=pred_class_idx)
            selected_class_idx = list(class_names).index(selected_class)

            max_features = st.slider("Top N impactful features:", 5, len(shap_values[selected_class_idx].values), 20)
            st.markdown(f"### SHAP Waterfall Plot: **{selected_class}**")
            shap.plots.waterfall(shap_values[selected_class_idx], max_display=max_features)

        with col2:
            probs_df = pd.DataFrame({
                'Class': label_encoder.inverse_transform(np.arange(len(pred_probs[0]))),
                'Probability': pred_probs.flatten()
            }).sort_values("Probability", ascending=True)

            fig = px.bar(
                probs_df,
                x="Probability",
                y="Class",
                orientation='h',
                title="🎯 Class Probabilities",
                labels={"Probability": "Predicted Probability", "Class": "Class"},
                text=probs_df["Probability"].apply(lambda x: f"{x:.2%}"),
                color="Probability",
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis=dict(title=''), xaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "shap_values" in st.session_state:
        st.markdown("### ✨ Word-Level SHAP Impact")
        highlighted_html = generate_shap_highlight(
            st.session_state["cleaned"], 
            st.session_state["shap_values"][st.session_state["pred_class_idx"]], 
            tokenizer
        )
        st.markdown(highlighted_html, unsafe_allow_html=True)

        with st.expander("ℹ️ What do the colors mean?"):
            st.markdown("""
            - **Red**: increases the chance of the predicted class  
            - **Blue**: decreases the chance  
            - More intense = stronger impact
            """)
