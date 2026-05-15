# toxic_comment_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="⚠️",
    layout="wide"
)

# -----------------------------
# Load Saved Files
# -----------------------------
lemmatizer = joblib.load("lemmatizer.pkl")
stop_words = joblib.load("stopwords.pkl")
tokenizer = joblib.load("tokenizer.pkl")

model = load_model("toxic_comment_model.keras")

# -----------------------------
# Constants
# -----------------------------
MAX_LEN = 120

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = re.sub(r'[^a-zA-Z]', ' ', str(text))

    text = text.lower()

    text = word_tokenize(text)

    text = [
        lemmatizer.lemmatize(word)
        for word in text
        if word not in stop_words
    ]

    text = ' '.join(text)

    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_comment(text):

    cleaned = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post'
    )

    prediction = model.predict(padded, verbose=0)[0]

    return prediction

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Single Prediction",
        "Bulk Prediction",
        "Model Insights",
        "Sample Test Cases"
    ]
)

# =========================================================
# PAGE 1 : SINGLE PREDICTION
# =========================================================

if page == "Single Prediction":

    st.title("⚠️ Toxic Comment Classification")

    st.write(
        "Enter a comment below and the model will predict toxicity categories."
    )

    user_input = st.text_area(
        "Enter Comment",
        height=150
    )

    if st.button("Predict Toxicity"):

        if user_input.strip() == "":

            st.warning("Please enter a comment.")

        else:

            prediction = predict_comment(user_input)

            st.subheader("Prediction Results")

            result_df = pd.DataFrame({
                "Label": LABELS,
                "Probability": prediction
            })

            st.dataframe(result_df)

            # Bar Chart
            fig, ax = plt.subplots(figsize=(8,4))

            ax.bar(LABELS, prediction)

            plt.xticks(rotation=20)

            ax.set_ylabel("Probability")

            ax.set_title("Toxicity Scores")

            st.pyplot(fig)

# =========================================================
# PAGE 2 : BULK PREDICTION
# =========================================================

elif page == "Bulk Prediction":

    st.title("📂 Bulk CSV Prediction")

    st.write(
        "Upload a CSV file containing comments."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(df.head())

        column = st.selectbox(
            "Select Comment Column",
            df.columns
        )

        if st.button("Run Bulk Prediction"):

            predictions = []

            for text in df[column]:

                pred = predict_comment(text)

                predictions.append(pred)

            predictions = np.array(predictions)

            for i, label in enumerate(LABELS):

                df[label] = predictions[:, i]

            st.subheader("Prediction Results")

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "Download Predictions CSV",
                csv,
                "toxic_predictions.csv",
                "text/csv"
            )

# =========================================================
# PAGE 3 : MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    st.title("📊 Model Insights")

    st.subheader("Model Architecture")

    st.code(model.summary())

    st.subheader("Project Information")

    st.write("""
    ### Features
    - Text preprocessing using NLTK
    - Tokenization and padding
    - Bidirectional LSTM model
    - Multi-label classification
    - Real-time predictions
    - Bulk CSV predictions

    ### Labels
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate
    """)

    st.subheader("Model Parameters")

    st.write(f"Vocabulary Size: 5000")
    st.write(f"Maximum Sequence Length: {MAX_LEN}")

# =========================================================
# PAGE 4 : SAMPLE TEST CASES
# =========================================================

elif page == "Sample Test Cases":

    st.title("🧪 Sample Test Cases")

    sample_comments = [
        "I really appreciate your help.",
        "You are an idiot.",
        "I will kill you.",
        "Have a wonderful day.",
        "This is the worst thing ever."
    ]

    for comment in sample_comments:

        st.write("### Comment")

        st.info(comment)

        prediction = predict_comment(comment)

        result_df = pd.DataFrame({
            "Label": LABELS,
            "Probability": prediction
        })

        st.dataframe(result_df)