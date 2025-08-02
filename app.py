import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from pathlib import Path


project_root = Path(__file__).parent  
model_path = project_root / 'stacking_model.pkl'
with open(model_path, 'rb') as f:
    model = joblib.load(f)
tfidf_path = project_root / 'tfidf_vectorizer.pkl'
with open(tfidf_path, 'rb') as f:
    tfidf = joblib.load(f)


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)


def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    sentiment_label = {
        0: "😠",
        1: "😊",
        2: "😐"
    }
    text_label = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }
    return text_label[prediction], sentiment_label[prediction]

st.set_page_config(page_title="Vaccine Sentiment Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center; color: #1DA1F2;'>💬 Pfizer Vaccine Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Analyze the sentiment of a tweet related to Pfizer's COVID-19 vaccine 💉</p>", unsafe_allow_html=True)


st.markdown("### ✍️ Enter a tweet below:")
user_input = st.text_area(
    label="",
    placeholder="Type your tweet here...",
    height=150
)

predict_button = st.button("🔍 Predict")


if predict_button:
    if user_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        label_text, label_emoji = predict_sentiment(user_input)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Predicted Sentiment:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: white;'>{label_text} {label_emoji}</h2>", unsafe_allow_html=True)
