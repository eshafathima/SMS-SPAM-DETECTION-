import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    y = df["label"].map({"ham": 0, "spam": 1})
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english")
    X = tfidf.fit_transform(df["text"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return tfidf, model

tfidf, model = train_model()

st.title("SMS Spam Classifier")
message = st.text_area("Paste a message:")

if st.button("Check"):
    if message.strip():
        vec = tfidf.transform([message])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]
        if pred == 1:
            st.error(f"SPAM (confidence {prob:.0%})")
        else:
            st.success(f"Not spam (spam probability {prob:.0%})")
    else:
        st.warning("Please type a message first.")