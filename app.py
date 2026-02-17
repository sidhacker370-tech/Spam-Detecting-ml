import streamlit as st
import pandas as pd
import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

st.set_page_config(page_title="Spam Detection App", page_icon="📩")

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "dataset.csv"

# ------------------------------
# Custom Feature Engineering
# ------------------------------
def extra_features(text_series):
    url_feature = text_series.apply(lambda x: 1 if re.search(r'http|www|\.com|\.net', x.lower()) else 0)
    length_feature = text_series.apply(len)
    urgent_words = text_series.apply(lambda x: 1 if "urgent" in x.lower() or "verify" in x.lower() else 0)

    return pd.DataFrame({
        "url": url_feature,
        "length": length_feature,
        "urgent_flag": urgent_words
    })

# ------------------------------
# Train Model
# ------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    df["label"] = df["target"].map({"ham": 0, "spam": 1})

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X_text = vectorizer.fit_transform(df["text"])

    X_extra = extra_features(df["text"])
    scaler = StandardScaler()
    X_extra_scaled = scaler.fit_transform(X_extra)

    X_combined = hstack([X_text, X_extra_scaled])

    model = LogisticRegression(max_iter=2000)
    model.fit(X_combined, df["label"])

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return model, vectorizer, scaler

# ------------------------------
# Load or Train
# ------------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    scaler = joblib.load(SCALER_FILE)
else:
    model, vectorizer, scaler = train_model()

# ------------------------------
# UI
# ------------------------------
st.title("📩 Advanced Spam Detection")
st.write("Now with phishing-aware detection.")

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        text_vec = vectorizer.transform([user_input])
        extra = extra_features(pd.Series([user_input]))
        extra_scaled = scaler.transform(extra)

        final_input = hstack([text_vec, extra_scaled])

        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0]

        if prediction == 1:
            st.error("🚨 SPAM DETECTED")
            st.write(f"Confidence: {probability[1]*100:.2f}%")
        else:
            st.success("✅ SAFE MESSAGE")
            st.write(f"Confidence: {probability[0]*100:.2f}%")
