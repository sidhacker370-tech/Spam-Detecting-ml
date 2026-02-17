import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Spam Detection App", page_icon="📩")

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATA_FILE = "dataset.csv"

# ------------------------------
# Train Model (Only if Needed)
# ------------------------------
@st.cache_resource
def train_model():
    if not os.path.exists(DATA_FILE):
        st.error("Dataset file not found. Please upload dataset.csv")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Validate dataset structure
    if "target" not in df.columns or "text" not in df.columns:
        st.error("Dataset must contain 'target' and 'text' columns.")
        st.stop()

    # Convert labels safely
    df["label"] = df["target"].map({"ham": 0, "spam": 1})

    if df["label"].isnull().any():
        st.error("Unexpected label values found in dataset.")
        st.stop()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer


# ------------------------------
# Load or Train
# ------------------------------
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
else:
    model, vectorizer = train_model()


# ------------------------------
# UI
# ------------------------------
st.title("📩 Spam Detection App")
st.write("Enter a message to check if it is Spam or Not Spam.")

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM.")
            st.write(f"Confidence: {probability[1]*100:.2f}%")
        else:
            st.success("✅ This message is NOT Spam.")
            st.write(f"Confidence: {probability[0]*100:.2f}%")
