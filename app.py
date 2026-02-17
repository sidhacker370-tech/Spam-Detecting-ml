import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam Detection App")
st.write("Enter a message to check if it is Spam or Not Spam.")

# -----------------------------
# Train Model If Not Exists
# -----------------------------
def train_and_save_model():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        sep="\t",
        header=None,
        names=["label", "message"],
    )

    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer


# -----------------------------
# Load or Train
# -----------------------------
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    model, vectorizer = train_and_save_model()
else:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# -----------------------------
# User Input
# -----------------------------
message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT Spam.")
