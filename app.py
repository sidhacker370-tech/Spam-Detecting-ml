
import streamlit as st
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="Spam Detective", page_icon="🕵️")

# 2. Load the Saved Model
@st.cache_resource
def load_model():
    if not os.path.exists('spam_model.pkl'):
        return None, None
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# 3. The User Interface
st.title("🕵️ SMS/Email Spam Classifier")
st.write("Enter a message below to check if it's **Spam** or **Safe (Ham)**.")

if model is None:
    st.error("Model not found! Please run 'train_model.py' first to generate the model files.")
else:
    # Text Input
    user_input = st.text_area("Message Content:", height=150, placeholder="Type your message here... e.g., 'You won a free lottery!'")

    if st.button("Analyze Message"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # 4. Prediction Logic
            vec_input = vectorizer.transform([user_input])

            # Predict
            prediction = model.predict(vec_input)[0]
            probability = model.predict_proba(vec_input)[0]

            # 5. Display Results
            st.subheader("Analysis Result:")

            if prediction == 1:
                st.error(f"🚨 **SPAM DETECTED**")
                st.write(f"Confidence: {probability[1]*100:.2f}%")
            else:
                st.success(f"✅ **SAFE MESSAGE (HAM)**")

                        st.write(f"Confidence: {probability[0]*100:.2f}%")
