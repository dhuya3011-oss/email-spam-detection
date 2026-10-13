import os

import joblib
import streamlit as st

import src.preprocess  # needed so the saved model can be loaded

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "spam_model.joblib")

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("📧 Email Spam Detector")
st.write("Paste an email below and the model will tell you if it looks like spam.")

text = st.text_area(
    "Email text",
    height=200,
    placeholder="Subject: Congratulations! You have won a free prize...",
)

if st.button("Check email"):
    if not text.strip():
        st.warning("Please paste some email text first.")
    else:
        prob = model.predict_proba([text])[0][1]
        if prob >= 0.5:
            st.error(f"🚨 Spam ({prob:.1%} spam probability)")
        else:
            st.success(f"✅ Not spam ({prob:.1%} spam probability)")
        st.progress(float(prob))

st.caption("Model: TF-IDF + Multinomial Naive Bayes, about 99% accuracy on the test set.")
