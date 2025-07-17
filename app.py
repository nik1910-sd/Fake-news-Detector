import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Title
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Check whether a news article is <b>Fake</b> or <b>Real</b>.</p>", unsafe_allow_html=True)

# Input box
st.markdown("---")
user_input = st.text_area("✏️ Enter the news headline below:", height=200)

# Button and prediction
if st.button("🔍 Check"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        vectorized_input = tfidf.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("✅ This news is likely **Real**.")
        else:
            st.error("🚨 This news is likely **Fake**.")





