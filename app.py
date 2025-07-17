import streamlit as st
import joblib

# Load the model and TF-IDF vectorizer
model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")

# Streamlit app interface
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection")
st.markdown("Check if a news article is **Real** or **Fake** using a Machine Learning model.")

# User input
user_input = st.text_area("Enter the news content here:", height=200)

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter some news text.")
    else:
        # Preprocess and predict
        vectorized_input = tfidf.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        # Output result
        if prediction == 1:
            st.success("✅ This news is likely **Real**.")
        else:
            st.error("🚨 This news is likely **Fake**.")


