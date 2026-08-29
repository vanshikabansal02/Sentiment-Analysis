import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# ---------- Page config ----------
st.set_page_config(page_title="Amazon Review Sentiment Analyzer", page_icon="🛒")

# ---------- Ensure NLTK stopwords are available ----------
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

stp_words = load_stopwords()

# ---------- Load model + vectorizer (cached so it only loads once) ----------
@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_artifacts()

# ---------- Text cleaning: lowercase + strip punctuation + remove stopwords ----------
def clean_text(review):
    review = review.lower()
    review = "".join(ch for ch in review if ch not in string.punctuation)
    cleanreview = " ".join(word for word in review.split() if word not in stp_words)
    return cleanreview

# ---------- UI ----------
st.title("🛒 Amazon Review Sentiment Analyzer")
st.write("Enter a product review below and the model will predict whether the sentiment is positive or negative.")

review = st.text_area("Review text", height=150, placeholder="e.g. This product exceeded my expectations, great quality...")

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]  # 0 = negative/neutral (rating <=3), 1 = positive (rating >3)

        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100

        st.subheader("Result")
        if prediction == 1:
            st.success("✅ Positive sentiment")
        else:
            st.error("❌ Negative sentiment (rating 3 or below)")

        st.write(f"Confidence: **{confidence:.1f}%**")

st.divider()
st.caption("Model: TF-IDF (max_features=2500) + Logistic Regression | Built with Streamlit")
