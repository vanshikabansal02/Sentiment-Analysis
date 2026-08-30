# Amazon Review Sentiment Analyzer

A Streamlit web app that predicts whether an Amazon product review is positive or negative using a TF-IDF + Logistic Regression model trained on labeled review data.

🔗 **Live App:** [Click here to try it](https://sentiment-analysis-jyvo8ezl8bynrpdlihz4tu.streamlit.app/)

## Features
- Enter any product review text and get an instant sentiment prediction
- Shows model confidence score
- Preprocessing: lowercasing, punctuation removal, stopword removal

## Screenshots

**Positive Review**
![Positive review prediction](screenshots/positive_review.png)

**Negative Review**
![Negative review prediction](screenshots/negative_review.png)

## Tech Stack
- Python, scikit-learn (TF-IDF, Logistic Regression)
- NLTK (stopword removal)
- Streamlit (web app)

## Run locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Dataset
Amazon product reviews with star ratings, converted to binary sentiment (positive: rating > 3, negative: rating ≤ 3).
