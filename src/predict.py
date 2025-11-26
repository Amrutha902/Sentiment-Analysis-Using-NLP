import joblib

def predict(text):
    vectorizer = joblib.load('tfidf.pkl')
    model = joblib.load('sentiment_model.pkl')
    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)[0]
    print(f"Predicted Sentiment: {pred}")
    return pred

# Example usage:
if __name__ == "__main__":
    predict("I love this place!")
    predict("This is the worst movie ever")
