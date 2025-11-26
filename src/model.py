from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_tfidf, y_train)
    joblib.dump(vectorizer, 'tfidf.pkl')
    joblib.dump(model, 'sentiment_model.pkl')
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.2f}")
