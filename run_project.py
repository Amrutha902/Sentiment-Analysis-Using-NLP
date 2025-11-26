from src.preprocessing import load_and_preprocess
from src.model import train_model, evaluate_model
from src.predict import predict

if __name__ == "__main__":
    # Load and preprocess data (expects raw text 'Review' column)
    X_train, X_test, y_train, y_test = load_and_preprocess('data/reviews.csv')

    # Train the model and evaluate on test set
    model, vectorizer = train_model(X_train, y_train)
    evaluate_model(model, vectorizer, X_test, y_test)

    # Example prediction on new text
    predict("This is the best movie ever!")

