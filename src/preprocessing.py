import pandas as pd
import re
from sklearn.model_selection import train_test_split
import csv

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    return text.strip()

def load_and_preprocess(csv_path, text_col='Review', label_col='Sentiment'):
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL)
    
    # Insert this line here to drop rows with missing text or labels
    df = df.dropna(subset=[text_col, label_col])
    
    df[text_col] = df[text_col].astype(str).apply(clean_text)  # if using raw text

    X = df[text_col].values
    y = df[label_col].values

    return train_test_split(X, y, test_size=0.2, random_state=42)




