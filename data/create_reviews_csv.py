import pandas as pd

data = {
    "Review": [
        "I love this product, it works great!",
        "This is the worst service I have ever had.",
        "Absolutely fantastic experience, highly recommend.",
        "I am very disappointed with the quality."
    ],
    "Sentiment": [
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)
df.to_csv('data/reviews.csv', index=False)
print("reviews.csv created successfully.")
