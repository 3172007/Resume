import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("news.csv")

# Inputs and outputs
X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully!")

while True:
    news = input("Enter a news headline: ")

    news_vector = vectorizer.transform([news])

    result = model.predict(news_vector)

    if result[0] == 1:
        print("Real News")
    else:
        print("Fake News")

    again = input("Check another? y/n: ")

    if again.lower() != "y":
        break