import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "text": [
        "Win money now!!!",
        "Hello how are you",
        "Free gift offer click now",
        "Let's meet tomorrow",
        "Congratulations you won lottery",
        "Call me later",
        "Urgent! claim your prize",
        "Good morning friend"
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)


X = df["text"]
y = df["label"]


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


model = MultinomialNB()
model.fit(X_vectorized, y)

print("Model trained successfully!")


while True:
    msg = input("\nEnter a message (or type 'exit' to quit): ")
    
    if msg.lower() == "exit":
        break
    
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)
    
    if prediction[0] == "spam":
        print("Result: Spam Message")
    else:
        print("Result: Not Spam")