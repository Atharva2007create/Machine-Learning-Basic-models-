import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'text': [
        'Hello, how are you today?',
        'WINNER! Claim your free gift card now by calling 69-69-420-420',
        'Are we still meeting for lunch at 1pm?',
        'URGENT: Your account has been compromised, call immediately',
        'Just checking in to see if you got my email',
        'CONGRATULATIONS! You have won a luxury cruise, dial now',
        'Hello son how are you',
        'Would like to know todays horoscope ?',
        'Pre book the new maruti suzuki Desire today !',
        'This is Neha call me now',
        'Namaste, are we still going to the Ganpati Pandal today at 6pm?',
        'CONGRATULATIONS! You have won 25 Lakhs in KBC Lottery. Call 9876543210 to claim via WhatsApp now.',
        'Beta, please pick up some milk and bread from the Kirana store on your way home.',
        'URGENT: Your SBI YONO account will be blocked today. Please update your KYC by clicking here: bit.ly/fake-link',
        'Sir, I am calling from your society gate. Please give entry for the Swiggy delivery boy.',
        'Happy Diwali to you and your family! Let us meet for dinner this weekend.',
        'Get a personal loan up to 10 Lakhs with zero processing fee. Reply YES to apply now.',
        'Can you send me the notes for the CE2 division lecture from MMCOE?',
        'FREE GIFT! Your mobile number has been selected for a free iPhone 15. Click to claim your reward.',
        'Know live score'
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print(f"Accuracy: {accuracy_score(y_test, predictions)}")

sample_call_1 = ["Click on this to know the live score"]
sample_vec = vectorizer.transform(sample_call_1)
result = model.predict(sample_vec)

print(f"Result: {'Spam 🔴' if result[0] == 1 else 'Not Spam✅'}")

sample_call_2 = ["What about coffee at 2 pm ?"]
sample_vec_2 = vectorizer.transform(sample_call_2)
result = model.predict(sample_vec_2)

print(f"Result: {'Spam 🔴' if result[0] == 1 else 'Not Spam✅'}")