import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()

X = data.data
y = data.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)
model = LogisticRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)

accuracy = model.score(X_train , y_train)

print("Accuracy =",accuracy*100,"%")