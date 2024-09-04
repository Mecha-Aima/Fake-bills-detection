"""
Evaluate the accuracy of four machine learning models on the banknotes dataset.

The models are a Perceptron, an SVM, a Naive Bayes classifier, and a K-Nearest Neighbors
classifier. The accuracy of each model is printed as a percentage.
"""

import pandas as pd
import numpy as np
import os

# Import the four machine learning models
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load the banknotes dataset
file_path = os.path.join(os.path.dirname(__file__), "banknotes.csv")
data: pd.DataFrame = pd.read_csv(file_path)

# Split the dataset into features (X) and labels (y)
y: np.ndarray = np.array(data['class'])
data = data.drop('class', axis=1)
X: np.ndarray = data.values

# Split the dataset into training and testing sets
holdout = 0.40 * len(X)
X_train: np.ndarray = X[:int(holdout)]
X_test: np.ndarray = X[int(holdout):]
y_train: np.ndarray = y[:int(holdout)]
y_test: np.ndarray = y[int(holdout):]

# Train the four machine learning models
model_perceptron = Perceptron()
model_svm = svm.SVC()
model_naive_bayes = GaussianNB()
model_knn = KNeighborsClassifier(n_neighbors=3)

model_perceptron.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_naive_bayes.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred_perceptron: np.ndarray = model_perceptron.predict(X_test)
y_pred_svm: np.ndarray = model_svm.predict(X_test)
y_pred_naive_bayes: np.ndarray = model_naive_bayes.predict(X_test)
y_pred_knn: np.ndarray = model_knn.predict(X_test)

# Evaluate the accuracy of each model
accuracy_perceptron: float = np.mean(y_pred_perceptron == y_test)
accuracy_svm: float = np.mean(y_pred_svm == y_test)
accuracy_naive_bayes: float = np.mean(y_pred_naive_bayes == y_test)
accuracy_knn: float = np.mean(y_pred_knn == y_test)

# Print the accuracy of each model
print(f"Perceptron accuracy: {accuracy_perceptron*100:.2f} %")
print(f"SVM accuracy: {accuracy_svm*100:.2f} %")
print(f"Naive Bayes accuracy: {accuracy_naive_bayes*100:.2f} %")
print(f"KNN accuracy: {accuracy_knn*100:.2f} %")

