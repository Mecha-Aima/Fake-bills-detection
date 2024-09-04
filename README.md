# Banknote Classification Model Comparison

## Overview
This project implements a simple classification model comparison using scikit-learn to classify banknotes as either "Authentic" or "Counterfeit" based on four features: variance, skewness, curtosis, and entropy. The code reads data from a CSV file, preprocesses the data, trains four different models (Perceptron, SVM, Gaussian Naive Bayes, and K-Nearest Neighbors), makes predictions on a test set, and prints out the accuracy of each model.

## Data
The code expects a CSV file with the following structure:

| Feature   | Description                         |
|-----------|-------------------------------------|
| variance  | Variance of the banknote image      |
| skewness  | Skewness of the banknote image      |
| curtosis  | Curtosis of the banknote image      |
| entropy   | Entropy of the banknote image       |
| class     | Class label (0 for Authentic, 1 for Counterfeit) |

## Code
The code is written in Python and uses the scikit-learn library for machine learning tasks. It consists of the following steps:

1. **Data loading**: The code loads the banknote data from a CSV file.
2. **Data preprocessing**: The code splits the data into training and testing sets.
3. **Model training**: The code trains four different classification models (Perceptron, SVM, Gaussian Naive Bayes, and K-Nearest Neighbors).
4. **Model evaluation**: The code makes predictions on the test set using each trained model and calculates the accuracy of each model.
5. **Results**: The code prints out the accuracy of each model as a percentage.

## Usage
To run the code, simply execute the Python script. The code will load the data, train the models, and print out the accuracy of each model.

## Acknowledgments
I completed this exercise as part of CS50's Intro to AI Course. The course provided a comprehensive introduction to artificial intelligence and machine learning, and this project was a great opportunity to apply the concepts learned in the course to a real-world problem.
