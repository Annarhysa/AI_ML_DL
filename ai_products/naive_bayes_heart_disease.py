#import all the necessary libraries and the dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset (replace 'dataset.csv' with your dataset file)
dataset = pd.read_csv('Datasets/heart.csv')
print(dataset.head())

# Assuming the dataset has features and a target variable
X = dataset.drop('target', axis=1)  # Features
y = dataset['target']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier
classifier = GaussianNB()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Analyze performance
# 1. Cross-validation score
cv_score = cross_val_score(classifier, X, y, cv=5)
print("Cross-validation Score: ", cv_score.mean())

# 2. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 3. Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

#4. Accuracy of the model
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the model is: ", accuracy*100)