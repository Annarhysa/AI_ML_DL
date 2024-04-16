import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

covid_data = pd.read_csv('Datasets/covid_data.csv')

avg_covid = covid_data['TotalCases'].mean()

covid_prone = []
# Iterate over each row in the DataFrame
for index, row in covid_data.iterrows():
    if row['TotalCases'] > avg_covid:
      covid_prone.append(1)
    else:
      covid_prone.append(0)

covid_data['CovidProne'] = covid_prone

covid_data = covid_data.dropna(subset = ['TotalCases','ActiveCases'])

# Preprocess data and engineer features
# Example: Extract features like location, date, number of cases, population density, etc.

# Define features and target variable
X = covid_data[['TotalCases','ActiveCases']]  # Features
y = covid_data['CovidProne']  # Target variable (binary: 1 if prone, 0 otherwise)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train RandomForestClassifier (you can choose a different model as per your preference)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Find the top 5 maximum values in the 'Population' column
top_5_max = covid_data.nlargest(5, 'TotalCases')

# Extract country names and corresponding population values
countries = top_5_max['Country/Region']
populations = top_5_max['TotalCases']

#import matplotlib.pyplot as plt
# Plot the top values against their country names
#plt.bar(countries, populations)
#plt.xlabel('Country')
#plt.ylabel('TotalCases')
#plt.title('Top 5 Countries by Covid Cases')
#plt.show()