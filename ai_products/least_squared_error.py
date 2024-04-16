from sklearn.datasets import load_diabetes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the linnerrud dataset
X,y = datasets.load_diabetes(return_X_y = True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate Least Squares Error (LSE)
train_lse = mean_squared_error(y_train, y_pred_train)
test_lse = mean_squared_error(y_test, y_pred_test)
print("Train LSE:", train_lse)
print("Test LSE:", test_lse)

# Assess generalization ability
# Compare the errors on training and testing data
if train_lse < test_lse:
    print("The model may have good generalization ability.")
else:
    print("The model may have overfitting issues.")

import matplotlib.pyplot as plt

# Plot actual vs. predicted values for training data
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Actual vs. Predicted (Training)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Training)')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs. predicted values for testing data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='green', label='Actual vs. Predicted (Testing)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Testing)')
plt.legend()
plt.grid(True)
plt.show()
