# data_science_project.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')

# Exploratory Data Analysis (EDA)
print("Head of the Iris dataset:")
print(iris.head(), "\n")

print("Dataset summary statistics:")
print(iris.describe(), "\n")

print("Class distribution:")
print(iris['species'].value_counts(), "\n")

# Data visualization: create a pairplot to visualize feature relationships
sns.pairplot(iris, hue='species')
plt.suptitle("Pairplot of the Iris Dataset", y=1.02)
plt.show()

# Preparing data for modeling
X = iris.drop('species', axis=1)  # Features
y = iris['species']               # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy, "\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

