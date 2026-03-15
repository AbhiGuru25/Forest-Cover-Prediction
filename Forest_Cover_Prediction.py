#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type Prediction

# ## Objective
# Build a system that can predict the type of forest cover using analysis data for a 30m x 30m patch of land in the forest.

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import warnings
warnings.filterwarnings('ignore')


# ## Load Dataset

# In[ ]:


# Load the data
# Assuming 'train.csv' is in the same directory
data_path = 'train.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
df.head()


# ## Exploratory Data Analysis & Preprocessing

# In[ ]:


# Drop the Id column as it's not a feature
if 'Id' in df.columns:
    df = df.drop(['Id'], axis=1)

# Check for missing values
print("Missing values in dataset:\n", df.isnull().sum().max())

# Split features and target
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# ## Model Building (Random Forest Classifier)

# In[ ]:


# Build a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Model initialized.")


# ## Training the Model

# In[ ]:


# Train the model
rf_model.fit(X_train, y_train)
print("Model training complete.")


# ## Evaluation

# In[ ]:


# Predictions
y_pred = rf_model.predict(X_test)

# Accuracy
rf_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()


# ## Saving Model

# In[ ]:


# Save the model
joblib.dump(rf_model, 'forest_cover_rf_model.pkl')
print("Model saved as forest_cover_rf_model.pkl")

