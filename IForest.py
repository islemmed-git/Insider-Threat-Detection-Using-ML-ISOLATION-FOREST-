import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
email_df = pd.read_csv('data.csv')

# Display the first few rows of the DataFrame
print(email_df.head())

# Drop columns that are not needed for anomaly detection
email_df = email_df.drop(columns=['id', 'to', 'cc', 'bcc', 'content'])

# Fill missing values with 0
email_df = email_df.fillna(0)

# Convert categorical features to numeric
email_df['date'] = pd.to_datetime(email_df['date'])
email_df['date'] = email_df['date'].astype(np.int64) // 10**9  # Convert to Unix timestamp
email_df['user'] = email_df['user'].astype('category').cat.codes
email_df['pc'] = email_df['pc'].astype('category').cat.codes
email_df['from'] = email_df['from'].astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test = train_test_split(email_df, test_size=0.2, random_state=42)

# Create and train the Isolation Forest model
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train)

# Predict anomalies in the test set
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# Count the anomalies
print(f"Anomalies in training data: {Counter(y_pred_train)}")
print(f"Anomalies in test data: {Counter(y_pred_test)}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.hist(y_pred_train, bins=3, alpha=0.7, label='Training Data')
plt.hist(y_pred_test, bins=3, alpha=0.7, label='Test Data')
plt.title('Anomalies in Training and Test Data')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.legend()
plt.show()