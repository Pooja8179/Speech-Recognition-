# Import necessary libraries
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from zipfile import ZipFile
from google.colab import files

# Step 1: Upload the ZIP file
uploaded = files.upload()

# Step 2: Extract the ZIP file
for filename in uploaded.keys():
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/extracted_data')

# Define a function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)  # Extract MFCCs
    return mfccs

# Initialize lists to store features and labels
features = []
labels = []

# Iterate through audio files and extract features
for root, _, files in os.walk('/content/extracted_data/speech'):  # Update path as needed
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            label = file.split('-')[2]  # Update based on your naming convention
            labels.append(label)
            features.append(extract_features(file_path))

# Convert features and labels into a DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# Preprocess the data
numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
df[numeric_df.columns] = df[numeric_df.columns].fillna(numeric_df.mean())  # Fill NaN in numeric columns only

X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Gradient Boosting model
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = gbc.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
