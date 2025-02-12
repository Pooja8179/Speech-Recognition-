# Import required libraries
import zipfile
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files

# Step 1: Upload and unzip the dataset
uploaded = files.upload()  # Manually upload 'speech-emotion-recognition-ravdess-data.zip'

# Extract the uploaded zip file
zip_path = 'speech-emotion-recognition-ravdess-data.zip'  # File name should match the uploaded file
unzip_path = '/content/speech_emotion_data'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

# Step 2: Load and preprocess the audio data
def extract_features(file_path):
    """Extract features from an audio file for emotion recognition."""
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Prepare lists to hold data and labels
features = []
labels = []

# Loop through the dataset and extract features and labels
for root, dirs, files in os.walk(unzip_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            features.append(extract_features(file_path))
            # Extract label based on file name pattern (customize as per your dataset's structure)
            emotion_label = int(file.split('-')[2])  # Modify this if different structure
            labels.append(emotion_label)

# Convert to DataFrame
features = np.array(features)
labels = np.array(labels)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # k=5 as a starting point
knn.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
