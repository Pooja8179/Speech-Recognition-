import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Helper function to extract MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Initialize lists to store features and labels
features = []
labels = []

# Mapping for emotions based on file name convention
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Replace 'speech_data_path' with the path to your extracted data
speech_data_path = 'C:\\Users\\JYOTHIKA.G\\OneDrive\\Documents\\ML LAB\\speech-emotion-recognition-ravdess-data'

for actor_folder in os.listdir(speech_data_path):
    actor_folder_path = os.path.join(speech_data_path, actor_folder)
    for file_name in os.listdir(actor_folder_path):
        emotion_code = file_name.split("-")[2]
        emotion_label = emotion_map.get(emotion_code)
        
        if emotion_label:
            file_path = os.path.join(actor_folder_path, file_name)
            mfcc_features = extract_mfcc(file_path)
            features.append(mfcc_features)
            labels.append(emotion_label)

# Convert lists to arrays
features = np.array(features)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Logistic Regression classifier
logistic_model = LogisticRegression(max_iter=5500)
logistic_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = logistic_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
