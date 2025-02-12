### *Speech Emotion Recognition Using Machine Learning - Research Report Summary*

#### *Introduction & Objective*
Speech Emotion Recognition (SER) is an advanced machine learning (ML) method for classifying human emotions from speech signals. It has extensive applications in human-computer interaction, mental health diagnosis, and disabled accessibility software. The research in this report aims at the extraction of useful audio features such as *Mel-Frequency Cepstral Coefficients (MFCC), Mel Spectrogram, and Chroma* for classifying emotional states.

SER is broken down into three major steps:
1. *Data Processing:* Preprocessing and organization of speech datasets.
2. *Feature Extraction:* Extraction of useful audio features for emotion recognition.
3. *Classification:* Use of different ML algorithms for predicting emotions.

One of the primary challenges in SER is high accuracy due to subjective emotional perception and dataset limitations. This research explores different ML approaches to enhance SER accuracy and reliability.

#### *Dataset Description*
The research uses the *Ryerson Audiovisual Database of Emotional Speech and Song (RAVDESS)*. The dataset includes:
- *7356 speech samples* recorded by 24 professional actors (12 male, 12 female).
- Seven emotional expressions: *happiness, sadness, anger, surprise, disgust, calm, and fear*.
- Each phrase read with two intensity levels: *normal and strong*.

Only the *audio recordings* were used for model training and testing for this research.

#### *Machine Learning Approaches*
The research compares *traditional and ensemble ML classifiers* to determine the best model for SER. The algorithms explored are:

1. *K-Nearest Neighbors (KNN):* A simple classification method that classifies emotions based on the nearest neighbors in feature space. Though easy to understand, it is poor with high-dimensional data and noise.
2. *Logistic Regression:* A linear model used for predicting probabilities for emotion classes. It is best used when emotions are well separated but doesn't suit complex audio signals.
3. *Gradient Boosting:* An ensemble used to iteratively build on weak classifiers. It performs well with noisy and complex data.
4. *Adaptive Boosting (AdaBoost):* An ensemble that assigns higher weights to incorrectly classified samples, iteratively refining predictions.
5. *Other models experimented in earlier work:* Support Vector Machine (SVM), Decision Trees, Random Forest, Multilayer Perceptron (MLP), and Convolutional Neural Networks (CNN).

#### *Methodology*
1. *Preprocessing:*
   - *Data Augmentation:* Noise injection was some of the techniques employed to add diversity and robustness to datasets.
   - *Normalization:* Guaranteed attributes have equal influence on models.

2. *Feature Extraction:*
   - *MFCC:* Picked vocal tract features.
   - *Mel Spectrogram:* A representation of the short-term power spectrum of sound.
   - *Chroma Features:* Examines the distribution of pitches.
   - *Contrast & Tonnetz:* Complements speech modulation and tonal features.

3. *Model Training & Evaluation:*
   - Data were split into *training (70%) and testing (30%)* or *training (80%) and validation (20%)*.
   - Performance was calculated using *accuracy, precision, recall, and F1-score*.
   - A *confusion matrix* was utilized to determine classification results.

#### *Results & Comparison*
The research compared *existing models (SVM, Decision Tree, Random Forest, MLP, CNN)* to *proposed models (KNN, Logistic Regression, Gradient Boosting, AdaBoost)*.

- *Gradient Boosting achieved highest accuracy (0.87)* among proposed models.
- *Random Forest was the best (0.83)* in earlier work.
- *Logistic Regression achieved lowest accuracy (0.43), and it performed poorly with complex audio data.*
- *KNN (0.57) and AdaBoost (0.63) achieved moderate performance.*
- *Deep Learning techniques (CNN, MLP) performed similarly well, although with increased computational requirements.*

#### *Conclusion & Future Work*
The experiment is successfully shown to be effective for *emotion classification from speech signals*. The **Gradient Boosting model was found to be the best* out of the methods proposed.
Future work aims to:
- Integrate *deep learning* to represent features more effectively.
- Classify *compound emotions* (happily surprised, angrily fearful).
- Generalize better for *speaker-independent* SER models.

This research is of interest for *designing robust emotion recognition systems*, and it may help enhance AI-based interactions in applications like customer service, mental health assistance, and accessibility solutions.
