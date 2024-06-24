# README

# Sprint 2: Language Detection System

## Introduction

In this project, we are developing a robust language detection system designed to accurately detect the language spoken in audio recordings and subsequently output subtitles in the detected language. The primary objective is to create a seamless and efficient multilingual subtitle generation system that can be utilized in various scenarios such as entertainment, education, business, and professional settings. To achieve this, we are leveraging spectrogram images of the audio data as our primary feature set.

## Project Workflow

To develop this language detection system, we follow a systematic workflow involving several crucial steps, each contributing to the final goal of accurate language detection and subtitle generation:

### 1. Data Collection and Preparation

- **Data Collection**: We collect a diverse dataset of audio recordings in multiple languages.
- **Spectrogram Conversion**: Audio recordings are converted into spectrogram images.
- **Initial Data Exploration**: We load the dataset into a pandas DataFrame, display basic information, check for missing values, and visualize the distribution of languages.

### 2. Data Visualization

- **First and Mean Images**: We visualize the first and mean spectrogram images for each language to identify visual patterns.
- **Pixel Intensity Distribution**: We plot the distribution of pixel intensities for each language.

### 3. Data Preprocessing

- **Label Encoding**: Language labels are encoded using `LabelEncoder`.
- **Data Splitting**: The dataset is split into training and testing sets.
- **Data Standardization**: Spectrogram image data is standardized using `StandardScaler`.

### 4. Principal Component Analysis (PCA)

- **Dimensionality Reduction**: PCA is applied to the standardized data to reduce its dimensionality.
- **Visualization**: We visualize the data in 2D space using the first two principal components.
- **Optimal Components**: The optimal number of PCA components is determined by plotting the explained variance ratio.

### 5. Model Training and Evaluation

- **Model Training**: We train and evaluate multiple models, including Logistic Regression, Random Forest, and XGBoost, using the optimal number of PCA components.
- **Model Evaluation**: Each model's performance is assessed based on accuracy, classification reports, and ROC curves.

### 6. Model Comparison and Selection

- **Comparison**: We compare the performance of the trained models and select the best-performing model based on their evaluation metrics.
- **Saving Models**: The models are saved to disk for future use.

### 7. KMeans Clustering

- **Clustering**: KMeans clustering is applied to the PCA-transformed data to explore the data structure.
- **Optimal Clusters**: The optimal number of clusters is determined using the elbow method and silhouette scores.

### 8. Feature Importance Analysis

- **Random Forest**: Feature importance is analyzed to identify significant features (pixels) contributing to the model's predictions.
- **XGBoost**: Feature importance is similarly analyzed for the XGBoost model.

### 9. Predictions on New Data

- **New Dataset**: We load a new dataset of spectrogram images and make predictions using the trained models.
- **Evaluation**: The results are evaluated to ensure the models' robustness and generalizability to unseen data.

## Code Example: Data Collection and Preparation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import xgboost as xgb
from itertools import cycle

# Load the CSV file
csv_path = '../data/spectrogram_flattend_image_10000_data.csv'
df = pd.read_csv(csv_path)
df.head()

# Display basic information about the DataFrame
print("Basic Information:")
display(df.info())

# Display basic statistics about the data
print("\nBasic Statistics:")
display(df.describe())

# Check for missing values
display("\nMissing Values:")
display(df.isnull().sum())

# Display the distribution of languages
display("\nLanguage Distribution:")
display(df['language'].value_counts())

# Visualize the distribution of languages
plt.figure(figsize=(10, 6))
sns.countplot(x='language', data=df)
plt.title('Distribution of Languages')
plt.xlabel('Language')
plt.ylabel('Count')
plt.show()