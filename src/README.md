# Sprint 2: Language Detection System

## Introduction

This project develops a robust language detection system to accurately detect spoken languages in audio recordings and output subtitles in the detected language. The primary objective is to create a seamless and efficient multilingual subtitle generation system for various scenarios including entertainment, education, business, and professional settings. We leverage spectrogram images of audio data as our primary feature set.

## Project Workflow

Our language detection system follows a systematic workflow:

### 1. Data Collection and Preparation

- Data Collection: Collect diverse audio recordings in multiple languages
- Spectrogram Conversion: Convert audio recordings into spectrogram images
- Initial Data Exploration: Load dataset, display basic information, check for missing values, visualize language distribution

### 2. Data Visualization

- First and Mean Images: Visualize first and mean spectrogram images for each language
- Pixel Intensity Distribution: Plot distribution of pixel intensities for each language

### 3. Data Preprocessing

- Label Encoding: Encode language labels using LabelEncoder
- Data Splitting: Split dataset into training and testing sets
- Data Standardization: Standardize spectrogram image data using StandardScaler

### 4. Principal Component Analysis (PCA)

- Dimensionality Reduction: Apply PCA to standardized data
- Visualization: Visualize data in 2D space using first two principal components
- Optimal Components: Determine optimal number of PCA components

### 5. Model Training and Evaluation

- Model Training: Train and evaluate multiple models (Logistic Regression, Random Forest, XGBoost)
- Model Evaluation: Assess each model's performance based on accuracy, classification reports, and ROC curves

### 6. Model Comparison and Selection

- Comparison: Compare performance of trained models
- Saving Models: Save models to disk for future use

### 7. KMeans Clustering

- Clustering: Apply KMeans clustering to PCA-transformed data
- Optimal Clusters: Determine optimal number of clusters using elbow method and silhouette scores

### 8. Feature Importance Analysis

- Random Forest: Analyze feature importance to identify significant features
- XGBoost: Analyze feature importance for the XGBoost model

### 9. Predictions on New Data

- New Dataset: Load new dataset of spectrogram images and make predictions using trained models
- Evaluation: Evaluate results to ensure models' robustness and generalizability

### 10. Model Evaluation on Test Set

- Load Model: Load trained ResNet50-based CNN model from HDF5 file
- Evaluate Model: Evaluate loaded model on test set

### 11. Making Predictions with the Loaded Model

- Predictions: Use loaded model to make predictions on subset of test data
- Class Labels: Convert predicted class indices to original class labels

## Conclusion

The ResNet50-based CNN model demonstrated excellent performance, achieving a test accuracy of 99.04%. The model's effectiveness was evident through rapid convergence and high accuracy on both training and validation datasets. The ability to save, load, and reuse the model enhances its practical utility. By leveraging transfer learning and fine-tuning, we effectively adapted the pre-trained ResNet50 model to our specific classification problem, achieving high accuracy and reliability. This approach can be generalized to other classification tasks, showcasing the robustness and versatility of deep learning models in handling complex datasets.