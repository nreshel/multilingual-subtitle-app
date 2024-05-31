# Language Classification using MFCC, Delta MFCC, Chroma, and Spectral Contrast Features

## Project Overview
This project aims to develop a machine learning model for automatic language classification using audio features. The primary features extracted from the audio files include Mel-Frequency Cepstral Coefficients (MFCC), Delta MFCC, Delta-Delta MFCC, Chroma, and Spectral Contrast. These features are crucial in capturing the unique characteristics of different languages, enabling effective classification.

## Problem Area
The proliferation of audio content across different languages necessitates the development of automated language identification systems. Such systems can be beneficial in various applications, including:
- Speech recognition systems
- Multilingual customer service automation
- Language-based content filtering
- Enhancing user experience in media services

## Proposed Data Science Solution
Our solution involves the extraction of significant audio features from recorded speech samples and employing machine learning models to classify the languages. By leveraging advanced feature extraction techniques and robust machine learning algorithms, we aim to achieve high accuracy in language identification.

## Impact
The successful implementation of this project can lead to:
- Improved efficiency in multilingual speech recognition systems.
- Enhanced user experience by automatically adapting content based on language.
- Cost savings in customer service operations through automated language identification.
- Broader accessibility of digital content across diverse linguistic audiences.

## Dataset
The dataset consists of audio files from various languages, specifically English, Spanish, Chinese, and Arabic. Each audio file has been processed to extract the following features:
- **MFCC (Mel-Frequency Cepstral Coefficients):** Captures the short-term power spectrum of sound.
- **Delta MFCC:** Represents the change in MFCC over time.
- **Delta-Delta MFCC:** Captures the change in Delta MFCC.
- **Chroma:** Represents the twelve different pitch classes.
- **Spectral Contrast:** Measures the difference in amplitude between peaks and valleys in a sound spectrum.

## Data Dictionary
| Column Index | Feature Type       | Description                                                        |
|--------------|--------------------|--------------------------------------------------------------------|
| 0-129        | MFCC and Delta MFCC| MFCC features (40) and their Delta (40) and Delta-Delta (40) coefficients |
| 130-141      | Chroma             | Chroma features (12)                                               |
| 142-148      | Spectral Contrast  | Spectral contrast features (7)                                      |
| 149          | Language           | The language of the audio file (e.g., English, Spanish, Chinese, Arabic) |
| 150          | Filename           | The name of the audio file                                         |

## Preliminary Exploratory Data Analysis (EDA)
### Data Quality Concerns
- **Missing Values:** Checked and found no missing values in the dataset.
- **Duplicates:** Identified and removed any duplicate entries.
- **Outliers:** Detected and handled outliers in the feature space.

### Key Findings
- The MFCC features showed distinct patterns for each language.
- Delta and Delta-Delta MFCCs further highlighted the temporal changes in the audio signals.
- Chroma and Spectral Contrast features provided additional layers of information for language differentiation.

## Next Steps
### Data Processing:
- Normalize the features to ensure they are on a comparable scale.
- Split the dataset into training and testing sets.

### Feature Engineering:
- Explore additional features that may enhance model performance.
- Perform dimensionality reduction techniques (e.g., PCA) if necessary.

### Baseline Modeling:
- Implement and evaluate baseline models such as Random Forest and SVM.
- Fine-tune hyperparameters to optimize model performance.

### Advanced Modeling:
- Experiment with deep learning models, such as Convolutional Neural Networks (CNNs), to improve accuracy.
- Implement cross-validation techniques to ensure the robustness of the models.

## How to Run
1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd <repository-directory>
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the feature extraction script:
    ```bash
    python scripts/feature_extraction.py
    ```
5. Train the models:
    ```bash
    python scripts/model_training.py
    ```

## Contact
For any inquiries or collaboration opportunities, please contact:
- **Name:** Nimrod Eshel
- **Email:** [nimrod.eshel@outlook.com]

Feel free to customize this template to better match the specifics and requirements of your project.
