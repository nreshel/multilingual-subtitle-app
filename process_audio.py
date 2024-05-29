import os
import numpy as np
import librosa
import pandas as pd

def extract_features(file_path, sr=16000, n_mfcc=13, hop_length=512):
    """Extract audio features such as MFCCs, Chroma, and Spectral Contrast."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        
        if len(y) == 0 or np.max(np.abs(y)) < 0.01:
            print(f"Skipping empty or low amplitude audio file: {file_path}")
            return None

        if len(y) < hop_length:
            hop_length = len(y) // 2
            print(f"Adjusting hop length for short file: {file_path}")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        
        features = np.concatenate((np.mean(mfccs, axis=1), 
                                   np.mean(chroma, axis=1), 
                                   np.mean(spectral_contrast, axis=1)), axis=0)
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_and_extract_features(input_dir):
    """Process audio files and extract features."""
    data = []
    labels = []
    filenames = []

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            language_dir = os.path.join(root, dir_name)
            for file in os.listdir(language_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(language_dir, file)
                    features = extract_features(file_path)
                    if features is not None:
                        data.append(features)
                        labels.append(dir_name)
                        filenames.append(file)

    return np.array(data), labels, filenames

if __name__ == "__main__":
    input_directory = "cleaned_data"
    output_csv = "features_all_languages.csv"

    data, labels, filenames = process_and_extract_features(input_directory)

    if len(data) == 0:
        print("No valid data found.")
    else:
        df = pd.DataFrame(data)
        df['label'] = labels
        df['filename'] = filenames
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
