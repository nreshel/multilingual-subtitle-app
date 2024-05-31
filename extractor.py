import os
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr

def load_and_denoise(file_name, sr=16000):
    # Load audio file
    audio, sample_rate = librosa.load(file_name, sr=sr)
    
    # Apply noise reduction
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sample_rate)
    
    return reduced_noise_audio, sample_rate

def extract_features(file_name, sr=16000, n_mfcc=40, hop_length=512, n_fft=2048):
    try:
        # Load and denoise audio file
        audio, sample_rate = load_and_denoise(file_name, sr=sr)
        if len(audio) == 0:
            print(f"Skipping empty file: {file_name}")
            return None
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Zero-padding if audio is shorter than the frame length
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)), 'constant')
        
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Compute Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
        
        # Compute Spectral Contrast features
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
        
        # Compute the mean of the features over all frames
        mfccs_mean = np.mean(mfccs.T, axis=0)
        delta_mfccs_mean = np.mean(delta_mfccs.T, axis=0)
        delta2_mfccs_mean = np.mean(delta2_mfccs.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # Combine all features into a single feature vector
        features = np.concatenate((mfccs_mean, delta_mfccs_mean, delta2_mfccs_mean, chroma_mean, spectral_contrast_mean))
        
        return features
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

# Define the root directory containing the subdirectories of audio files
root_dir = 'cleaned-data'

# Initialize an empty list to hold the extracted features and labels
extracted_features = []

# Walk through each subdirectory in the root directory
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            # Extract features
            features = extract_features(file_path)
            if features is not None:
                # Get the language (subdirectory name)
                language = os.path.basename(subdir)
                # Append the features, language, and file name to the list
                extracted_features.append([features, language, file])

# Create a DataFrame from the extracted features
df = pd.DataFrame(extracted_features, columns=['features', 'language', 'filename'])

# Split the features column into separate columns
features_df = pd.DataFrame(df['features'].tolist())
final_df = pd.concat([features_df, df['language'], df['filename']], axis=1)

# Save the DataFrame to a CSV file
output_csv = 'enhanced_audio_features-new.csv'
final_df.to_csv(output_csv, index=False)

print(f"Features extracted and saved to {output_csv}")
