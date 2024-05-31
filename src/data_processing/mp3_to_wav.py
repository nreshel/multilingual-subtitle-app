import os
from pydub import AudioSegment
import tempfile
from pydub.effects import normalize

def convert_to_wav(input_path, output_path):
    """Convert MP3 file to WAV format."""
    audio = AudioSegment.from_mp3(input_path)
    audio.export(output_path, format="wav")

def normalize_audio(input_path, output_path):
    """Normalize audio file."""
    audio = AudioSegment.from_wav(input_path)
    normalized_audio = normalize(audio)
    normalized_audio.export(output_path, format="wav")

def trim_silence(input_path, output_path, silence_thresh=-50, min_silence_len=500):
    """Remove leading and trailing silence from the audio file."""
    audio = AudioSegment.from_wav(input_path)
    non_silent_chunks = AudioSegment.silent(duration=0)
    
    # Split audio into non-silent chunks
    chunks = audio[::1000]  # Split audio into 1-second chunks
    for chunk in chunks:
        if chunk.dBFS > silence_thresh and len(chunk) > min_silence_len:
            non_silent_chunks += chunk

    non_silent_chunks.export(output_path, format="wav")

def process_audio_files(input_dir, output_dir):
    """Process audio files: convert to WAV, normalize, trim silence."""
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            language_input_dir = os.path.join(root, dir_name)
            language_output_dir = os.path.join(output_dir, dir_name)

            if not os.path.exists(language_output_dir):
                os.makedirs(language_output_dir)

            for file in os.listdir(language_input_dir):
                if file.endswith('.mp3'):
                    mp3_path = os.path.join(language_input_dir, file)
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_wav_path = os.path.join(temp_dir, "temp.wav")
                        normalized_wav_path = os.path.join(temp_dir, "normalized.wav")
                        trimmed_wav_path = os.path.join(temp_dir, "trimmed.wav")
                        final_wav_path = os.path.join(language_output_dir, os.path.splitext(file)[0] + '.wav')
                        
                        # Convert MP3 to WAV
                        convert_to_wav(mp3_path, temp_wav_path)
                        
                        # Normalize the audio
                        normalize_audio(temp_wav_path, normalized_wav_path)
                        
                        # Trim silence
                        trim_silence(normalized_wav_path, trimmed_wav_path)
                        
                        # Move the trimmed file to the final destination
                        os.rename(trimmed_wav_path, final_wav_path)
                        print(f'Processed and saved {final_wav_path}')

if __name__ == "__main__":
    input_directory = "data"
    output_directory = "cleaned_data"

    process_audio_files(input_directory, output_directory)
    print("Conversion complete.")
