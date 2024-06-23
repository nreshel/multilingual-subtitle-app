import os
import csv
from PIL import Image
import numpy as np

# Define the directory containing the spectrogram folders
base_dir = 'mel_spectrograms'

# Specify the output CSV file
output_csv = 'spectrogram_flattend_image_first_10000_data.csv'

# Open a CSV file to write the data
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Define the desired image dimensions
    img_height, img_width = 128, 128

    # Generate header for the CSV file
    header = ['pixel_' + str(i) for i in range(img_height * img_width)] + ['language']
    writer.writerow(header)

    # Process each subdirectory in the base directory
    for language in os.listdir(base_dir):
        language_dir = os.path.join(base_dir, language)
        if os.path.isdir(language_dir):
            count = 0
            # Process each image file in the subdirectory
            for filename in os.listdir(language_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')) and count < 10000:
                    image_path = os.path.join(language_dir, filename)
                    try:
                        # Open the image
                        with Image.open(image_path) as img:
                            # Convert to RGB if not already
                            img_rgb = img.convert('RGB')
                            
                            # Resize the image
                            img_resized = img_rgb.resize((img_height, img_width))
                            
                            # Convert the image to a numpy array and compute the mean along the RGB channels
                            img_array = np.array(img_resized)
                            img_mean = img_array.mean(axis=2).astype(int)
                            
                            # Flatten the image and convert to a list of integers
                            flattened_image = img_mean.reshape(-1).tolist()
                            
                            # Write the row to the CSV
                            writer.writerow(flattened_image + [language])

                            # Update user on progress
                            count += 1
                            print(f"Processed {count} images in {language}")
                            
                    except Exception as e:
                        print(f"Failed to process {image_path}: {e}")
                elif count >= 10000:
                    break

print("Data preprocessing complete.")