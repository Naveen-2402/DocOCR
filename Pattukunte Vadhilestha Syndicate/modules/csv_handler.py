import csv
import os

def append_to_csv(image_path, extracted_info):
    """Append extracted information to a CSV file."""
    with open('../details.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_path, extracted_info])

def cleanup_images(directory):
    """Delete all images in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)