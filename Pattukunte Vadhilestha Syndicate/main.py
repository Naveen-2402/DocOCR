import cv2
import os
import csv
import json
from modules.capture_frames import initialize_camera, capture_frame
from modules.adjust_bbox import adjust_bbox
from modules.text_extraction import initialize_reader, extract_text_from_image
from modules.info_extraction import initialize_groq, extract_info

def append_to_csv(image_path, extracted_info):
    """Append extracted information to a CSV file or save as a text file if parsing fails."""
    csv_path = "./details.csv"
    file_exists = os.path.isfile(csv_path)

    # Ensure extracted_info is a dictionary
    if isinstance(extracted_info, str):
        try:
            # Attempt to parse the response as JSON
            extracted_info = json.loads(extracted_info)  # Safely parse JSON string
        except json.JSONDecodeError as e:
            print(f"Error parsing extracted information: {e}")
            print("Saving extracted information as a text file instead.")

            # Save the extracted_info as a text file
            txt_path = "./extracted_info.txt"
            with open(txt_path, mode='a', encoding='utf-8') as txtfile:
                txtfile.write("\n\n" + extracted_info)
            return  # Exit the function after saving as text

    # Read existing columns from the CSV file (if it exists)
    existing_columns = []
    if file_exists:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_columns = reader.fieldnames or []

    # Determine all columns (existing + new)
    all_columns = list(set(existing_columns + list(extracted_info.keys())))

    # Write data to the CSV file
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)

        # Write header if the file is being created for the first time
        if not file_exists:
            writer.writeheader()

        # Write the data
        writer.writerow(extracted_info)

def cleanup_images(directory):
    """Delete all images in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def main():
    # Step 1: Capture frames using YOLO
    cam_number = 1  # Change this to the desired camera number
    model, cap = initialize_camera(cam_number)
    image, bbox = capture_frame(model, cap)

    if image is not None and bbox is not None:
        # Step 2: Adjust the bounding box and save the warped image
        warped_image = adjust_bbox(image, bbox)

        if warped_image is not None:
            # Debug: Check if warped_image is valid
            print(f"Warped image shape: {warped_image.shape}")
            print(f"Warped image dtype: {warped_image.dtype}")

            # Ensure the directory exists
            os.makedirs("./saved_objects", exist_ok=True)

            # Save the warped image (replace the existing image)
            save_path = "./saved_objects/warped_image.jpg"
            try:
                cv2.imshow("Warped Image", warped_image)
                cv2.waitKey(0)
                cv2.imwrite(save_path, warped_image)
                print(f"Warped image saved: {save_path}")
            except Exception as e:
                print(f"Error saving image: {e}")

            # Step 3: Extract text from the warped image
            reader = initialize_reader()
            detected_text = extract_text_from_image(reader, save_path)
            print("Extracted Text:")
            print(detected_text)

            # Step 4: Extract structured information
            llm = initialize_groq()
            extracted_info = extract_info(llm, detected_text)
            print("Extracted Information:")
            print(extracted_info)

            # Step 5: Save to CSV or text file and cleanup
            append_to_csv(save_path, extracted_info)
            cleanup_images("./saved_objects")
        else:
            print("No warped image was saved. Exiting.")
    else:
        print("No image was captured. Exiting.")

if __name__ == "__main__":
    main()