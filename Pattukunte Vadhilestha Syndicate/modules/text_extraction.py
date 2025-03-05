import easyocr

def initialize_reader():
    """Initialize EasyOCR reader."""
    return easyocr.Reader(['en'])

def extract_text_from_image(reader, image_path):
    """Extract text from an image using EasyOCR."""
    results = reader.readtext(image_path)
    extracted_text = "\n".join([result[1] for result in results if result[2] >= 0.5])
    return extracted_text