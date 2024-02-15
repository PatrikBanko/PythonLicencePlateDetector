import pytesseract
from PIL import Image

def recognize_text(image_path):
    """
    Recognizes text in an image using Tesseract OCR.

    Parameters:
        image_path: str
            The path to the image file.

    Returns:
        str
            The recognized text.
    """
    # Open the image
    image = Image.open(image_path)

    pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
    
    # Use pytesseract to perform OCR on the image
    text = pytesseract.image_to_string(image)
    
    return text

# Example usage:
image_path = "filename.jpg"  # Replace with the path to your image file
recognized_text = recognize_text(image_path)
print("Recognized text:")
print(recognized_text)
