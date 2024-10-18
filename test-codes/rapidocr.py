from paddleocr import PaddleOCR

# Initialize the OCR model with English language
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Change 'en' to other language codes as needed

# Path to your image
image_path = 'expiry_label.png'  # Replace with your actual image path

# Perform OCR
result = ocr.ocr(image_path, cls=True)

# Extract and print the text and confidence from the results
for line in result[0]:
    text = line[1][0]  # The recognized text
    confidence = line[1][1]  # Confidence score for the text
    print(f"Detected text: {text} (Confidence: {confidence:.2f})")
