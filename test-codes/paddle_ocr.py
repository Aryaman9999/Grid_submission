from paddleocr import PaddleOCR

# Initialize the PaddleOCR model (use the model for English recognition)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to your input image
img_path = 'photos/ashwagandha.jpg'

# Perform OCR on the image
result = ocr.ocr(img_path)

# Print the OCR results (detected text and confidence scores)
for line in result:
    for element in line:
        text, confidence = element[1]
        print(f"Detected text: {text}, Confidence: {confidence}")
