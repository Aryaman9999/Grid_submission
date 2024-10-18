import easyocr
import cv2
import torch
import time
 # Start measuring time before OCR inference
start_time = time.time()
# Set number of threads for CPU optimization
torch.set_num_threads(4)  # Use 4 threads for processing

# Initialize the EasyOCR reader with CPU
reader = easyocr.Reader(['en'], gpu=False)  # Specify gpu=False to enforce CPU usage

# Function to perform OCR on an image
def perform_ocr(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Downscale the image resolution
    downscale_factor = 0.5  # Adjust the factor based on your needs
    image_resized = cv2.resize(image, None, fx=downscale_factor, fy=downscale_factor)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

   

    # Perform OCR on the grayscale image
    result = reader.readtext(gray_image)

    # Measure the time after inference
    end_time = time.time()

    # Calculate the time taken for inference
    inference_time = end_time - start_time
    print(f"OCR Inference Time: {inference_time:.4f} seconds")

    # Print the detected text
    for detection in result:
        text = detection[1]
        print(text)

# Example of calling the function with an image
image_path = 'expiry_label.png'
perform_ocr(image_path)
