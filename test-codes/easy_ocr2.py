import easyocr
import cv2
from matplotlib import pyplot as plt

# Initialize the EasyOCR reader (specify the language you want to detect)
reader = easyocr.Reader(['en'])

# Load the image
image_path = 'photos/ashwagandha.jpg'
image = cv2.imread(image_path)

# Perform OCR on the image
result = reader.readtext(image_path)

# Print the detected text
for detection in result:
    print(detection)

# Optionally visualize the detected text on the image
for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    
    # Draw a rectangle around the detected text
    image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    # Put the detected text above the rectangle
    image = cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Display the image with detected text
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
