import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load the image
image = Image.open('photos/rotten-apple.png')

# Prepare the inputs
inputs = processor(text=["fresh apple", "rotten apple", "fresh tomato", "fresh strawberry"], images=image, return_tensors="pt", padding=True)

# Measure inference time
start_time = time.time()  # Start timer

# Perform inference
outputs = model(**inputs)

end_time = time.time()  # End timer

# Calculate the time taken
inference_time = end_time - start_time

# Get the image-text similarity score
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)  # Get the label probabilities

# Print the results
print("Inference Time: {:.4f} seconds".format(inference_time))
print("Probabilities:", probs)
