import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image


model_name = "Aryaman9999/maggi-parle-g"

# Load the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Ensure that model is on the right device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get the class label mapping from the model config
label_mapping = model.config.id2label  # This is the mapping of id to label

# Function to perform inference and return class label
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    # Retrieve the predicted class label from id2label
    predicted_class = label_mapping[predicted_class_idx]
    if(predicted_class_idx==0):
        print("maggi")
    else:
        print("parle-g")  
    
    # print(f"Predicted class: {predicted_class}")
    return predicted_class

# Example usage with an image file path
image_path = "photos/parleg.jpg"  # Replace with the actual image path
predicted_class = predict_image(image_path)
