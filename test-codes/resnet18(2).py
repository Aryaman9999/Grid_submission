##MORE ACCURATE AS IT HAS USED UNFROZEN LAYERS


import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet18_Weights


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        
        # Use the weights parameter instead of pretrained
        self.base = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze the earlier layers
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
                    
        self.base.fc = nn.Sequential()  # Remove the original fully connected layer
            
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9)  # Assuming 9 classes for fruits
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # Assuming 2 classes for freshness (fresh, rotten)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1, y2 = self.block2(x), self.block3(x)
        return y1, y2

# Function to perform image transformation for inference
def image_transform(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Resize the image to the model's expected input size
        transforms.Normalize(mean=0, std=1)  # Adjust if different normalization was used during training
    ])
    return transform(img)

# Function to perform inference on a single image
def predict_single_image(image_path, model, device):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess the image
    image = image_transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device
    image = image.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        y1_pred, y2_pred = model(image)
    
    # Get the predicted class indices
    y1_class = torch.argmax(y1_pred, dim=1).item()
    y2_class = torch.argmax(y2_pred, dim=1).item()
    
    return y1_class, y2_class

# Load the trained model weights
def load_model(model_path, device):
    # Safe loading of the model's state dictionary
    model = Model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


# Example class labels (adjust these according to your actual class labels)
fruit_labels = ['apple', 'banana', 'bittergourd', 'capsicum', 'cucumber', 'okra', 'oranges', 'potato', 'tomato']
fresh_labels = ['fresh', 'rotten']

# Function to run the entire inference process
def run_inference(image_path, model_path):
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = load_model(model_path, device)

    # Perform prediction on a single image
    y1_class, y2_class = predict_single_image(image_path, model, device)

    # Map the predicted indices to the class labels
    predicted_fruit = fruit_labels[y1_class]
    predicted_freshness = fresh_labels[y2_class]

    # Print the results
    print(f'Predicted Fruit: {predicted_fruit}')
    print(f'Predicted Freshness: {predicted_freshness}')

# Example usage
image_path = 'photos/rotten_orange.png'  # Replace with the path to your image
model_path = 'models/model_weights.pth'  # Replace with the path to your saved model weights
run_inference(image_path, model_path)
