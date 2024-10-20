from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

# Load the pre-trained model and processor
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")  # Placeholder name, update it with the exact model name
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

image = Image.open('photos/expiry_label.png')
# Preprocess the image using the processor
inputs = processor(images=image, return_tensors="pt")
# Define a prompt for the task
question = "What is the expiry date in this image?"
inputs.update({"input_ids": processor.tokenizer(question, return_tensors="pt").input_ids})

# Generate response using the model
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode the output to readable text
expiry_date = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Extracted Expiry Date:", expiry_date)
import re

def extract_date(text):
    date_pattern = r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b'
    match = re.search(date_pattern, text)
    if match:
        return match.group(0)
    return "Date not found"

expiry_date_formatted = extract_date(expiry_date)
print("Formatted Expiry Date:", expiry_date_formatted)
