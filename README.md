# Flipkart Grid Challenge Submission

This repository contains our solutions for the **Flipkart Grid Challenge**, focusing on various computer vision tasks such as Optical Character Recognition (OCR), image recognition, and freshness detection of fresh produce.

# IMPORTANT
Models trained by our team have almost 125 downloads on Hugging Face which means many other teams will also be using our model.

## Table of Contents

- [Competition Overview](#competition-overview)
- [Solutions](#solutions)
  - [1. OCR to Extract Details from Images/Labels](#1-ocr-to-extract-details-from-imageslabels)
  - [2. Using OCR to Get Expiry Date Details](#2-using-ocr-to-get-expiry-date-details)
  - [3. Image Recognition and Infrared (IR) Based Counting](#3-image-recognition-and-infrared-ir-based-counting)
  - [4. Detecting Freshness of Fresh Produce](#4-detecting-freshness-of-fresh-produce)
- [Models and Techniques](#models-and-techniques)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Competition Overview

The Flipkart Grid Challenge presented multiple tasks aimed at enhancing retail operations through machine learning and computer vision:

1. **OCR to Extract Details from Images/Labels (20%)**  
   Extract details such as brand name, pack size, and other information from packaging materials using OCR.

2. **Using OCR to Get Expiry Date Details (10%)**  
   Utilize OCR to read expiry dates and MRP details printed on items for validation purposes.

3. **Image Recognition and IR Based Counting (30%)**  
   Implement machine learning to recognize brands, count items, and extract other details from images.

4. **Detecting Freshness of Fresh Produce (40%)**  
   Predict the shelf life of fresh fruits and vegetables by assessing their freshness through visual cues and patterns.

---

## Solutions

### 1. OCR to Extract Details from Images/Labels

- **Objective:** Extract details available on packaging materials, including brand name, pack size, and other relevant information.
- **Approach:**
- - **RapidOCR:** Great accuracy on labels.
  - **EasyOCR:** Chosen for its lightweight architecture and good accuracy.
  - **PaddleOCR:** Selected for higher accuracy despite being slower than EasyOCR.
  - **GOT OCR:** Used for multi-column or uniquely formatted labels (requires GPU*).
  - **Qwen:** Tested for its high accuracy, though it's relatively large in size.

### 2. Using OCR to Get Expiry Date Details

- **Objective:** Extract expiry dates and MRP details printed on items for validation.
- **Approach:**
  - Leveraged the same OCR tools as above, focusing on fields related to expiry dates and MRP.
  - Implemented pre-processing steps to enhance text detection in challenging conditions.

### 3. Image Recognition and Infrared (IR) Based Counting

- **Objective:** Recognize brands and confirm item counts using image recognition as we did not have access to IR data.
- **Approach:**
  - **Brand Recognition:**
    - Finetuned a Vision Transformer (ViT) model on a custom dataset containing images of Parle-G and Maggi packets.

### 4. Detecting Freshness of Fresh Produce

- **Objective:** Predict the shelf life of fresh fruits and vegetables by assessing freshness.
- **Approach:**
  - **Vision Transformer (ViT):**
    - Finetuned on a custom dataset of fresh and not-so-fresh produce.
    - Achieved the best results with 30 downloads on Hugging Face.
  - **ResNet18:**
    - Unfroze the last block layers and finetuned on the dataset.
    - Model weights are provided (`model_weight.pth`).
  - **One-Shot Classification:**
    - Implemented to recognize freshness with minimal training examples.
  - **ResNet50:**
    - Finetuned on the custom dataset with 10 downloads on Hugging Face.
  - **VGG16:**
    - Achieved good accuracy; the model file is provided (`my_model.h5`).

---

## Models and Techniques

- **Vision Transformer (ViT):**
  - Leveraged for its effectiveness in image classification tasks.
  - Finetuned on custom datasets for both freshness detection and brand recognition.

- **Convolutional Neural Networks (CNNs):**
  - **ResNet18 and ResNet50:**
    - Used transfer learning by finetuning on custom datasets.
  - **VGG16:**
    - Provided good accuracy in freshness detection tasks.

- **OCR Tools:**
  - **EasyOCR and PaddleOCR:**
    - Selected based on a balance between speed and accuracy.
  - **GOT OCR and Qwen:**
    - Utilized for specialized cases requiring high accuracy or handling complex label formats.

---

## Usage

### Prerequisites

- Python 3.6 or higher
- PyTorch, TensorFlow, or Keras (depending on the model)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/flipkart-grid-challenge.git
   cd flipkart-grid-challenge


2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
3.**Install Dependencies:**
   ```bash
    pip install -r requirements.txt
```
### Model Files

**Model Weights:**

- `models/model_weight.pth`: Weights for the ResNet18 model.
- `models/my_model.h5`: Trained VGG16 model for freshness detection.

**Pre-trained Models on Hugging Face:**

- ViT models for freshness detection and brand recognition are available on [Hugging Face](https://huggingface.co/).

### Repository Structure

```plaintext
flipkart-grid-challenge/
├── photos/   
├── models/
│   ├── model_weight.pth
│   └── my_model.h5
├── test_codes/
├── requirements.txt
├── README.md plaintext
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- **Flipkart** for organizing the Grid Challenge.
- **Hugging Face** for providing a platform to host and share models.
- **Open-source OCR Tools** like EasyOCR and PaddleOCR for their invaluable contributions.

### References

- [EasyOCR Documentation](https://www.jaided.ai/easyocr/)
- [PaddleOCR GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
