import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import models
import numpy as np

# Load the pre-trained Faster R-CNN model as a proxy for Sparse R-CNN
class SparseRCNNModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(SparseRCNNModel, self).__init__()
        # Using Faster R-CNN with ResNet50 backbone and FPN
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    def forward(self, images):
        return self.model(images)

# Initialize and load the model
def load_model():
    model = SparseRCNNModel(pretrained=True)
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the input image for the model
def preprocess_image(image_path):
    # Load the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # Convert image to a PyTorch tensor
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Post-process and visualize the output
def postprocess_and_draw(image_path, outputs):
    # Load the original image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
    
    # Extract bounding boxes, labels, and scores
    boxes = outputs[0]['boxes'].cpu().detach().numpy()
    labels = outputs[0]['labels'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()

    # Set a confidence threshold for displaying detections
    threshold = 0.5
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            # Draw the bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add the label and confidence score
            label_text = f"Label {label}: {score:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Main function to detect objects in the image
def detect_objects(image_path):
    # Load the pre-trained model
    model = load_model()
    
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Visualize the results
    postprocess_and_draw(image_path, outputs)

# Test the script on a sample image
image_path = "test.jpeg"  # Replace with the path to your test image
detect_objects(image_path)
