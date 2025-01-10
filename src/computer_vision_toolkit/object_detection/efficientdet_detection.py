import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained Faster R-CNN model from torchvision
def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the input image
def preprocess_image(image_path):
    # Open the image using PIL and convert to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Define the preprocessing transformations (similar to what was done during training)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Post-process the output and draw bounding boxes
def postprocess_and_draw(image_path, outputs):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Get the detections
    boxes = outputs[0]['boxes'].cpu().detach().numpy()
    labels = outputs[0]['labels'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()

    # Set a threshold for object detection confidence
    threshold = 0.5  # You can adjust this as needed
    for box, score in zip(boxes, scores):
        if score > threshold:
            # Draw a bounding box around the detected object
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to run object detection
def detect_objects(image_path):
    # Load the model
    model = load_model()
    
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Post-process and display results
    postprocess_and_draw(image_path, outputs)

# Test the script with an image
image_path = "test.jpeg"  # Replace with your image path
detect_objects(image_path)
