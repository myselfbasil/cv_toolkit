import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Load the pretrained Faster R-CNN model from torchvision
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
])

def detect_objects(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert("RGB")

    # Apply the transformation to the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get the bounding boxes and labels
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Convert image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    # Draw the bounding boxes around the detected objects
    for i, box in enumerate(boxes):
        if scores[i] > 0.5:  # Only draw boxes for predictions with confidence > 0.5
            xmin, ymin, xmax, ymax = box.tolist()
            cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    # Convert the image back to RGB for proper display
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes
    cv2.imshow('Object Detection', image_rgb)

    # Wait for the window to be closed by the user
    while True:
        if cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(1)  # Check for window events

    # Close all OpenCV windows after the window is closed
    cv2.destroyAllWindows()

# Replace 'your_image.jpg' with the path to your image
image_path = 'test.jpeg'
detect_objects(image_path)
