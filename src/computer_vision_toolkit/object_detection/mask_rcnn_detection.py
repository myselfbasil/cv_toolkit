import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Load the pretrained Mask R-CNN model from torchvision
model = models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")  # Using 'weights' instead of 'pretrained'
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

    # Get the bounding boxes, labels, masks, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']

    # Convert image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    # Loop over the detections and draw boxes and masks
    for i, box in enumerate(boxes):
        if scores[i] > 0.5:  # Only draw boxes for predictions with confidence > 0.5
            xmin, ymin, xmax, ymax = box.tolist()

            # Draw bounding box
            cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

            # Get the mask for the detected object
            mask = masks[i, 0]  # Take the first mask (there's only one mask per object)
            mask = mask.mul(255).byte().cpu().numpy()  # Convert mask to 255 scale (for visualization)

            # Resize the mask to match the image size (just in case it's smaller or larger)
            mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

            # Create a 3D mask (the same size as the image) to apply to the image
            mask_3d = np.stack([mask_resized] * 3, axis=-1)

            # Apply the mask to the image, coloring the detected object in green
            image_np[mask_3d[:, :, 0] == 255] = [0, 255, 0]  # Green color for the mask

    # Display the image with bounding boxes and masks
    cv2.imshow('Mask R-CNN Object Detection', image_np)

    # Wait for the window to be closed by the user
    while True:
        # Check if the window is still open
        if cv2.getWindowProperty('Mask R-CNN Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break  # If the window is closed, break the loop

        # Otherwise, continue to check
        cv2.waitKey(1)

    cv2.destroyAllWindows()  # Close all OpenCV windows

# Replace 'your_image.jpg' with the path to your image
image_path = 'test.jpeg'
detect_objects(image_path)
