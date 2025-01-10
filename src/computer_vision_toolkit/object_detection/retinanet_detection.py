import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define a function to download and prepare the RetinaNet model
def get_retinanet_model():
    # Load a pre-trained RetinaNet model
    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Define a function to preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

# Define a function to draw bounding boxes around detected objects
def draw_boxes(image, detections, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(detections['boxes']):
        score = detections['scores'][i]
        if score >= threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{score:.2f}", fill="red")
    return image

# Main function
def main(image_path):
    # Step 1: Get the model
    model = get_retinanet_model()

    # Step 2: Preprocess the image
    input_tensor, original_image = preprocess_image(image_path)

    # Step 3: Perform object detection
    with torch.no_grad():
        detections = model(input_tensor)[0]

    # Step 4: Draw bounding boxes
    output_image = draw_boxes(original_image.copy(), detections)

    # Step 5: Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Provide the path to your image
    image_path = "test.jpeg"  # Replace with the actual path
    main(image_path)
