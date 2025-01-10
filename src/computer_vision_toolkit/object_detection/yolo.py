import os
import torch
from ultralytics import YOLO
import requests
import sys

class YOLOModelDownloader:
    def __init__(self, version):
        self.version = version
        self.base_url = self.get_base_url(version)
        self.model_path = f"yolo{version}.pt"
    
    def get_base_url(self, version):
        # Define URLs for YOLO models from v5 to v11
        urls = {
            "5": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov5/yolov5m.pt",
            "6": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov6/yolov6m.pt",
            "7": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov7/yolov7m.pt",
            "8": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov8/yolov8m.pt",
            "9": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov9/yolov9m.pt",
            "10": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov10/yolov10m.pt",
            "11": "https://github.com/myselfbasil/cv_toolkit/blob/main/src/computer_vision_toolkit/object_detection/models/yolov11/yolov11m.pt"
        }
        if version not in urls:
            raise ValueError("Unsupported version. Use versions from '5' to '11'.")
        
        return urls[version]
    
    def download_model(self):
        if not os.path.exists(self.model_path):
            try:
                response = requests.get(self.base_url, stream=True)
                response.raise_for_status()
                
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Successfully downloaded YOLO{self.version} model")
            except Exception as e:
                print(f"Error downloading model: {e}")
                sys.exit(1)
        
        return self.model_path

def main():
    if len(sys.argv) < 2 or not sys.argv[1].startswith("-yolov"):
        print("Usage: python yolo.py -yolov*")
        sys.exit(1)
    
    version = sys.argv[1].replace("-", "")
    
    # Check if the version is less than 5
    if int(version) < 5:
        print("This script only supports YOLO versions from 5 and above.")
        sys.exit(1)

    # Download model
    downloader = YOLOModelDownloader(version)
    model_path = downloader.download_model()
    
    # Load model
    model = YOLO(model_path)
    
    # Example inference on multi-class detection
    results = model("path/to/test/image.jpg")  # Ensure this path is valid and points to an image file
    results.show()
    results.save()

if __name__ == "__main__":
    main()
