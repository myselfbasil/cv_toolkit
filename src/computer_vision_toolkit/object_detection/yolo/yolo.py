import os
import torch
from ultralytics import YOLO
import requests
import sys
from glob import glob

class YOLOModelDownloader:
    def __init__(self, version):
        self.version = version
        self.base_url = self.get_base_url(version)
        self.model_path = f"yolo{version}.pt"
    
    def get_base_url(self, version):
        # Define URLs for YOLO models from v5 to v11
        urls = {
            "5": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m.pt",
            "6": "https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt",
            "7": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "8": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
            "9": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov9m.pt",
            "10": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10m.pt",
            "11": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
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
    if len(sys.argv) < 3 or not sys.argv[1].startswith("-yolov"):
        print("Usage: python yolo.py -yolov* <path_to_dataset>")
        sys.exit(1)
    
    version = sys.argv[1].replace("-yolov", "")
    if not version.isdigit():
        print("Invalid version format. Please use a format like '-yolov8'.")
        sys.exit(1)
    
    # Check if the version is less than 5
    if int(version) < 5:
        print("This script only supports YOLO versions from 5 and above.")
        sys.exit(1)

    dataset_path = sys.argv[2]
    
    # Download model
    downloader = YOLOModelDownloader(version)
    model_path = downloader.download_model()
    
    # Load model
    model = YOLO(model_path)
    
    # Process each image in the dataset directory
    image_paths = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.jpeg")) + glob(os.path.join(dataset_path, "*.png"))
    
    for image_path in image_paths:
        results = model(image_path)  # Run inference on each image
        
        # Iterate through results list
        for result in results:
            result.show()  # Display results
            # result.save()  # Remove or comment this line to avoid saving

if __name__ == "__main__":
    main()
