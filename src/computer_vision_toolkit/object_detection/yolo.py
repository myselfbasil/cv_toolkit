import os
import torch
from ultralytics import YOLO
import requests
import sys

class YOLOModelDownloader:
    def __init__(self, version):
        self.version = version
        self.base_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo{version}.pt"
        self.model_path = f"yolo{version}.pt"
    
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
    
    # Download model
    downloader = YOLOModelDownloader(version)
    model_path = downloader.download_model()
    
    # Load model
    model = YOLO(model_path)
    
    # Example inference
    results = model("path/to/test/image.jpg")
    results.show()
    results.save()

if __name__ == "__main__":
    main()

