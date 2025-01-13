import os
import torch
import requests
import sys
from glob import glob
import subprocess
import venv  # For creating a virtual environment
import shutil

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
            "9": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
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


class YOLOv7Runner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.initialize_yolov7()

    def initialize_yolov7(self):
        # Create or activate YOLOv7 environment
        yolov7_env_path = "yolov7_env"
        if not os.path.exists(yolov7_env_path):
            print("Creating a virtual environment for YOLOv7...")
            venv.create(yolov7_env_path, with_pip=True)

        # Activate the environment and install dependencies
        print("Installing YOLOv7 dependencies...")
        subprocess.run([os.path.join(yolov7_env_path, "bin", "pip"), "install", "-r", "yolov7/requirements.txt"], check=True)

        # Add YOLOv7 path to sys.path
        sys.path.append("yolov7")

        # Import YOLOv7 dependencies
        try:
            from models.experimental import attempt_load
            from utils.datasets import LoadImages
            from utils.general import non_max_suppression, scale_coords
            from utils.torch_utils import select_device
        except ImportError as e:
            print(f"Error importing YOLOv7 components: {e}")
            sys.exit(1)

        self.device = select_device("")
        self.model = attempt_load(self.model_path, map_location=self.device)
        self.non_max_suppression = non_max_suppression

    def run_inference(self, dataset_path):
        image_paths = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.jpeg")) + glob(os.path.join(dataset_path, "*.png"))
        
        for image_path in image_paths:
            results = self.model(image_path)  # Run inference on each image
            
            # Iterate through results list
            for result in results:
                result.show()  # Display results
                # result.save()  # Remove or comment this line to avoid saving


class YOLOModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)

    def run_inference(self, dataset_path):
        image_paths = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.jpeg")) + glob(os.path.join(dataset_path, "*.png"))
        
        for image_path in image_paths:
            results = self.model(image_path)  # Run inference on each image
            
            # Iterate through results list
            for result in results:
                result.show()  # Display results
                # result.save()  # Remove or comment this line to avoid saving


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

    if version == "7":
        # If version 7 is specified, run the YOLOv7 runner
        yolov7_runner = YOLOv7Runner(model_path)
        yolov7_runner.run_inference(dataset_path)
    else:
        # For other versions, load the model and perform inference normally
        model = YOLOModel(model_path)
        model.run_inference(dataset_path)


if __name__ == "__main__":
    main()

