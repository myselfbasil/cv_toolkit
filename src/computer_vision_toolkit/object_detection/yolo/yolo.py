import os
import sys
import requests
from glob import glob
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOModelDownloader:
    """A class to handle downloading YOLO models based on version."""

    def __init__(self, version: str):
        """Initialize with YOLO version and generate model download URL."""
        self.version = version
        self.base_url = self._get_base_url(version)
        self.model_path = f"yolo{version}.pt"
    
    def _get_base_url(self, version: str) -> str:
        """Returns the appropriate URL for the model based on version."""
        urls = {
            "5": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m.pt",
            "6": "https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt",
            "7": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "8": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
            "9": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov9m.pt",
            "10": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10m.pt",
            "11": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        }

        url = urls.get(version)
        if not url:
            raise ValueError(f"Unsupported YOLO version: {version}. Supported versions: 5-11.")
        
        return url

    def download_model(self) -> str:
        """Download the model if not already present."""
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading YOLO{self.version} model...")
            try:
                response = requests.get(self.base_url, stream=True)
                response.raise_for_status()  # Will raise HTTPError for bad responses
                
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Successfully downloaded YOLO{self.version} model to {self.model_path}")
            except requests.RequestException as e:
                logger.error(f"Error downloading model: {e}")
                sys.exit(1)
        
        return self.model_path

def main():
    """Main function to parse arguments, download the model, and process images."""
    if len(sys.argv) < 3 or not sys.argv[1].startswith("-yolov"):
        logger.error("Usage: python yolo.py -yolov* <path_to_dataset>")
        sys.exit(1)

    version = sys.argv[1].replace("-yolov", "")
    
    if not version.isdigit():
        logger.error("Invalid version format. Please use a format like '-yolov8'.")
        sys.exit(1)

    version_int = int(version)
    if version_int < 5:
        logger.error("This script only supports YOLO versions 5 and above.")
        sys.exit(1)

    dataset_path = sys.argv[2]

    # Download the model
    downloader = YOLOModelDownloader(version)
    model_path = downloader.download_model()

    # Load model
    model = YOLO(model_path)

    # Process each image in the dataset directory
    image_paths = glob(os.path.join(dataset_path, "*.jpg")) + \
                  glob(os.path.join(dataset_path, "*.jpeg")) + \
                  glob(os.path.join(dataset_path, "*.png"))

    if not image_paths:
        logger.warning("No images found in the dataset directory.")

    for image_path in image_paths:
        try:
            results = model(image_path)  # Run inference on each image
            for result in results:
                result.show()  # Display results
                # result.save()  # Uncomment this line if saving results is required
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")

if __name__ == "__main__":
    main()
