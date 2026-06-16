# 1. Mount Google Drive filesystem storage arrays
from google.colab import drive
drive.mount('/content/gdrive')

# 2. Install modern high-throughput Ultralytics Computer Vision libraries
!pip install ultralytics

# 3. Initialize training run utilizing transfer learning loops
import os
from ultralytics import YOLO

# Instantiate the medium architecture model benchmarked heavily on your paper
model = YOLO('yolov8m-pose.pt') 

# Execute training pipeline utilizing your custom camera configurations
# Sets image dimensions to letterboxed 640 inputs as configured in Section VI-B
model.train(
    data='/content/gdrive/My Drive/PoseDetectionSystem/config.yaml', 
    epochs=50, 
    imgsz=640,
    batch=16,
    device=0  # Pin operations cleanly to your target cloud GPU
)

# 4. Safely export compiled weights back into your persistence cloud drive
!scp -r /content/runs '/content/gdrive/My Drive/PoseDetectionSystem/weights_output'