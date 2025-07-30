
🧠 Brain Tumor Segmentation using YOLOv11 and SAM2

This project implements brain tumor detection and segmentation using a hybrid approach that combines YOLOv11 for object detection and SAM2 (Segment Anything Model 2) for precise segmentation. It was developed using Python in Google Colab with a medical brain MRI dataset.

---

📌 Project Overview

- Goal: Automatically detect and segment tumors in brain MRI scans.
- Techniques Used:
  - YOLOv11: Fast and accurate real-time object detection model.
  - SAM2: High-resolution segmentation using prompt-based transformer model.
- Tools: Python, OpenCV, PyTorch, Ultralytics, Segment Anything, Google Colab

---

📁 Dataset Structure

The dataset is organized as follows:

dataset/
├── train/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/

Each label is in YOLO format and corresponds to a bounding box annotation for tumor regions.

---

⚙️ Installation & Setup

🔧 Install Dependencies

pip install ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python-headless matplotlib

📥 Download SAM2 Checkpoint

mkdir -p sam_checkpoints
wget -O sam_checkpoints/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

---

🧠 YOLOv11 Training

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="brain-tumor.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="yolov11_brain_tumor",
    verbose=True
)

---

✂️ Tumor Segmentation with SAM2

from segment_anything import sam_model_registry, SamPredictor
import cv2, numpy as np
import matplotlib.pyplot as plt

sam = sam_model_registry["vit_h"](checkpoint="sam_checkpoints/sam_vit_h.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

image_path = "path/to/test/image.png"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

h, w = image_rgb.shape[:2]
input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

plt.imshow(image_rgb)
for mask in masks:
    plt.contour(mask, colors='red', linewidths=2)
plt.title("Tumor Segmentation")
plt.axis('off')
plt.show()

---

📊 Evaluation & Visualization

- Class distributions visualized with bar and pie charts.
- Evaluation of YOLOv11 predictions and SAM2 masks.
- Segmentation results plotted using matplotlib.

---

📝 Report Summary

Includes:
- Introduction & Objectives
- Dataset Description
- YOLOv11 Training Results
- SAM2 Visualization


📄 Brain_Tumor_Segmentation_Report.docx is also available.

---

📌 Observations

- YOLOv11 detects boxes quickly and accurately.
- SAM2 provides precise segmentation masks.
- Hybrid use enhances interpretability and detail.

---

👩‍💻 Author

- Hafsa Mustafa
- Final Project — AI/Machine Learning
- Developed using Google Colab and Python 3

---

🔗 References

- YOLOv11 - Ultralytics GitHub
- Segment Anything - Facebook AI Research
- Tutorial by CodeWithArohi Hind
- https://youtu.be/pFIwBmlm2O4?si=4ZbaA0KfPf9OD3m6
https://youtu.be/rPOYIUiij90?si=PObaV0HFtb1Tutfb
https://youtu.be/UOoSw9VfdS4?si=2wQHgXssBJ7dMr0V
https://youtu.be/5er9ozQdjyk?si=NPFoUcxrzNmT3Y4Y
https://www.kaggle.com/code/aryashah2k/tutorial-yolo-v11-sam2-tumor-detection/notebook



---

📃 License

For educational and research purposes only. Commercial use is not permitted.
