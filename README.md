# Video Synopsis Tool

A comprehensive video synopsis generation tool that supports multiple tracking methods including bounding boxes and masks.

## 🚀 Installation

### Environment Setup
```bash
pip install -r requirements.txt
```

### CenterMask2 Installation
Choose one of the following methods:

**Method 1: Git Clone**
```bash
git clone https://github.com/youngwanLEE/centermask2.git
```

**Method 2: Download from Google Drive (Recommended)**

Download `centermask2.zip` from [Google Drive](https://drive.google.com/file/d/1mtqJdoLYfPrdr744ZTE9cMIU6fI3Dytp/view?usp=sharing)

## 💻 Usage

### Bounding Box Synopsis
Generate video synopsis using bounding box annotations:
```bash
python synopsis_box.py --anno annotation_Tung_mydinh3.txt --video_path mydinh3.mp4
```

### Mask-based Synopsis
Generate video synopsis using mask annotations:
```bash
python synopsis_anno_mask.py --anno ./centermask2/mydinh3.npy --video_path mydinh3.mp4
```

### Vehicle Lane Alignment
Create synopsis with vehicles aligned to the same lane:
```bash
python synopsis_shift.py --anno annotation_Tung_mydinh3.txt --video_path mydinh3.mp4
```

### Additional Options
For more configuration options, refer to `solo_daxua.py`

## 📁 Demo Data

### Available Videos
- `vcc6.mp4` with annotation file `annotation_vcc6.txt`
- `mydinh3.mp4` with annotation file `annotation_Tung_mydinh3.txt`

### Annotation Methods
- **Bounding Box Annotations**: Generated using YOLOv4 + DeepSORT
- **Mask Annotations**: Generated using CenterMaskv2 + DeepSORT


