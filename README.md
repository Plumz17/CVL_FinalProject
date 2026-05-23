# CVL_FinalProject

Final project repository for UGM's **Computer Vision and Image Analysis** course.

## Project Title

**Vehicle Speed Limit Violation Detection in CCTV Recordings Using YOLO11n and ByteTrack**

**Indonesian Title:**  
**Deteksi Pelanggaran Batas Kecepatan Kendaraan pada Rekaman CCTV Menggunakan YOLO11n dan ByteTrack**

## Overview

This project develops a computer vision system to detect vehicle speed limit violations from CCTV recordings. The system detects vehicles using a trained **YOLO11n** model, tracks them across frames using **ByteTrack**, estimates their speed through a homography-based **Bird's Eye View (BEV)** transformation, and marks vehicles that exceed the configured speed limit.

## Team Members

- Kukuh Agus Hermawan (24/533395/PA/22573)
- Aloysius Pijar Hutama Indrianto (24/534591/PA/22675)
- Mahardika Ramdhana (24/538247/PA/22831)
- Anders Emmanuel Tan (24/541351/PA/22964)
- Evan Razzan Adytaputra (24/545257/PA/23166)

## Method

The system pipeline consists of:

1. Extracting frames from CCTV videos.
2. Annotating vehicle objects using Roboflow.
3. Training a YOLO11n vehicle detection model.
4. Detecting vehicles in CCTV video frames.
5. Tracking detected vehicles using ByteTrack.
6. Selecting the road Region of Interest (ROI).
7. Transforming the ROI into Bird's Eye View using homography.
8. Estimating vehicle speed in km/h.
9. Detecting and saving speed violation evidence.

## Dataset

The dataset is based on CCTV traffic recordings from **Dinas Perhubungan Daerah Istimewa Yogyakarta (Dishub DIY)** at the **Maguwo T-junction, Yogyakarta**. The reference footage covers 6 hours of traffic on June 10, 2025, including morning, midday, and evening conditions.

The annotated dataset was prepared using **Roboflow**.

Dataset information:

- Dataset name: `CCTV Vehicle Detection Comvis`
- Version: `v1`
- Export date: May 19, 2026
- Total images: 1421 images
- Annotation format: YOLOv8
- Image size: 640 × 640 pixels
- License: CC BY 4.0

Roboflow preprocessing:

- Auto-orientation
- Resize to 640 × 640 pixels

Roboflow augmentation:

- Random brightness adjustment between -14% and +14%
- Random Gaussian blur between 0 and 1 pixels

Dataset access:

[Google Drive Dataset](https://drive.google.com/drive/folders/1IublVv7SSChvKHlX4PQ7xrLaGVNv9GmK?usp=sharing)

Main video files:

- `pagisiang.3gp`
- `siangmalam.3gp`
- `pagisiangtest.3gp`
- `siangmalamtest.3gp`

## Trained Model

Training results are stored in:

`Hasil_Training_CCTV/`

Available model folders:

- `Model_V1`
- `Model_V12`
- `Model_V13`

The final system uses:

`Hasil_Training_CCTV/Model_V13/weights/best.pt`

## Repository Contents

- `1_extractframe.py`  
  Extracts frames from CCTV videos.

- `2_trainyolo.py`  
  Trains the YOLO11n vehicle detection model using the Roboflow dataset.

- `3_testyolo.py`  
  Tests the trained YOLO model on CCTV footage.

- `4_speedviolationdetection.py`  
  Main script for speed violation detection using YOLO11n, ByteTrack, and homography.

- `Hasil_Training_CCTV/`  
  Stores trained model results.

- `data/`  
  Stores CCTV video files.

## Output

The system displays detected vehicles with bounding boxes, tracking IDs, estimated speeds, violation status, and Bird's Eye View visualization. When a vehicle exceeds the speed limit, the system marks it as a violation and saves the evidence image.

## Course Information

Final Project for **Computer Vision and Image Analysis**  
Department of Computer Science and Electronics  
Faculty of Mathematics and Natural Sciences  
Universitas Gadjah Mada  
Yogyakarta, 2026
