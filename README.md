# CVL_FinalProject

Final project repository for UGM's Computer Vision and Image Analysis course.

## Project Title

**Vehicle Speed Limit Violation Detection in CCTV Recordings on Protocol Roads Using YOLO11n and ByteTrack**

**Indonesian Title:**  
**Deteksi Pelanggaran Batas Kecepatan Kendaraan pada Rekaman CCTV Jalan Protokol Menggunakan YOLO11n dan ByteTrack**

## Overview

This project builds a computer vision system to detect vehicle speed limit violations from CCTV recordings. The system uses YOLO11n for vehicle detection, ByteTrack for vehicle tracking, and homography-based Bird's Eye View (BEV) transformation for speed estimation.

## Team Members

- Aloysius Pijar Hutama Indrianto (24/534591/PA/22675) 
- Anders Emmanuel Tan (24/541351/PA/22964) 
- Evan Razzan Adytaputra (24/545257/PA/23166) 
- Kukuh Agus Hermawan (24/533395/PA/22573) 
- Mahardika Ramadhana (24/538247/PA/22831) 

## System Pipeline

1. Extract frames from CCTV videos.
2. Annotate vehicles using Roboflow.
3. Train YOLO11n vehicle detection model.
4. Detect vehicles in CCTV frames.
5. Track vehicles using ByteTrack.
6. Apply ROI and homography transformation.
7. Estimate vehicle speed.
8. Detect speed limit violations.
9. Save violation evidence.

## Dataset

The dataset is based on CCTV traffic recordings from Dinas Perhubungan Daerah Istimewa Yogyakarta (Dishub DIY) at the Bandara Adisucipto T-junction, Yogyakarta.

Dataset information:

- Source: Dishub DIY CCTV traffic recording
- Recording date: June 10, 2025
- Duration: approximately 6 hours
- Vehicle classes: motorcycle, car, bus, and truck
- Annotation platform: Roboflow
- Annotation format: YOLOv8
- Original frame size: 1280 × 960 pixels
- Training image size: 640 × 640 pixels

Dataset access:

[Google Drive Dataset](https://drive.google.com/drive/folders/1IublVv7SSChvKHlX4PQ7xrLaGVNv9GmK?usp=sharing)


## Output

The system displays detected vehicles with bounding boxes, tracking IDs, estimated speeds, violation status, and Bird's Eye View visualization. When a vehicle exceeds the speed limit, the system marks it as a violation and saves the evidence image. The system also capture the video recording into a .mp4 file.

Demo video:
[Google Drive Demo Video](https://drive.google.com/drive/folders/19Uu5CYzMat72s_ja8aCx2Wy2-3ZsVhmp?usp=sharing)

## Course Information

Final Project for Computer Vision and Image Analysis  
Department of Computer Science and Electronics  
Faculty of Mathematics and Natural Sciences  
Universitas Gadjah Mada  
Yogyakarta, 2026
