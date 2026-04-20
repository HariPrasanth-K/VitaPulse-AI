# VitaPulse AI  
### AI-Powered Remote Physiological Monitoring from Facial Video

---

## Overview

VitaPulse AI is a cloud-native, deep learning-based remote photoplethysmography (rPPG) system designed to extract physiological signals from facial video data.

The system estimates key health indicators, including heart rate, heart rate variability, respiration rate, blood pressure, oxygen saturation, stress index, hemoglobin, and HbA1c. It is designed for scalable deployment using AWS Lambda with Docker containers, enabling real-time inference from videos stored in Amazon S3.

---

## Key Features

- Video processing from Amazon S3  
- Multi-model rPPG ensemble inference  
- Heart rate estimation using FFT and peak detection methods  
- Heart rate variability (SDNN) computation  
- Respiration rate estimation  
- Blood pressure prediction using derived models  
- Oxygen saturation, hemoglobin, and HbA1c estimation  
- Face detection and frame preprocessing pipeline  
- Support for multiple deep learning architectures including PhysFormer, DeepPhys, EfficientPhys, TSCAN, RhythmFormer, PhysNet, FactorizePhys, PhysMamba, BigSmall, and iBVP variants  
- Fully serverless deployment using AWS Lambda  
- Memory-optimized inference for large-scale models  

---

## System Architecture

S3 Video Upload  
→ API Gateway Trigger  
→ AWS Lambda (Docker Container)  
→ Face Extraction and Frame Preprocessing  
→ Tensor Conversion (PyTorch)  
→ Multi-Model rPPG Inference Engine  
→ Signal Processing Layer  
   - FFT-based heart rate estimation  
   - Peak-based heart rate and HRV estimation  
→ Vital Signs Computation Engine  
→ PostgreSQL Database Storage  
→ JSON API Response  

---

## Project Structure
├── lambda_function.py # AWS Lambda entry point

├── Dockerfile # Container configuration for AWS Lambda

├── requirements.txt # Project dependencies

├── serverless.yml # Deployment configuration


├── lib/

│ ├── rppg_pipeline.py # Core inference pipeline

│ ├── rppg_inference.py # Signal processing and metric computation

│ ├── rppg_models.py # Deep learning model implementations

│ ├── rppg_video.py # Video loading and preprocessing

│ ├── helper.py # API helper functions

│ ├── util.py # Database utilities

│ └── request_data.py # AWS Lambda request handler


---

## How It Works

1. A facial video is uploaded to Amazon S3  
2. API Gateway triggers AWS Lambda function  
3. The video is decoded into frames  
4. Facial regions are detected and extracted  
5. Frames are converted into PyTorch tensors  
6. Multiple rPPG models process the input sequentially  
7. Physiological signals are extracted from deep features  
8. Signal processing computes:
   - Heart rate using FFT and peak detection  
   - Heart rate variability (SDNN)  
   - Respiration rate  
9. Final physiological metrics are generated  
10. Results are stored in PostgreSQL and returned via API response  

---

## Why VitaPulse AI is Unique

- Multi-model ensemble approach for rPPG estimation  
- Combination of deep learning and classical signal processing  
- Fully serverless architecture using AWS Lambda  
- Optimized for CPU-based inference at scale  
- Robust performance on real-world noisy video inputs  
- Generates clinically relevant physiological insights  

---

## Future Improvements

- GPU-based inference using AWS SageMaker  
- Real-time streaming support using WebRTC  
- Mobile application integration  
- Model quantization for faster inference  
- Expansion of clinical validation datasets  

---

## Requirements

- Python 3.11  
- PyTorch (CPU version)  
- OpenCV (headless)  
- AWS Lambda and API Gateway  
- Amazon S3  
- Docker and AWS ECR  
- PostgreSQL  

