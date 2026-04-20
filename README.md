# VitaPulse AI  
### AI-Powered Remote Physiological Monitoring from Facial Video

---

##  Overview

**VitaPulse AI** is a cloud-native, deep learning-based remote photoplethysmography (rPPG) system that extracts human physiological signals from facial video data.

It estimates key vital parameters such as:
- Heart Rate (HR)
- Heart Rate Variability (HRV)
- Respiration Rate (RR)
- Blood Pressure (SYS/DIA)
- Oxygen Saturation (SpO₂)
- Stress Index
- Hemoglobin (Hb)
- HbA1c

The system is designed for **serverless deployment on AWS Lambda using Docker containers**, enabling scalable, real-time health inference from video streams stored in Amazon S3.

---

##  Key Features

- 📹 Video processing from AWS S3
- 🧠 Multi-model rPPG ensemble inference
- ❤️ Dual HR estimation (FFT + Peak detection)
- 📊 HRV (SDNN) computation
- 🌬️ Respiration rate estimation
- 🩸 Blood pressure prediction (derived model)
- 🧬 SpO₂, Hb, HbA1c estimation
- 🧑 Face detection + frame preprocessing pipeline
- ⚙️ Supports 10+ deep learning architectures:
  - PhysFormer
  - DeepPhys
  - EfficientPhys
  - TSCAN
  - RhythmFormer
  - PhysNet
  - FactorizePhys
  - PhysMamba
  - BigSmall
  - iBVP variants
-  Fully deployed on AWS Lambda (container-based)
-  Memory-optimized inference for large models

---

##  System Architecture
S3 Video Upload
↓
API Gateway Trigger
↓
AWS Lambda (Docker Container)
↓
Face Extraction & Frame Preprocessing
↓
Tensor Conversion (PyTorch)
↓
Multi-Model rPPG Inference Engine
↓
Signal Processing Layer
├── FFT-based HR estimation
└── Peak-based HR/HRV estimation
↓
Vital Signs Computation Engine
↓
PostgreSQL Database Storage
↓
JSON API Response


---

##  Project Structure
├── lambda_function.py # AWS Lambda entry point
├── Dockerfile # Container definition for Lambda
├── requirements.txt # Python dependencies
├── serverless.yml # Deployment configuration
│
├── lib/
│ ├── rppg_pipeline.py # Core inference pipeline
│ ├── rppg_inference.py # Signal processing & metrics
│ ├── rppg_models.py # All rPPG deep learning models
│ ├── rppg_video.py # Video loading & preprocessing
│ ├── helper.py # API helpers (trigger/results)
│ ├── util.py # Database utilities
│ └── request_data.py # Lambda request parsing


---

##  How It Works

1. A user uploads a facial video to AWS S3  
2. API Gateway triggers AWS Lambda  
3. Video is decoded into frames  
4. Face region is extracted and cleaned  
5. Frames are converted into PyTorch tensors  
6. Multiple rPPG models run sequentially  
7. Raw signals are extracted from deep features  
8. Signal processing computes:
   - Heart Rate (FFT + Peak)
   - HRV (SDNN)
   - Respiration Rate  
9. Final physiological metrics are generated  
10. Results are stored in PostgreSQL and returned via API

##  Why VitaPulse AI is Unique

- First-class multi-model rPPG ensemble system
- Combines deep learning + classical signal processing
- Fully serverless medical AI pipeline (AWS Lambda)
- Optimized for CPU inference at scale
- Robust to noisy real-world video inputs
- Produces clinically relevant physiological metrics

##  Future Improvements

- GPU-based inference optimization (AWS SageMaker)
- Real-time streaming support (WebRTC)
- Mobile app integration
- Model quantization for faster Lambda execution
- Clinical validation dataset expansion

##  Requirements

- Python 3.11  
- PyTorch (CPU)  
- OpenCV (headless)  
- AWS Lambda + API Gateway  
- Amazon S3  
- Docker + ECR  
- PostgreSQL  
