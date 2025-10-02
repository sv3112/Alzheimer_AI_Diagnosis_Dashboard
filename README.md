# 🧠 Dual-Path Explainable AI Dashboard for Alzheimer’s Diagnosis

A modular AI framework for early diagnosis of **Alzheimer’s Disease (AD)** using dual-path analysis of structured clinical data and MRI scans. The system provides interpretable predictions through an interactive Streamlit dashboard powered by **SHAP** and **CAM** visualizations.

---
## 📖 Overview

This project integrates **machine learning** and **deep learning** approaches into a single explainable framework:

### 🔬 Clinical Data Path
- Machine learning–based predictions from structured patient data  
- **SHAP explainability** for global and individual feature importance  
- Patient comparison and **what-if analysis** for sensitivity testing  

### 🧬 MRI Image Path
- Deep learning predictions using **InceptionV3 transfer learning**  
- **Score-CAM** highlighting discriminative brain regions  
- Robust computer vision pipeline for neuroimaging analysis  

### 📊 Interactive Dashboards
- Real-time upload: CSV (clinical data) / MRI scans (images)  
- Predictions with explainability and patient-specific recommendations  
- Persistent data storage with **SQLite database**  
- Modular navigation: Home, Upload, Clinical Dashboard, MRI Dashboard  

---
## 📂 Dataset Download
Before running the pipelines, download the datasets from Kaggle:
- **Clinical Dataset (CSV)** -> https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset 
- **MRI Dataset (Images)** -> https://www.kaggle.com/datasets/muhammadbakhsh277/balanced-dataset-of-alzheimers-disease 

---
## 🚀 Key Features
- ✨ **Dual-Modal Analysis** — Clinical + MRI independent pipelines  
- 🔍 **Explainable AI** — SHAP (clinical) + CAM (MRI)  
- 📱 **Interactive Dashboard** — Streamlit-based web app  
- 🧪 **What-If Analysis** — Adjust features, test prediction sensitivity  
- 💾 **Persistent Storage** — Centralized SQLite database  
- 🏥 **Clinical Integration** — Patient-level insights + recommendations  

---
## 📋 Prerequisites
- Python **3.10+**  
- Virtual environment recommended  

---
## ⚙️ Installation

Install dependencies:

pip install -r requirements.txt

---
## ⚙️ Configuration

Update the input/output paths in the respective files before running.

**Clinical Classification (`Alzheimer_Clinical_BinaryClassification.ipynb`)** 

input_path = "/path/to/alzheimers_disease_data.csv"
output_folder = "/path/to/alzheimers_model_files"

**MRI Classification (`Alzheimer_MRIImage_MultiClassification.ipynb`)**

DATA_DIR = "/path/to/MRI_Image_Data"
OUTPUT_DIR = "/path/to/output_results"

**Database (`alzheimers_db_setup.py`)**

base_dir = "/path/to/alzheimer_predictions.db"

**Dashboard Upload Page (`uploaddatapage.py`)**

CSV_MODEL_FOLDER = "/path/to/alzheimers_model_files"
IMAGE_MODEL_FOLDER = "/path/to/alzheimer_model_4class.keras"
SHAP_UTILITY_PATH = "/path/to/shap_utils.py"
SCORECAM_PATH = "/path/to/scorecam.py"

**Clinical Dashboard (`ClinicalDashboardPage.py`)**

model_folder = "/path/to/alzheimers_model_files"

**MRI Dashboard (`MRIDashboardPage.py`)**

sys.path.append("/path/to/project/root")

---
### 🏃‍♂️ Usage

**Database Setup**

python alzheimers_db_setup.py

**Train Models (Jupyter Notebooks)**

jupyter notebook Alzheimer_Clinical_BinaryClassification.ipynb
jupyter notebook Alzheimer_MRIImage_MultiClassification.ipynb

**Launch Dashboard**

streamlit run homepage.py

**Navigate Dashboard**

Homepage → Project overview

Upload Page → Upload clinical CSV or MRI images

Clinical Dashboard → Predictions + SHAP analysis

MRI Dashboard → Predictions + ScoreCAM visualizations

---
### 📂 Project Structure

alzheimer-ai-dashboard/
│
├── 📊 Jupyter Notebooks (Development & Training)
│   ├── Alzheimer_Clinical_BinaryClassification.ipynb    # Clinical ML pipeline
│   └── Alzheimer_MRIImage_MultiClassification.ipynb     # MRI CNN pipeline
│
├── 🖥️ Streamlit Dashboard Components
│   ├── homepage.py                                     # Landing page
│   ├── pages/
│       ├── uploaddataPage.py                               # File upload interface
│       ├── ClinicalDashboardPage.py                        # Clinical predictions & SHAP
│       └── MRIDashboard.py                                 # MRI predictions & CAM
│
├── 🔧 Core Utilities & Processing
│   ├── alzheimers_db_setup.py                          # Database setup & operations
│   ├── shap_utils.py                                   # SHAP explainability utilities
│   ├── scorecam.py                                     # Score-CAM/Grad-CAM utilities
│   ├── clinical_explanations.py                       # Clinical recommendations
│   └── style.py                                        # Dashboard styling
│
├── 📂 Data Files
│   ├── alzheimers_disease_data.csv                     # Clinical training dataset
│   └── MRI_Image_Data/                                 # MRI image dataset
│       ├── MildDemented/
│       ├── ModerateDemented/
│       ├── NonDemented/
│       └── VeryMildDemented/
│
├── 📦 Configuration
│   ├── requirements.txt                                # Python dependencies
│   └── README.md                                       # Project documentation
│
└── 🤖 Generated Models (After Training)
    ├── /alzheimers_model_files/                        # Clinical model storage
    └── /output_results/                                # MRI model outputs

---
### 📌 Notes

Update all hardcoded file paths before deployment.

Models and data can be stored locally or hosted (e.g., GitHub LFS, Google Drive).

SQLite ensures all predictions and interpretability results are tracked.