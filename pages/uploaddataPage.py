# uploaddatapage.py - Upload Data Page with Form Input
"""
Streamlit Upload Data Page for Alzheimer's Disease Prediction

This module allows users to input patient data for analysis:

- **Clinical Data**: Enter patient information through an interactive form for feature-based predictions.
- **MRI Data**: Upload one or multiple MRI brain scans for deep learning–based prediction.
  The system uses a CNN (e.g., InceptionV3) to analyze MRI scans and provides interpretable
  visual insights via Grad-CAM, highlighting brain regions most relevant to the prediction.

User input is processed, predictions are generated, and interpretability outputs
are displayed. Users can navigate to the corresponding dashboards for detailed results.
"""

# ------------------------------
# 📦 Core imports
# ------------------------------
import os
import warnings
from datetime import datetime

# ------------------------------
# 📊 Data and scientific libraries
# ------------------------------
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path

# ------------------------------
# 🧪 Machine learning & deep learning
# ------------------------------
from tensorflow.keras.models import load_model
import shap

# ------------------------------
# 🖼️ Image processing
# ------------------------------
import cv2
from PIL import Image

# ------------------------------
# 📈 Visualization
# ------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------
# 🌐 Streamlit & extras
# ------------------------------
import streamlit as st

# ------------------------------
# ⚙️ System and import utilities
# ------------------------------
import importlib.util

# ------------------------------
# 🔕 Suppress warnings
# ------------------------------
warnings.filterwarnings('ignore')
from style import *
from alzheimers_db_setup import AlzheimerPredictionStorage

# Add these imports at the top of your file
import urllib.request
import ssl

BASE_DIR = Path("/tmp/alzheimer_app")
BASE_DIR.mkdir(exist_ok=True, parents=True)

MODEL_DIR = BASE_DIR / "alzheimers_model_files"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

IMG_SIZE = 331
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

def download_models_from_github():
    """Download model files from GitHub repository if they don't exist locally"""
    
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/sv3112/Alzheimer_AI_Diagnosis_Dashboard/main/alzheimers_model_files"
    
    MODEL_FILES = [
        'alzheimers_best_model.pkl',
        'alzheimers_preprocessor_top10.pkl',
        'alzheimers_top10_features.pkl',
        'alzheimers_shap_explainer.pkl',
        'alzheimers_feature_names_processed.pkl'
    ]
    
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    ssl_context = ssl.create_default_context()
    
    print(f"📥 Checking model files in: {MODEL_DIR}")
    
    for filename in MODEL_FILES:
        local_path = MODEL_DIR / filename
        
        if local_path.exists():
            print(f"✅ {filename} already exists")
            continue
        
        github_url = f"{GITHUB_RAW_URL}/{filename}"
        print(f"📥 Downloading {filename} from GitHub...")
        
        try:
            with urllib.request.urlopen(github_url, context=ssl_context) as response:
                with open(local_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {str(e)}")
            raise

    print("✅ All model files ready")


def download_utilities_from_github():
    """Download utility files from GitHub repository if they don't exist locally"""
    
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/sv3112/Alzheimer_AI_Diagnosis_Dashboard/main"
    
    UTILITY_FILES = [
        'shap_utils.py',
        'scorecam.py'
    ]
    
    BASE_DIR.mkdir(exist_ok=True, parents=True)
    ssl_context = ssl.create_default_context()
    
    print(f"📥 Checking utility files in: {BASE_DIR}")
    
    for filename in UTILITY_FILES:
        local_path = BASE_DIR / filename
        
        if local_path.exists():
            print(f"✅ {filename} already exists")
            continue
        
        github_url = f"{GITHUB_RAW_URL}/{filename}"
        print(f"📥 Downloading {filename} from GitHub...")
        
        try:
            with urllib.request.urlopen(github_url, context=ssl_context) as response:
                with open(local_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {str(e)}")
            if filename == 'scorecam.py':
                print(f"⚠️ ScoreCAM functionality will be unavailable")

@st.cache_resource
def load_utilities():
    """Load and cache SHAP utilities for explainability"""
    try:
        download_utilities_from_github()
        shap_utility_path = BASE_DIR / "shap_utils.py"
        
        print(f"Loading SHAP utilities from: {shap_utility_path}")
        
        if not shap_utility_path.exists():
            raise FileNotFoundError(f"shap_utils.py not found at {shap_utility_path}")
        
        spec = importlib.util.spec_from_file_location("shap_utility", str(shap_utility_path))
        shap_utility = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shap_utility)
        
        print("✅ SHAP utilities loaded successfully")
        return shap_utility
    except Exception as e:
        print(f"❌ Failed to load SHAP utilities: {str(e)}")
        st.error(f"❌ Error loading SHAP utilities: {str(e)}")
        return None

shap_utility = load_utilities()

if shap_utility is not None:
    create_shap_analysis_results = shap_utility.create_shap_analysis_results
else:
    st.warning("⚠️ SHAP utilities could not be loaded. Some features may be unavailable.")
    create_shap_analysis_results = None

try:
    scorecam_path = BASE_DIR / "scorecam.py"
    
    if scorecam_path.exists():
        spec = importlib.util.spec_from_file_location("scorecam", str(scorecam_path))
        scorecam_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorecam_module)
        
        ScoreCAMBrainAnalysis = scorecam_module.ScoreCAMBrainAnalysis
        SCORECAM_AVAILABLE = True
        print("✅ ScoreCAM imported successfully")
    else:
        raise ImportError("scorecam.py not found")
        
except ImportError as e:
    print(f"❌ Failed to import ScoreCAM: {e}")
    SCORECAM_AVAILABLE = False
    ScoreCAMBrainAnalysis = None

apply_custom_css()

st.set_page_config(
    page_title="Upload & Analyze Data - Alzheimer's Diagnosis AI", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'img_array' not in st.session_state:
    st.session_state.img_array = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'data_type' not in st.session_state:
    st.session_state.data_type = 'csv'

@st.cache_data
def create_hero_section():
    """Render hero section with title and subtitle"""
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">🧠 AI-Driven Alzheimer's Diagnosis</h1>
        <p class="hero-subtitle">Enter patient clinical data or upload imaging data and get clear AI-driven Alzheimer's insights — fast and reliable</p>
    </div>
    """, unsafe_allow_html=True)

create_hero_section()

st.markdown("""
<div style="text-align: center; margin: 1.5rem 0;">
    <h2 style="color: #222; margin-bottom: 0.8rem; font-size: 2rem; font-weight: 900;">
        🎯 Select Your Data Type to Analyze
    </h2>
    <p style="color: #555; font-size: 1.25rem; margin: 0 auto;  line-height: 1.5;">
        Harness the power of our AI models to generate precise, personalized insights quickly and easily.
    </p>
</div>
""", unsafe_allow_html=True)

current_data_type = st.session_state.get('data_type', 'csv')

col1, col2 = st.columns(2, gap="large")

with col1:
    clinical_button_type = "primary" if current_data_type == 'csv' else "secondary"
    
    if st.button(
        "📊 Clinical Data Analysis\n\n✨ Binary Classification\n🔍 SHAP Explainability\n🎯 95% Accuracy",
        key="csv_select",
        use_container_width=True,
        type=clinical_button_type,
        help="Enter clinical data through interactive form"
    ):
        st.session_state.data_type = 'csv'
        st.rerun()

with col2:
    brain_button_type = "primary" if current_data_type == 'image' else "secondary"
    
    if st.button(
        "🧠 Brain Scan Analysis\n\n✨ 4-Stage Classification\n🔍 Grad-CAM & Region Visualization\n🎯 95% Accuracy",
        key="img_select",
        use_container_width=True,
        type=brain_button_type,
        help="Analyze brain scan images"
    ):
        st.session_state.data_type = 'image'
        st.rerun()

st.markdown(f"""
<style>
    div[data-testid="column"]:nth-child(1) .stButton > button {{
        height: 250px;
        background: {'linear-gradient(135deg, #6B46C1 0%, #4C1D95 100%)' if st.session_state.data_type == 'csv' else 'white'};
        color: {'white' if st.session_state.data_type == 'csv' else '#333'};
        border: 3px solid #6B46C1;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        white-space: pre-line;
        line-height: 1.6;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    
    div[data-testid="column"]:nth-child(1) .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }}
    
    div[data-testid="column"]:nth-child(2) .stButton > button {{
        height: 250px;
        background: {'linear-gradient(135deg, #6B46C1 0%, #4C1D95 100%)' if st.session_state.data_type == 'image' else 'white'};
        color: {'white' if st.session_state.data_type == 'image' else '#333'};
        border: 3px solid #6B46C1;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        white-space: pre-line;
        line-height: 1.6;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    
    div[data-testid="column"]:nth-child(2) .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_csv_models():
    """Load all required CSV model files for clinical data analysis"""
    try:
        download_models_from_github()
        CSV_MODEL_PATH = MODEL_DIR
        
        print(f"Looking for models in: {CSV_MODEL_PATH}")
        
        model_path = CSV_MODEL_PATH / 'alzheimers_best_model.pkl'
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        preprocessor = joblib.load(CSV_MODEL_PATH / 'alzheimers_preprocessor_top10.pkl')
        top_features = joblib.load(CSV_MODEL_PATH / 'alzheimers_top10_features.pkl')
        explainer = joblib.load(CSV_MODEL_PATH / 'alzheimers_shap_explainer.pkl')
        feature_names = joblib.load(CSV_MODEL_PATH / 'alzheimers_feature_names_processed.pkl')
        
        print("✅ All models loaded successfully")
        return model, preprocessor, top_features, explainer, feature_names
    
    except Exception as e:
        print(f"❌ Detailed error loading CSV models: {str(e)}")
        st.error(f"❌ Error loading CSV models: {str(e)}")
        return None, None, None, None, None

storage = AlzheimerPredictionStorage()

@st.cache_resource
def load_alzheimer_model():
    """Load the Alzheimer's CNN classification model for MRI images"""
    try:
        if os.path.exists(IMAGE_MODEL_PATH):
            model = load_model(IMAGE_MODEL_PATH, compile=False)
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            _ = model.predict(dummy_input, verbose=0)
            
            return model
        else:
            st.error(f"Model file not found at: {IMAGE_MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(331, 331)):
    """Convert uploaded image to model-ready batch"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

st.markdown('<h2 class="section-header">📤 Enter Your Data</h2>', unsafe_allow_html=True)

# ------------------------------
# 🧩 Clinical Data Form Input Section
# ------------------------------
if st.session_state.data_type == 'csv':
    st.markdown("""
        <div class="upload-section">
            <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap;">
                <div class="upload-icon">📊</div>
                <div style="text-align: left; max-width: 500px;">
                    <h3 class="upload-title">Enter Clinical Data</h3>
                    <p class="upload-description">
                        Fill in the patient's clinical information to get immediate binary classification and personalized explainable AI insights.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("🤖 Initializing AI models and preparing analysis pipeline..."):
        model, preprocessor, top_features, explainer, feature_names = load_csv_models()
    
    if model is None:
        st.error("Failed to load models. Please check the configuration.")
    else:
        st.markdown('<h3 class="section-header">📝 Patient Information Form</h3>', unsafe_allow_html=True)
        
        # Create form for patient data entry
        with st.form("patient_data_form"):
            st.markdown("#### 👤 Patient Identification")
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input("Patient ID*", placeholder="e.g., P001", help="Unique identifier for the patient")
            
            
            st.markdown("#### 📊 Clinical Features")
            st.markdown("Please enter all required clinical measurements:")
            
            # Feature descriptions and reasonable ranges
            feature_info = {
                'Age': {'min': 0.0, 'max': 120.0, 'default': 65.0, 'step': 1.0, 'help': 'Patient age in years'},
                'MMSE': {'min': 0.0, 'max': 30.0, 'default': 24.0, 'step': 1.0, 'help': 'Mini-Mental State Examination score (0-30)'},
                'FunctionalAssessment': {'min': 0.0, 'max': 10.0, 'default': 7.0, 'step': 0.1, 'help': 'Functional assessment score (0-10)'},
                'ADL': {'min': 0.0, 'max': 10.0, 'default': 7.0, 'step': 0.1, 'help': 'Activities of Daily Living score (0-10)'},
                'MemoryComplaints': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Has memory complaints? (0=No, 1=Yes)'},
                'BehavioralProblems': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Has behavioral problems? (0=No, 1=Yes)'},
                'Confusion': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Shows confusion? (0=No, 1=Yes)'},
                'Disorientation': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Shows disorientation? (0=No, 1=Yes)'},
                'PersonalityChanges': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Has personality changes? (0=No, 1=Yes)'},
                'DifficultyCompletingTasks': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Has difficulty completing tasks? (0=No, 1=Yes)'},
                'Forgetfulness': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Shows forgetfulness? (0=No, 1=Yes)'},
                'SleepQuality': {'min': 0.0, 'max': 10.0, 'default': 7.0, 'step': 0.1, 'help': 'Sleep quality score (0-10)'},
                'Diabetes': {'min': 0.0, 'max': 1.0, 'default': 0.0, 'step': 1.0, 'help': 'Has diabetes? (0=No, 1=Yes)'},
            }
            
            # Create input fields for each feature in top_features
            feature_values = {}
            
            # Organize features into rows of 3 columns each
            num_cols = 3
            feature_list = list(top_features)
            
            for i in range(0, len(feature_list), num_cols):
                cols = st.columns(num_cols)
                for j, col in enumerate(cols):
                    if i + j < len(feature_list):
                        feature = feature_list[i + j]
                        with col:
                            # Get feature info or use defaults
                            info = feature_info.get(feature, {
                                'min': 0.0,
                                'max': 100.0,
                                'default': 0.0,
                                'step': 0.01,
                                'help': f'Enter value for {feature}'
                            })
                            
                            feature_values[feature] = st.number_input(
                                feature,
                                min_value=info['min'],
                                max_value=info['max'],
                                value=info['default'],
                                step=info['step'],
                                help=info['help']
                            )
            
            st.markdown("---")
            submitted = st.form_submit_button("🧠 Analyze Patient Data", use_container_width=True, type="primary")
            
            if submitted:
                # Validate required fields
                if not patient_id:
                    st.error("⚠️ Patient ID is required!")
                else:
                    with st.spinner("🔄 Running comprehensive AI analysis..."):
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.text("📊 Step 1/4: Preprocessing clinical data...")
                            progress_bar.progress(25)
                            
                            # Create DataFrame from form input
                            input_data = pd.DataFrame([feature_values])
                            X_preprocessed = preprocessor.transform(input_data)

                            status_text.text("🤖 Step 2/4: Generating AI predictions...")
                            progress_bar.progress(50)
                            prediction = model.predict(X_preprocessed)[0]
                            probability = model.predict_proba(X_preprocessed)[0, 1]

                            status_text.text("🔍 Step 3/4: Computing SHAP explanations...")
                            progress_bar.progress(75)
                            fresh_explainer = shap.TreeExplainer(model)
                            shap_values = fresh_explainer.shap_values(X_preprocessed)

                            status_text.text("📋 Step 4/4: Generating comprehensive report...")
                            progress_bar.progress(90)

                            # Create prediction data matching database schema
                            prediction_data = {
                                'Patient_ID': patient_id,
                                'Predicted_Diagnosis': 'Alzheimer\'s Disease' if prediction == 1 else 'No Alzheimer\'s',
                                'Prediction_Probability': float(probability),
                                'Prediction_Confidence': float(probability),
                                'model_name': 'CatBoost',
                                'model_version': 'v1'
                            }
                            
                            # Add processed feature values (these match top_features)
                            for feature, value in feature_values.items():
                                prediction_data[feature] = float(value)

                            # Store in database FIRST before SHAP analysis
                            storage.store_individual_prediction(
                                prediction_data=prediction_data,
                                model_name='CatBoost',
                                model_version='v1'
                            )

                            # Create SHAP results (this may add extra columns we don't need for DB)
                            shap_results = create_shap_analysis_results(
                                shap_values=shap_values,
                                predictions=np.array([prediction]),
                                probabilities=np.array([probability]),
                                feature_names=feature_names,
                                actual_labels=None,
                                data=input_data
                            )

                            progress_bar.progress(100)
                            status_text.empty()

                            # Display results
                            csv_title = "🎉 Clinical Data Analysis Completed!"
                            csv_desc = f"Patient {patient_id} has been analyzed successfully. Prediction: {'Alzheimer\'s Detected' if prediction == 1 else 'No Alzheimer\'s Detected'} (Confidence: {probability*100:.1f}%)"
                            st.markdown(success_message(csv_title, csv_desc), unsafe_allow_html=True)
                            
                            # Display key metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Patient ID", patient_id)
                            with col2:
                                st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
                            with col3:
                                st.metric("Confidence", f"{probability*100:.1f}%")

                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)

# [Rest of the MRI image upload code remains the same as in original]
# ... (keeping the image upload section unchanged)

def create_navigation_section():
    """Enhanced navigation section with Home and Dashboard buttons"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            if st.button("🏠 Home", key="main_upload_btn", use_container_width=True, 
                        help="Return to home page"):
                st.switch_page("home.py")
        
        with subcol2:
            if st.session_state.data_type == 'csv':
                if st.button("📊 Clinical Dashboard", key="main_dashboard_btn", use_container_width=True,
                            help="View existing predictions and analytics dashboard"):
                    st.switch_page("pages/ClinicalDashboardPage.py")
            else:
                if st.button("🧠 MRI Dashboard", key="main_mri_dashboard_btn", use_container_width=True,
                            help="View MRI scan predictions and analytics dashboard"):
                    st.switch_page("pages/MRIDashboardPage.py")

create_navigation_section()
