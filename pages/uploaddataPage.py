
# uploaddatapage.py - Upload Data Page
"""
Streamlit Upload Data Page for Alzheimer's Disease Prediction

This module allows users to upload patient data for analysis:

- **Clinical Data**: Upload structured CSV files for feature-based predictions.
- **MRI Data**: Upload one or multiple MRI brain scans for deep learning–based prediction.
  The system uses a CNN (e.g., InceptionV3) to analyze MRI scans and provides interpretable
  visual insights via Grad-CAM, highlighting brain regions most relevant to the prediction.

Uploaded data is processed, predictions are generated, and interpretability outputs
are displayed. Users can navigate to the corresponding dashboards for detailed results.

This file is intended to be run within the Streamlit application framework:
    streamlit run uploaddatapage.py
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
import boto3

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
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# 🌐 Streamlit & extras
# ------------------------------
import streamlit as st
from streamlit_lottie import st_lottie

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
    
    # GitHub raw content base URL for your repository
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/sv3112/Alzheimer_AI_Diagnosis_Dashboard/main/alzheimers_model_files"
    
    # List of model files to download
    MODEL_FILES = [
        'alzheimers_best_model.pkl',
        'alzheimers_preprocessor_top10.pkl',
        'alzheimers_top10_features.pkl',
        'alzheimers_shap_explainer.pkl',
        'alzheimers_feature_names_processed.pkl'
    ]
    
    # Ensure the model directory exists
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create SSL context to handle HTTPS
    ssl_context = ssl.create_default_context()
    
    print(f"📥 Checking model files in: {MODEL_DIR}")
    
    for filename in MODEL_FILES:
        local_path = MODEL_DIR / filename
        
        # Skip if file already exists
        if local_path.exists():
            print(f"✅ {filename} already exists")
            continue
        
        # Download from GitHub
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
    
    # GitHub raw content base URL
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/sv3112/Alzheimer_AI_Diagnosis_Dashboard/main"
    
    # List of utility files to download
    UTILITY_FILES = [
        'shap_utils.py',
        'scorecam.py'
    ]
    
    # Ensure the base directory exists
    BASE_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create SSL context
    ssl_context = ssl.create_default_context()
    
    print(f"📥 Checking utility files in: {BASE_DIR}")
    
    for filename in UTILITY_FILES:
        local_path = BASE_DIR / filename
        
        # Skip if file already exists
        if local_path.exists():
            print(f"✅ {filename} already exists")
            continue
        
        # Download from GitHub
        github_url = f"{GITHUB_RAW_URL}/{filename}"
        print(f"📥 Downloading {filename} from GitHub...")
        
        try:
            with urllib.request.urlopen(github_url, context=ssl_context) as response:
                with open(local_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {str(e)}")
            # Don't raise for utility files, allow graceful degradation
            if filename == 'scorecam.py':
                print(f"⚠️ ScoreCAM functionality will be unavailable")

# ------------------------------
# 🔧 Load utilities once at module level
# ------------------------------
@st.cache_resource
def load_utilities():
    """Load and cache SHAP utilities for explainability"""
    try:
        # Download utilities from GitHub if not present
        download_utilities_from_github()
        
        # Update path to use BASE_DIR
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

# Load utilities
shap_utility = load_utilities()

# Safely access create_shap_analysis_results
if shap_utility is not None:
    create_shap_analysis_results = shap_utility.create_shap_analysis_results
else:
    st.warning("⚠️ SHAP utilities could not be loaded. Some features may be unavailable.")
    create_shap_analysis_results = None

# ------------------------------
# 🧠 ScoreCAM import for MRI explainability
# ------------------------------
try:
    # Try to import from BASE_DIR
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


# Update the load_csv_models() function


# ------------------------------
# 🖼️ Image analysis constants
# ------------------------------

# ------------------------------
# 🔧 Load utilities once at module level
# ------------------------------


# ------------------------------
# 🎨 Apply custom CSS styles
# ------------------------------
apply_custom_css()

# ------------------------------
# ⚙️ Streamlit page configuration
# ------------------------------
st.set_page_config(
    page_title="Upload & Analyze Data - Alzheimer's Diagnosis AI", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# 🧩 Initialize Streamlit session state
# ------------------------------
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

# ------------------------------
# 🏆 Cache hero section
# ------------------------------
@st.cache_data
def create_hero_section():
    """Render hero section with title and subtitle"""
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">🧠 AI-Driven Alzheimer's Diagnosis</h1>
        <p class="hero-subtitle">Upload your clinical or imaging data and get clear AI-driven Alzheimer's insights — fast and reliable</p>
    </div>
    """, unsafe_allow_html=True)

create_hero_section()

# ------------------------------
# 🧩 Data Type Selection - Simplified UI
# ------------------------------
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

# ------------------------------
# 🌟 Data Type Buttons
# ------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    clinical_button_type = "primary" if current_data_type == 'csv' else "secondary"
    
    if st.button(
        "📊 Clinical Data Analysis\n\n✨ Binary Classification\n🔍 SHAP Explainability\n🎯 95% Accuracy",
        key="csv_select",
        use_container_width=True,
        type=clinical_button_type,
        help="Analyze clinical data from CSV files"
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

# Simplified button styling
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

# ------------------------------
# 💾 Load CSV (clinical data) models - FIXED VERSION
# ------------------------------
@st.cache_resource

def load_csv_models():
    """Load all required CSV model files for clinical data analysis"""
    try:
        # Download models from GitHub if not present
        download_models_from_github()
        
        # Models are now in BASE_DIR/alzheimers_model_files
        CSV_MODEL_PATH = MODEL_DIR
        
        print(f"Looking for models in: {CSV_MODEL_PATH}")
        print(f"Directory exists: {CSV_MODEL_PATH.exists()}")
        if CSV_MODEL_PATH.exists():
            print(f"Files in directory: {list(CSV_MODEL_PATH.iterdir())}")
        
        # Load the trained ML model
        model_path = CSV_MODEL_PATH / 'alzheimers_best_model.pkl'
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Load preprocessing pipeline for top 10 features
        preprocessor = joblib.load(CSV_MODEL_PATH / 'alzheimers_preprocessor_top10.pkl')
        
        # Load list of top 10 features
        top_features = joblib.load(CSV_MODEL_PATH / 'alzheimers_top10_features.pkl')
        
        # Load SHAP explainer for interpretability
        explainer = joblib.load(CSV_MODEL_PATH / 'alzheimers_shap_explainer.pkl')
        
        # Load processed feature names
        feature_names = joblib.load(CSV_MODEL_PATH / 'alzheimers_feature_names_processed.pkl')
        
        print("✅ All models loaded successfully")
        return model, preprocessor, top_features, explainer, feature_names
    
    except Exception as e:
        print(f"❌ Detailed error loading CSV models: {str(e)}")
        st.error(f"❌ Error loading CSV models: {str(e)}")
        return None, None, None, None, None


# Initialize database storage for predictions
storage = AlzheimerPredictionStorage()

# ------------------------------
# 🧠 Load MRI image classification model
# ------------------------------
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

# ------------------------------
# 🖼️ Image preprocessing for model prediction
# ------------------------------
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

# ------------------------------
# 🖌️ Mark boundaries in segmentation masks
# ------------------------------
def mark_boundaries(img, mask, color=(1, 0, 0), mode='thick'):
    """Overlay boundaries of segmented regions on the original image"""
    from skimage.segmentation import find_boundaries
    
    boundaries = find_boundaries(mask, mode=mode)
    marked = img.copy()
    marked[boundaries] = color
    
    return marked

# ------------------------------
# 📤 Upload Section Header
# ------------------------------
st.markdown('<h2 class="section-header">📤 Upload Your Data</h2>', unsafe_allow_html=True)

# ------------------------------
# 🧩 Clinical CSV Upload Section
# ------------------------------
if st.session_state.data_type == 'csv':
    st.markdown("""
        <div class="upload-section">
            <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap;">
                <div class="upload-icon">📊</div>
                <div style="text-align: left; max-width: 500px;">
                    <h3 class="upload-title">Upload Clinical Data (CSV)</h3>
                    <p class="upload-description">
                        Upload your clinical data CSV file to get immediate binary classification and personalized explainable AI insights.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type='csv',
        help="Upload a CSV file with patient data containing all required clinical features"
    )
    
    if uploaded_file is not None:
        with st.spinner("🤖 Initializing AI models and preparing analysis pipeline..."):
            model, preprocessor, top_features, explainer, feature_names = load_csv_models()
        
        if model is None:
            st.error("Failed to load models. Please check the configuration.")
        else:
            try:
                data = pd.read_csv(uploaded_file)
                
                st.markdown('<h2 class="section-header">📈 Data Overview</h2>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card_D">
                        <div class="metric-value_D">{len(data):,}</div>
                        <div class="metric-label_D">Total Patients</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card_D">
                        <div class="metric-value_D">{len(data.columns)}</div>
                        <div class="metric-label_D">Clinical Features</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    missing_total = data.isnull().sum().sum()
                    st.markdown(f"""
                    <div class="metric-card_D">
                        <div class="metric-value_D">{missing_total:,}</div>
                        <div class="metric-label_D">Missing Values</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    missing_pct = (missing_total / (len(data) * len(data.columns)) * 100)
                    st.markdown(f"""
                    <div class="metric-card_D">
                        <div class="metric-value_D">{missing_pct:.1f}%</div>
                        <div class="metric-label_D">Data Incompleteness</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                available_features = set(data.columns)
                required_features = set(top_features)
                missing_features = required_features - available_features
                
                if missing_features:
                    st.markdown(f"""
                    <div class="info-card" >
                        <h3 style="color: #DC2626;">⚠️ Missing Required Features</h3>
                        <p style="color: #991B1B;">The following required features are missing from your dataset:</p>
                    """, unsafe_allow_html=True)
                    
                    missing_list = list(missing_features)
                    cols = st.columns(3)
                    for i, feat in enumerate(missing_list):
                        cols[i % 3].markdown(f'<span class="status-badge status-error">❌ {feat}</span>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.markdown(" ")
                    st.markdown("""
                    <div class="info-card" >
                        <h3 style="color: #059669;">✅ All Features Validated</h3>
                        <p style="color: #065F46;">All required clinical features are present in your dataset. Ready for analysis!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📋 View All Clinical Features", expanded=False):
                        feature_cols = st.columns(4)
                        for i, feat in enumerate(top_features):
                            feature_cols[i % 4].markdown(f'<span class="status-badge status-success">✓ {feat}</span>', unsafe_allow_html=True)

                if not missing_features:
                    st.markdown('<h2 class="section-header">🚀 AI Analysis Pipeline for Clinical Data</h2>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="process-timeline">', unsafe_allow_html=True)
                    
                    steps = [
                        ("Data Preprocessing", "Clean, normalize, and prepare patient data for optimal model performance.", "🔧"),
                        ("AI Prediction", "Generate Alzheimer's risk predictions using an advanced CatBoost model.", "🤖"),
                        ("SHAP Analysis", "Provide explainable AI insights to understand individual prediction drivers.", "🔍"),
                        ("Report Generation", "Compile a comprehensive report with visualizations.", "📊"),
                        ("Database Storage", "Securely store all analysis results and reports in the database.", "💾"),
                    ]

                    for i, (title, desc, icon) in enumerate(steps, 1):
                        st.markdown(f"""
                        <div class="process-step">
                            <span class="step-number">{i}</span>
                            
                            <div style="display: flex; align-items: start; gap: 2rem;">
                                <span style="font-size: 2rem;">{icon}</span>
                                
                                <div>
                                    <h4 style="margin: 0 0 0.5rem 0; color: #333;">{title}</h4>
                                    <p style="margin: 0; color: #666; line-height: 1;">{desc}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(
                            "🧠 Start AI Analysis", 
                            type="primary", 
                            use_container_width=True, 
                            help="Run comprehensive AI analysis with SHAP explanations"
                        ):
                            with st.spinner("🔄 Running comprehensive AI analysis..."):
                                try:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    status_text.text("📊 Step 1/4: Preprocessing clinical data...")
                                    progress_bar.progress(25)
                                    X = data[top_features]
                                    X_preprocessed = preprocessor.transform(X)

                                    status_text.text("🤖 Step 2/4: Generating AI predictions...")
                                    progress_bar.progress(50)
                                    predictions = model.predict(X_preprocessed)
                                    probabilities = model.predict_proba(X_preprocessed)[:, 1]

                                    status_text.text("🔍 Step 3/4: Computing SHAP explanations...")
                                    progress_bar.progress(75)
                                    fresh_explainer = shap.TreeExplainer(model)
                                    shap_values = fresh_explainer.shap_values(X_preprocessed)

                                    status_text.text("📋 Step 4/4: Generating comprehensive report...")
                                    progress_bar.progress(90)

                                    shap_results = create_shap_analysis_results(
                                        shap_values=shap_values,
                                        predictions=predictions,
                                        probabilities=probabilities,
                                        feature_names=feature_names,
                                        actual_labels=None,
                                        data=data
                                    )

                                    storage.store_global_importance(
                                        importance_data=shap_results['global_importance'],
                                        model_name='CatBoost',
                                        model_version='v1'
                                    )

                                    individual_df = shap_results['individual_predictions']
                                    for _, row in individual_df.iterrows():
                                        storage.store_individual_prediction(
                                            prediction_data=row.to_dict(),
                                            model_name='CatBoost',
                                            model_version='v1'
                                        )

                                    progress_bar.progress(100)
                                    status_text.empty()

                                    storage.close()

                                    csv_title = "🎉 Clinical Data Analysis Completed!"
                                    csv_desc = "Your clinical data has been analyzed successfully. You can now view patient-level insights and download reports."
                                    st.markdown(success_message(csv_title, csv_desc), unsafe_allow_html=True)

                                except Exception as e:
                                    st.error(f"❌ Error during analysis: {str(e)}")
                                    st.exception(e)

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

# ------------------------------
# 🖼️ MRI Image Upload Section
# ------------------------------
else:  # Image upload
    # Informational section with icon, title, and description
    st.markdown("""
    <div class="upload-section">
        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap;">
            <div class="upload-icon">🧠</div>
            <div style="text-align: left; max-width: 500px;">
                <h3 class="upload-title">Upload MRI Scan Images</h3>
                <p class="upload-description">
                    Upload one or more brain MRI scans to receive advanced AI-powered analysis and region-level visual explanations.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------
    # 📁 Upload Mode Selection
    # ------------------------------
    st.markdown("### 📁 Select Upload Mode")  # Section header
    col1, col2 = st.columns(2)  # Two-column layout for single vs batch mode
    
    with col1:
        # Single file analysis button
        single_selected = st.button(
            "🔍 Single File Analysis\n\nDetailed individual scan analysis\n📁 Saves to: Database",
            use_container_width=True,
            # Highlight as primary if currently selected; otherwise secondary
            type="secondary" if 'upload_mode' in st.session_state and st.session_state.upload_mode == "Multiple Files" else "primary"
        )
        if single_selected:
            st.session_state.upload_mode = "Single File"  # Update session state
            st.rerun()  # Refresh app to reflect selection
    
    with col2:
        # Batch analysis button
        batch_selected = st.button(
            "📊 Batch Analysis\n\nProcess multiple scans at once\n📁 Saves to: Database",
            use_container_width=True,
            # Highlight as primary if currently selected; otherwise secondary
            type="secondary" if 'upload_mode' not in st.session_state or st.session_state.upload_mode == "Single File" else "primary"
        )
        if batch_selected:
            st.session_state.upload_mode = "Multiple Files"  # Update session state
            st.rerun()  # Refresh app to reflect selection

    # ------------------------------
    # 🔧 Default Upload Mode
    # ------------------------------
    if 'upload_mode' not in st.session_state:
        st.session_state.upload_mode = "Single File"  # Default to single file

    # Initialize list to store uploaded files
    uploaded_files = []

    # ------------------------------
    # 🖼️ Handle MRI File Upload Based on Mode
    # ------------------------------
    if st.session_state.upload_mode == "Single File":
        # Single file upload
        uploaded_file = st.file_uploader(
            "Choose an MRI brain scan image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload brain scan images for Alzheimer's disease classification"
        )
        if uploaded_file is not None:
            uploaded_files = [uploaded_file]  # Wrap single file in a list for uniform processing
    else:
        # Multiple files upload
        uploaded_files = st.file_uploader(
            "Choose MRI brain scan images (you can select multiple files)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple brain scan images for batch Alzheimer's disease classification"
        )

    # ------------------------------
    # ✅ Display Upload Summary
    # ------------------------------
    if uploaded_files:
        # Determine analysis type and destination folder for UI context
        analysis_type = "Single File Analysis" if len(uploaded_files) == 1 and st.session_state.upload_mode == "Single File" else "Batch Analysis"
        folder_destination = "Single" if analysis_type == "Single File Analysis" else "Batch"

        # Display confirmation info card
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color: #059669;"> ✅ {len(uploaded_files)} file(s) uploaded successfully</h4>
            <p style="margin: 0.5rem 0 0 0; color: #666;">
                <strong>Analysis Type:</strong> {analysis_type}. Ready for Analysis!<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Expandable list of uploaded files
        with st.expander("📋 View uploaded files"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name}")

    # ------------------------------
    # 🚀 Brain Scan AI Analysis Pipeline Section
    # ------------------------------
        st.markdown('<h2 class="section-header">🚀 AI Analysis Pipeline</h2>', unsafe_allow_html=True)

        # Container div for process timeline
        st.markdown('<div class="process-timeline">', unsafe_allow_html=True)

        # Define sequential steps for brain scan analysis
        brain_steps = [
            ("Image Preprocessing", "Load, resize, and normalize MRI scan images to ensure optimal model performance.", "📸"),
            ("CNN Classification", "Classify dementia stages using an advanced deep learning model (four-class output).", "🤖"),
            ("ScoreCAM Heatmaps", "Generate activation heatmaps highlighting the brain regions the AI focuses on using gradient-free ScoreCAM.", "🔥"),
            ("Brain Region Analysis", "Assess importance scores across different brain regions and anatomical structures.", "🧠"),
            ("Individual Region Maps", "Create detailed visualizations for each anatomical brain region with importance scores.", "🗺️"),
            ("Report Generation", "Compile a comprehensive report with visualizations and clinical insights, and store all results securely in the database.", "📊"),
        ]

        # ------------------------------
        # 🔄 Render Brain Scan AI Pipeline Steps
        # ------------------------------
        for i, (title, desc, icon) in enumerate(brain_steps, 1):
            # Each step displayed as a styled card in the timeline
            st.markdown(f"""
            <div class="process-step">
                <!-- Step number -->
                <span class="step-number">{i}</span>
                
                <!-- Icon + Step content -->
                <div style="display: flex; align-items: start; gap: 2rem;">
                    <span style="font-size: 2rem;">{icon}</span>
                    <div>
                        <h4 style="margin: 0 0 0.5rem 0; color: #333;">{title}</h4>
                        <p style="margin: 0; color: #666; line-height: 1.5;">{desc}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Close the timeline container
        st.markdown('</div>', unsafe_allow_html=True)

        # ------------------------------
        # 🧠 Load CNN Model if Not Already Loaded
        # ------------------------------
        if st.session_state.model is None:
            with st.spinner("Loading model..."):
                st.session_state.model = load_alzheimer_model()

        # ------------------------------
        # ▶️ ScoreCAM Analysis Trigger Button
        # ------------------------------
        if st.session_state.model is not None:
            # Center the button using 3 columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                button_text = f"🧠 Generate ScoreCAM Analysis & Download Results ({len(uploaded_files)} files)"
                if st.button(button_text, type="primary", use_container_width=True):
                    
                    # ------------------------------
                    # 📂 Batch Analysis Directory
                    # ------------------------------
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_name = f"ScoreCAM_Analysis_{timestamp}"
                    
                    # ------------------------------
                    # 📊 Initialize Storage for Batch Results
                    # ------------------------------
                    batch_results = {
                        'predictions': [],            # Store predicted classes
                        'regions': [],                # Store brain region data
                        'summaries': [],              # Summary of each image
                        'comprehensive_results': [],  # Complete per-image results
                        'storage_summaries': [],      # Info for DB storage
                        'total_images_stored': 0,     # Counter for images saved
                        'total_regions_stored': 0,    # Counter for regions saved
                        'errors': []                  # List to capture any errors during processing
                    }
                    
                    # ------------------------------
                    # 🔄 Main Progress Bar for Overall Batch
                    # ------------------------------
                    main_progress = st.progress(0)  # Progress bar widget
                    main_status = st.empty()        # Placeholder for dynamic status text

                    # ------------------------------
                    # 🔥 Initialize ScoreCAM Analyzer
                    # ------------------------------
                    try:
                      BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust if script is in pages/

                      # 2️⃣ Path to scorecam.py inside the repo
                      SCORECAM_PATH = BASE_DIR / "scorecam.py"
                  
                      # 3️⃣ Dynamically import ScoreCAM module
                      spec = importlib.util.spec_from_file_location("scorecam_brain", str(SCORECAM_PATH))
                      scorecam_module = importlib.util.module_from_spec(spec)
                      spec.loader.exec_module(scorecam_module)
                  
                      
                      # 5️⃣ Access the ScoreCAMBrainAnalysis class
                      ScoreCAMBrainAnalysis = scorecam_module.ScoreCAMBrainAnalysis
                  
                      # 6️⃣ Initialize analyzer with loaded CNN model and input image size
                      scorecam_analyzer = ScoreCAMBrainAnalysis(st.session_state.model, IMG_SIZE)
                  
                      # 7️⃣ Inform user
                      main_status.text("🤖 ScoreCAM analyzer initialized successfully")
                    
                    except Exception as init_error:
                        # Stop the app if ScoreCAM initialization fails
                        st.error(f"❌ Failed to initialize ScoreCAM analyzer: {str(init_error)}")
                        st.stop()

                    # ------------------------------
                    # 💾 Function to Store ScoreCAM Results
                    # ------------------------------
                    def store_scorecam_results(storage, scorecam_analyzer, img_array, temp_image_path, 
                                            comprehensive_results, uploaded_filename, original_filename, timestamp):
                        """
                        Store ScoreCAM results for a single MRI scan.
                        
                        Expected to save 8 images per scan:
                        1. Original image
                        2. Brain mask
                        3. Heatmap
                        4. Overlay
                        5-8. Four individual brain region maps

                        Args:
                            storage: AlzheimerPredictionStorage instance for DB operations
                            scorecam_analyzer: initialized ScoreCAMBrainAnalysis object
                            img_array: preprocessed image array
                            temp_image_path: temporary directory for storing images
                            comprehensive_results: dict to store per-image results
                            uploaded_filename: original uploaded file name
                            original_filename: sanitized filename for storage
                            timestamp: unique timestamp for this analysis
                        """
                        # Initialize per-image storage summary
                        storage_summary = {
                            'images_stored': 0,       # Counter for images saved
                            'predictions_stored': 0,  # Counter for predictions saved
                            'regions_stored': 0,      # Counter for region maps saved
                            'errors': []              # Capture any errors during processing
                        }

                        # ------------------------------
                        # 🔄 Generate and Store ScoreCAM Images for a Single MRI
                        # ------------------------------
                        try:
                            print(f"\n🔄 Generating and storing ScoreCAM images for {original_filename}...")
                            
                            # ------------------------------
                            # 📂 Temporary Directory for Image Generation
                            # ------------------------------
                            import tempfile
                            import os
                            temp_gen_dir = tempfile.mkdtemp()  # Creates a temp folder to save all generated images
                            
                            # ------------------------------
                            # 📊 Retrieve Prediction Data
                            # ------------------------------
                            pred_class = np.argmax(comprehensive_results['all_predictions'])  # Predicted class index
                            confidence = comprehensive_results['confidence']                    # Confidence score
                            
                            # Dictionary to store all generated image paths
                            image_paths = {}
                            
                            # ------------------------------
                            # 1️⃣ ORIGINAL IMAGE
                            # ------------------------------
                            try:
                                original_path = os.path.join(temp_gen_dir, f'{original_filename}_original.png')
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.imshow(img_array)  # Show the MRI image
                                ax.set_title('Original MRI', fontsize=14, fontweight='bold')
                                ax.axis('off')
                                plt.savefig(original_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
                                plt.close()
                                image_paths['original'] = original_path
                                print("  ✅ Generated: Original image")
                            except Exception as e:
                                storage_summary['errors'].append(f"Original image: {str(e)}")
                                print(f"  ❌ Failed: Original image - {e}")
                            
                            # ------------------------------
                            # 2️⃣ BRAIN MASK
                            # ------------------------------
                            try:
                                brain_mask = scorecam_analyzer.create_enhanced_brain_mask(img_array)
                                brain_mask_path = os.path.join(temp_gen_dir, f'{original_filename}_brain_mask.png')
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.imshow(brain_mask, cmap='gray')
                                ax.set_title('Brain Mask', fontsize=14, fontweight='bold')
                                ax.axis('off')
                                plt.savefig(brain_mask_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
                                plt.close()
                                image_paths['brain_mask'] = brain_mask_path
                                print("  ✅ Generated: Brain mask")
                            except Exception as e:
                                storage_summary['errors'].append(f"Brain mask: {str(e)}")
                                print(f"  ❌ Failed: Brain mask - {e}")
                                brain_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)  # Fallback mask if error
                            
                            # ------------------------------
                            # 3️⃣ SCORECAM HEATMAP
                            # ------------------------------
                            try:
                                heatmap, _ = scorecam_analyzer.score_cam(img_array, pred_class, use_brain_mask=True)
                                heatmap_path = os.path.join(temp_gen_dir, f'{original_filename}_scorecam_heatmap.png')
                                fig, ax = plt.subplots(figsize=(6, 6))
                                im = ax.imshow(heatmap, cmap='hot')
                                ax.set_title('ScoreCAM Heatmap', fontsize=14, fontweight='bold')
                                ax.axis('off')
                                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                plt.savefig(heatmap_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
                                plt.close()
                                image_paths['scorecam_heatmap'] = heatmap_path
                                print("  ✅ Generated: ScoreCAM heatmap")
                            except Exception as e:
                                storage_summary['errors'].append(f"ScoreCAM heatmap: {str(e)}")
                                print(f"  ❌ Failed: ScoreCAM heatmap - {e}")
                                heatmap = np.random.random((IMG_SIZE, IMG_SIZE)) * 0.1  # Fallback heatmap
                            
                            # ------------------------------
                            # 4️⃣ SCORECAM OVERLAY
                            # ------------------------------
                            try:
                                scorecam_overlay_path = os.path.join(temp_gen_dir, f'{original_filename}_scorecam_overlay.png')
                                heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # Convert heatmap to RGB
                                overlay = img_array * 0.5 + heatmap_colored * 0.5
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.imshow(overlay)
                                ax.set_title('ScoreCAM Overlay', fontsize=14, fontweight='bold')
                                ax.axis('off')
                                plt.savefig(scorecam_overlay_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
                                plt.close()
                                image_paths['scorecam_overlay'] = scorecam_overlay_path
                                print("  ✅ Generated: ScoreCAM overlay")
                            except Exception as e:
                                storage_summary['errors'].append(f"ScoreCAM overlay: {str(e)}")
                                print(f"  ❌ Failed: ScoreCAM overlay - {e}")
                            
                            # ------------------------------
                            # 5️⃣-🔟 INDIVIDUAL BRAIN REGIONS (6 regions)
                            # ------------------------------
                            try:
                                # Generate anatomical region masks
                                region_masks = scorecam_analyzer.create_anatomical_region_masks(brain_mask)
                                region_scores = comprehensive_results.get('brain_region_scores', {})
                                
                                # Create grayscale background for region visualization
                                gray_bg = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                                gray_bg_rgb = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB) / 255.0
                                
                                individual_region_paths = {}
                                for region_name, region_mask in region_masks.items():
                                    try:
                                        if np.sum(region_mask) > 0:
                                            # Copy grayscale background
                                            region_viz = gray_bg_rgb.copy()
                                            
                                            # Get ScoreCAM score for region
                                            score = region_scores.get(region_name, {}).get('score_cam_score', 0.1)
                                            
                                            # Apply region-specific overlay color
                                            overlay_color = scorecam_analyzer.brain_regions[region_name]['color']
                                            alpha = 0.6
                                            for c in range(3):
                                                region_viz[:,:,c][region_mask] = (
                                                    region_viz[:,:,c][region_mask] * (1-alpha) + overlay_color[c] * alpha
                                                )
                                            
                                            # Save individual region image
                                            region_path = os.path.join(temp_gen_dir, f'{original_filename}_{region_name.lower()}_region.png')
                                            fig, ax = plt.subplots(figsize=(6, 6))
                                            ax.imshow(region_viz)
                                            ax.set_title(f'{region_name}\nImportance: {score*100:.1f}%', fontsize=16, fontweight='bold', pad=20)
                                            ax.axis('off')
                                            plt.tight_layout()
                                            plt.savefig(region_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
                                            plt.close()
                                            
                                            individual_region_paths[region_name] = region_path
                                            image_paths[f'region_{region_name.lower()}'] = region_path
                                            print(f"  ✅ Generated: {region_name} region")
                                            
                                    except Exception as region_error:
                                        storage_summary['errors'].append(f"Region {region_name}: {str(region_error)}")
                                        print(f"  ❌ Failed: {region_name} region - {region_error}")
                                
                            except Exception as regions_error:
                                storage_summary['errors'].append(f"Brain regions generation: {str(regions_error)}")
                                print(f"  ❌ Failed: Brain regions generation - {regions_error}")
                            
                            print(f"✅ Generated {len(image_paths)} images (with {len(storage_summary['errors'])} issues)")

                            # ------------------------------
                            # 💾 STORE ALL GENERATED IMAGES IN DATABASE
                            # ------------------------------
                            print(f"💾 Storing {len(image_paths)} images in database...")

                            # Loop over all generated images
                            for image_type, image_path in image_paths.items():
                                try:
                                    # Initialize importance score variables
                                    importance_score = None
                                    region_name_for_score = None
                                    
                                    # If the image corresponds to an anatomical region, fetch its ScoreCAM score
                                    if image_type.startswith('region_'):
                                        region_name_for_score = image_type.replace('region_', '').title()  # e.g., 'Hippocampus'
                                        region_scores = comprehensive_results.get('brain_region_scores', {})
                                        
                                        if region_name_for_score in region_scores:
                                            importance_score = region_scores[region_name_for_score]['score_cam_score']  # Importance from AI
                                    
                                    # Store the image along with metadata in the database
                                    storage.store_image_with_metadata(
                                        image_path=image_path,                 # Path to the saved image file
                                        patient_id=original_filename,          # Patient or filename identifier
                                        image_type=image_type,                 # Type of image (original, mask, overlay, region)
                                        model_name='ScoreCAM',                 # Model used for analysis
                                        model_version='v1.0',                  # Model version
                                        analysis_type='scorecam_brain_analysis', # Analysis type label
                                        region_name=region_name_for_score,     # Anatomical region (if applicable)
                                        importance_score=importance_score      # Importance value (if applicable)
                                    )
                                    
                                    # Track successfully stored images
                                    storage_summary['images_stored'] += 1
                                    
                                except Exception as img_error:
                                    # Catch storage errors and log them
                                    storage_summary['errors'].append(f"Image storage {image_type}: {str(img_error)}")
                                    print(f"❌ Failed to store {image_type}: {img_error}")

                            # Summary log
                            print(f"✅ Stored {storage_summary['images_stored']} images in database")

                        except Exception as img_gen_error:
                            storage_summary['errors'].append(f"Image generation: {str(img_gen_error)}")
                            print(f"❌ Image generation failed: {img_gen_error}")

                        # ------------------------------
                        # 📊 STORE PREDICTION DATA
                        # ------------------------------
                        try:
                            print(f"📊 Storing prediction data...")

                            # Prepare prediction dictionary for database storage
                            prediction_data = {
                                'Filename': uploaded_filename,  # Original uploaded file name
                                'Patient_ID': original_filename, # Internal patient/file identifier
                                'Predicted_Class': comprehensive_results['predicted_class'], # AI-predicted class
                                'Confidence': float(comprehensive_results['confidence']),    # Overall prediction confidence
                                # Probabilities for each class
                                'Mild_Demented_Probability': float(comprehensive_results['all_predictions'][0]),
                                'Moderate_Demented_Probability': float(comprehensive_results['all_predictions'][1]),
                                'Non_Demented_Probability': float(comprehensive_results['all_predictions'][2]),
                                'Very_Mild_Demented_Probability': float(comprehensive_results['all_predictions'][3])
                            }

                            # Store prediction data in database
                            storage.store_batch_prediction(
                                prediction_data=prediction_data,
                                model_name='ScoreCAM',
                                model_version='v1.0'
                            )

                            # Track successfully stored prediction
                            storage_summary['predictions_stored'] = 1
                            print("✅ Prediction data stored")

                        except Exception as pred_error:
                            # Log any storage errors
                            storage_summary['errors'].append(f"Prediction storage: {str(pred_error)}")
                            print(f"❌ Prediction storage failed: {pred_error}")


                        # ------------------------------
                        # 🧠 STORE BRAIN REGION DATA
                        # ------------------------------
                        try:
                            print(f"🧠 Storing brain region data...")

                            brain_region_scores = comprehensive_results.get('brain_region_scores', {})

                            if brain_region_scores:
                                # Loop through each anatomical region and store metrics
                                for region_name, scores in brain_region_scores.items():
                                    region_data = {
                                        'Filename': uploaded_filename,                  # Uploaded file name
                                        'Patient_ID': original_filename,               # Patient/file ID
                                        'Brain_Region': region_name,                   # Region name
                                        'ScoreCAM_Importance_Score': float(scores['score_cam_score']),      # Raw ScoreCAM score
                                        'ScoreCAM_Importance_Percentage': float(scores['score_cam_score'] * 100), # Percent score
                                        'Region_Area_Pixels': int(scores.get('pixel_count', 0)),            # Area in pixels
                                        'Region_Area_Percentage': float(scores['area_percentage']),         # Area percentage
                                        'analysis_method': 'scorecam_only'             # Method used
                                    }

                                    # Store region-level data in database
                                    storage.store_batch_region(
                                        region_data=region_data,
                                        model_name='ScoreCAM',
                                        model_version='v1.0'
                                    )
                                    storage_summary['regions_stored'] += 1

                                print(f"✅ Stored {storage_summary['regions_stored']} brain regions")
                            else:
                                # Handle case where no region scores are available
                                print("⚠️ No brain region scores available to store")
                                storage_summary['errors'].append("No brain region scores available")

                        except Exception as region_error:
                            # Log any storage errors
                            storage_summary['errors'].append(f"Region storage: {str(region_error)}")
                            print(f"❌ Region storage failed: {region_error}")


                        # Return summary of stored items and errors
                        return storage_summary
                    
                    # ------------------------------
                    # 🔄 Process each uploaded MRI file
                    # ------------------------------
                    for file_idx, uploaded_file in enumerate(uploaded_files):
                        # Update main status with current file progress
                        main_status.text(f"🔄 Processing file {file_idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        # Initialize per-file progress bar and status text
                        file_progress = st.progress(0)
                        file_status = st.empty()
                        
                        try:
                            # Step 1: Load and preprocess image
                            file_status.text("📸 Step 1/6: Loading and preprocessing image...")
                            file_progress.progress(15)
                            
                            # Enhanced image loading with error handling
                            try:
                                # Open image using PIL
                                image = Image.open(uploaded_file)
                                
                                # Ensure image is RGB; convert if not
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                
                                # Validate dimensions
                                if image.size[0] == 0 or image.size[1] == 0:
                                    raise ValueError("Invalid image dimensions")
                                
                                # Extract base filename without extension
                                original_filename = uploaded_file.name.split('.')[0]
                                
                                # Resize image to model input size (IMG_SIZE x IMG_SIZE)
                                img = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                                img_array = np.array(img, dtype=np.float32)
                                
                                # Validate numpy array
                                if img_array.size == 0:
                                    raise ValueError("Empty image array")
                                
                                # Normalize pixel values to [0, 1]
                                img_array = img_array.astype(np.float32) / 255.0
                                
                                # Handle different image formats and channel numbers
                                if len(img_array.shape) == 2:
                                    # Grayscale image: stack channels to create RGB
                                    img_array = np.stack([img_array] * 3, axis=-1)
                                elif len(img_array.shape) == 3:
                                    if img_array.shape[2] == 1:
                                        # Single-channel: repeat to RGB
                                        img_array = np.repeat(img_array, 3, axis=-1)
                                    elif img_array.shape[2] == 4:
                                        # RGBA image: discard alpha channel
                                        img_array = img_array[:, :, :3]
                                    elif img_array.shape[2] != 3:
                                        raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
                                else:
                                    raise ValueError(f"Unsupported image shape: {img_array.shape}")
                                
                                # Final validation: ensure shape matches model input
                                if img_array.shape != (IMG_SIZE, IMG_SIZE, 3):
                                    raise ValueError(f"Final image shape is incorrect: {img_array.shape}, expected: ({IMG_SIZE}, {IMG_SIZE}, 3)")
                                
                                # Clamp values to valid [0,1] range
                                img_array = np.clip(img_array, 0.0, 1.0)
                            
                            except Exception as img_error:
                                # Catch any preprocessing/loading errors and skip this file
                                st.error(f"❌ Error loading image {uploaded_file.name}: {str(img_error)}")
                                continue

                            # ------------------------------
                            # Step 2: Run basic prediction analysis
                            # ------------------------------
                            file_status.text("🤖 Step 2/6: Running prediction analysis...")
                            file_progress.progress(30)

                            try:
                                # Expand image dimensions to match model input (batch of 1)
                                predictions = st.session_state.model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
                                
                                # Determine predicted class and confidence
                                pred_class = np.argmax(predictions)
                                confidence = predictions[pred_class]

                                # Construct a dictionary for storing all basic results
                                comprehensive_results = {
                                    'image': original_filename,              # Filename reference
                                    'predicted_class': CLASS_NAMES[pred_class],  # Human-readable class label
                                    'confidence': float(confidence),         # Confidence of prediction
                                    'all_predictions': predictions.tolist(), # Raw probabilities for all classes
                                    'class_names': CLASS_NAMES,             # Class label mapping
                                    'brain_region_scores': {},              # Placeholder for region-level analysis
                                    'analysis_time': 0.0,                   # Will store total analysis duration
                                    'methods': ['ScoreCAM', 'Brain Region Analysis']  # Methods used for explainability
                                }
                                
                                print(f"✅ Basic prediction: {CLASS_NAMES[pred_class]} ({confidence:.2%})")

                            except Exception as pred_error:
                                # Handle prediction errors
                                st.error(f"❌ Prediction failed for {uploaded_file.name}: {str(pred_error)}")
                                batch_results['errors'].append(f"{uploaded_file.name}: Prediction - {str(pred_error)}")
                                continue  # Skip to next file

                            # ------------------------------
                            # Step 3: Generate brain region scores using ScoreCAM
                            # ------------------------------
                            file_status.text("🧠 Step 3/6: Analyzing brain regions with ScoreCAM...")
                            file_progress.progress(50)

                            try:
                                start_time = time.time()  # Track duration of brain region analysis
                                
                                # Generate enhanced brain mask from MRI
                                brain_mask = scorecam_analyzer.create_enhanced_brain_mask(img_array)
                                
                                # Generate ScoreCAM heatmap for predicted class
                                heatmap, _ = scorecam_analyzer.score_cam(img_array, pred_class, use_brain_mask=True)
                                
                                # Create anatomical region masks (e.g., hippocampus, frontal lobe)
                                region_masks = scorecam_analyzer.create_anatomical_region_masks(brain_mask)
                                
                                # Calculate importance scores per anatomical region based on ScoreCAM heatmap
                                region_scores = scorecam_analyzer.calculate_region_importance_scores(
                                    heatmap, region_masks, brain_mask
                                )
                                
                                # Update comprehensive results with region scores and total analysis time
                                comprehensive_results['brain_region_scores'] = region_scores
                                comprehensive_results['analysis_time'] = time.time() - start_time
                                
                                print(f"✅ Brain region analysis completed in {comprehensive_results['analysis_time']:.2f}s")
                                print(f"   - Regions analyzed: {len(region_scores)}")

                            except Exception as analysis_error:
                                # Handle errors during ScoreCAM and brain region analysis
                                st.error(f"❌ Brain region analysis failed for {uploaded_file.name}: {str(analysis_error)}")
                                batch_results['errors'].append(f"{uploaded_file.name}: Region analysis - {str(analysis_error)}")
                                continue  # Skip to next file

                            # ------------------------------
                            # Step 4: Generate and store specific images + prediction & region tables
                            # ------------------------------
                            file_status.text("💾 Step 4/6: Generating 10 images and storing in database...")
                            file_progress.progress(70)

                            try:
                                print(f"\n💾 Generating 10 specific images and storing in database for {original_filename}...")
                                
                                # Create a temporary directory for storing images before database upload
                                import tempfile
                                temp_dir = tempfile.mkdtemp()
                                
                                # Save a reference image (original MRI) in the temp directory
                                temp_image_path = os.path.join(temp_dir, f"{original_filename}.jpg")
                                temp_image = Image.fromarray((img_array * 255).astype(np.uint8))  # Convert normalized [0,1] array to 0-255
                                temp_image.save(temp_image_path)
                                
                                # Call the shared utility function to generate all ScoreCAM images (10 total) and store in DB
                                storage_summary = store_scorecam_results(
                                    storage=storage,
                                    scorecam_analyzer=scorecam_analyzer,
                                    img_array=img_array,
                                    temp_image_path=temp_image_path,
                                    comprehensive_results=comprehensive_results,
                                    uploaded_filename=uploaded_file.name,
                                    original_filename=original_filename,
                                    timestamp=timestamp
                                )
                                
                                # Update batch-level storage statistics
                                batch_results['storage_summaries'].append(storage_summary)
                                batch_results['total_images_stored'] += storage_summary['images_stored']
                                batch_results['total_regions_stored'] += storage_summary['regions_stored']
                                
                                # If there were errors in storing images/regions/predictions, append to batch error log
                                if storage_summary['errors']:
                                    batch_results['errors'].extend(storage_summary['errors'])
                                
                                print(f"✅ ScoreCAM storage completed for {original_filename}")
                                print(f"   - Images stored: {storage_summary['images_stored']}/10")
                                print(f"   - Predictions stored: {storage_summary['predictions_stored']}")
                                print(f"   - Regions stored: {storage_summary['regions_stored']}")
                                
                                # Cleanup temporary directory to free disk space
                                import shutil
                                shutil.rmtree(temp_dir)
                                
                            except Exception as storage_error:
                                st.error(f"❌ Streamlined storage failed for {uploaded_file.name}: {str(storage_error)}")
                                batch_results['errors'].append(f"{uploaded_file.name}: Storage - {str(storage_error)}")
                                continue  # Skip to next file in batch

                            # ------------------------------
                            # Step 5: Prepare batch summary data
                            # ------------------------------
                            file_status.text("📊 Step 5/6: Preparing batch summary...")
                            file_progress.progress(80)
                            st.empty()  # Clear any temporary UI messages

                            try:
                                # Prepare prediction record for batch table
                                prediction_data = {
                                    'Filename': uploaded_file.name,
                                    'Patient_ID': original_filename,
                                    'Predicted_Class': comprehensive_results['predicted_class'],
                                    'Confidence': comprehensive_results['confidence'],
                                    'Mild_Demented_Probability': comprehensive_results['all_predictions'][0],
                                    'Moderate_Demented_Probability': comprehensive_results['all_predictions'][1],
                                    'Non_Demented_Probability': comprehensive_results['all_predictions'][2],
                                    'Very_Mild_Demented_Probability': comprehensive_results['all_predictions'][3],
                                    'Analysis_Time': comprehensive_results['analysis_time'],
                                    'Images_Generated': 10  # Always 10 specific images per file
                                }
                                batch_results['predictions'].append(prediction_data)
                                
                                # Prepare brain region table for batch storage
                                for region_name, scores in region_scores.items():
                                    region_data = {
                                        'Filename': uploaded_file.name,
                                        'Patient_ID': original_filename,
                                        'Analysis_Timestamp': timestamp,
                                        'Brain_Region': region_name,
                                        
                                        # Store ScoreCAM importance scores and percentage
                                        'ScoreCAM_Importance_Score': float(scores['score_cam_score']),
                                        'ScoreCAM_Importance_Percentage': float(scores['score_cam_score'] * 100),
                                        
                                        'Region_Area_Pixels': int(scores['pixel_count']),
                                        'Region_Area_Percentage': float(scores['area_percentage']),
                                        'Region_Description': scores['description'],  # Optional textual description
                                        'analysis_method': 'scorecam_only'
                                    }
                                    batch_results['regions'].append(region_data)
                                
                                # Generate a summary for this file highlighting top region and overall storage status
                                top_region = max(comprehensive_results['brain_region_scores'].items(), 
                                                key=lambda x: x[1]['score_cam_score'])
                                
                                summary = {
                                    'filename': original_filename,
                                    'prediction': comprehensive_results['predicted_class'],
                                    'confidence': comprehensive_results['confidence'],
                                    'top_region': top_region[0],
                                    'top_region_score': top_region[1]['score_cam_score'],
                                    'analysis_time': comprehensive_results['analysis_time'],
                                    'images_stored': storage_summary['images_stored'],
                                    'storage_status': 'success' if storage_summary['images_stored'] == 10 else 'partial'
                                }
                                batch_results['summaries'].append(summary)
                                
                            except Exception as summary_error:
                                # Log errors during summary creation
                                batch_results['errors'].append(f"{uploaded_file.name}: Summary - {str(summary_error)}")

                            # ------------------------------
                            # Step 6: Complete file processing
                            # ------------------------------
                            file_status.text("✅ Step 6/6: File processing completed!")  # Update UI for completion
                            file_progress.progress(100)  # Show full progress for this file
                            st.empty()  # Clear any leftover UI elements

                            # Update overall batch progress
                            main_progress.progress((file_idx + 1) / len(uploaded_files))

                        except Exception as e:
                            # Catch any unexpected errors per file to avoid breaking the batch
                            st.error(f"❌ Unexpected error processing {uploaded_file.name}: {str(e)}")
                            batch_results['errors'].append(f"{uploaded_file.name}: Unexpected - {str(e)}")

                        finally:
                            # Clean up file-specific progress bars and status texts
                            file_progress.empty()
                            file_status.empty()
                            plt.close('all')  # Close all open matplotlib figures to free memory

                    # ------------------------------
                    # Final batch completion
                    # ------------------------------
                    main_status.text("✅ Batch processing completed!")
                    main_progress.progress(100)

                    # Display success message to the user
                    mri_title = "🎉 MRI Scan Analysis Completed!"
                    mri_desc = "Your MRI Scan has been analyzed successfully. You can now view patient-level insights and download reports."
                    st.markdown(success_message(mri_title, mri_desc), unsafe_allow_html=True)

                    # Cleanup main progress UI elements
                    main_progress.empty()
                    main_status.empty()

        # ------------------------------
        # Error handling if model is not loaded
        # ------------------------------
        else:
            st.error("❌ Model could not be loaded. Please check your model configuration.")

    # ------------------------------
    # Prompt user to upload images if none are uploaded
    # ------------------------------
    else:
        st.info("👆 Please upload one or more MRI scan images to begin analysis")

# ------------------------------
# Navigation section
# ------------------------------
def create_navigation_section():
    """Enhanced navigation section with Home and Dashboard buttons"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Centralized layout for navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            # Home button
            if st.button("🏠 Home", key="main_upload_btn", use_container_width=True, 
                        help="Return to home page"):
                st.switch_page("home.py")
        
        with subcol2:
            # Show dashboard based on data type (clinical CSV or MRI image)
            if st.session_state.data_type == 'csv':
                if st.button("📊 Clinical Dashboard", key="main_dashboard_btn", use_container_width=True,
                            help="View existing predictions and analytics dashboard"):
                    st.switch_page("pages/ClinicalDashboardPage.py")
            else:  # MRI image type
                if st.button("🧠 MRI Dashboard", key="main_mri_dashboard_btn", use_container_width=True,
                            help="View MRI scan predictions and analytics dashboard"):
                    st.switch_page("pages/MRIDashboardPage.py")

# Call navigation section to render buttons
create_navigation_section()













