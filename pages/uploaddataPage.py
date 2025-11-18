# uploaddatapage.py - Upload Data Page with Form Input
"""
Streamlit Upload Data Page for Alzheimer's Disease Prediction

This module allows users to input patient clinical data for analysis:

- **Clinical Data**: Enter patient information through an interactive form for feature-based predictions.
  The system uses machine learning models to analyze clinical data and provides interpretable
  visual insights via SHAP, highlighting features most relevant to the prediction.

User input is processed, predictions are generated, and interpretability outputs
are displayed. Users can navigate to the dashboard for detailed results.
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
# 🧪 Machine learning
# ------------------------------
import shap

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

# Add these imports at the top of your file
import urllib.request
import ssl

BASE_DIR = Path("/tmp/alzheimer_app")
BASE_DIR.mkdir(exist_ok=True, parents=True)

MODEL_DIR = BASE_DIR / "alzheimers_model_files"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

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
    
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl._create_unverified_context()
    
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
        'shap_utils.py'
    ]
    
    BASE_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl._create_unverified_context()
    
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
            raise

    print("✅ All utility files ready")

@st.cache_resource
def load_utilities():
    """Load and cache SHAP utilities for explainability"""
    try:
        # Try to download utilities, but don't fail if it doesn't work
        try:
            download_utilities_from_github()
        except Exception as download_error:
            print(f"⚠️ Warning: Could not download utilities: {str(download_error)}")
        
        shap_utility_path = BASE_DIR / "shap_utils.py"
        
        print(f"Loading SHAP utilities from: {shap_utility_path}")
        
        if not shap_utility_path.exists():
            print(f"⚠️ shap_utils.py not found at {shap_utility_path}")
            return None
        
        spec = importlib.util.spec_from_file_location("shap_utility", str(shap_utility_path))
        shap_utility = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shap_utility)
        
        print("✅ SHAP utilities loaded successfully")
        return shap_utility
    except Exception as e:
        print(f"❌ Failed to load SHAP utilities: {str(e)}")
        return None

shap_utility = load_utilities()

if shap_utility is not None:
    create_shap_analysis_results = shap_utility.create_shap_analysis_results
else:
    st.warning("⚠️ SHAP utilities could not be loaded. Some features may be unavailable.")
    create_shap_analysis_results = None

apply_custom_css()

st.set_page_config(
    page_title="Clinical Data Analysis - Alzheimer's Diagnosis AI", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

@st.cache_data
def create_hero_section():
    """Render hero section with title and subtitle"""
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">🧠 AI-Driven Alzheimer's Clinical Analysis</h1>
        <p class="hero-subtitle">Enter patient clinical data and get clear AI-driven Alzheimer's insights — fast and reliable</p>
    </div>
    """, unsafe_allow_html=True)

create_hero_section()



@st.cache_resource
def load_csv_models():
    """Load all required CSV model files for clinical data analysis"""
    try:
        download_models_from_github()
        CSV_MODEL_PATH = MODEL_DIR
        
        print(f"Looking for models in: {CSV_MODEL_PATH}")
        
        # Try to import sklearn with custom unpickler
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
        
        # Add compatibility for different sklearn versions
        try:
            from sklearn.compose._column_transformer import _RemainderColsList
        except ImportError:
            # Create a dummy class if not available
            import sys
            from sklearn import compose
            
            class _RemainderColsList(list):
                """Compatibility class for older sklearn versions"""
                pass
            
            # Register the class in the module
            sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList
            print("⚠️ Added compatibility layer for _RemainderColsList")
        
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
        import traceback
        traceback.print_exc()
        st.error(f"❌ Error loading CSV models: {str(e)}")
        st.info("💡 Try: pip install --upgrade scikit-learn==1.3.0")
        return None, None, None, None, None




with st.spinner("🤖 Initializing AI models and preparing analysis pipeline..."):
    model, preprocessor, top_features, explainer, feature_names = load_csv_models()

if model is None:
    st.error("Failed to load models. Please check the configuration.")
else:
    st.markdown('<h3 class="section-header">📋 Patient Information Form</h3>', unsafe_allow_html=True)
    
    # Create form for patient data entry
    with st.form("patient_data_form"):
        st.markdown("#### 👤 Patient Identification")
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID*", placeholder="e.g., P001", help="Unique identifier for the patient")
        
        
        st.markdown("#### 📊 Clinical Features")
        st.markdown("Please enter all required clinical measurements:")

        # Define feature ranges and descriptions
        feature_info = {
            'MMSE': {
                'min': 0, 'max': 30, 'step': 1,
                'help': 'Mini-Mental State Examination (0-30). Lower scores indicate cognitive impairment.'
            },
            'FunctionalAssessment': {
                'min': 0, 'max': 10, 'step': 1,
                'help': 'Functional assessment score (0-10). Lower scores indicate greater impairment.'
            },
            'MemoryComplaints': {
                'min': 0, 'max': 1, 'step': 1,
                'help': 'Memory complaints: 0 = No, 1 = Yes'
            },
            'BehavioralProblems': {
                'min': 0, 'max': 1, 'step': 1,
                'help': 'Behavioral problems: 0 = No, 1 = Yes'
            },
            'ADL': {
                'min': 0, 'max': 10, 'step': 1,
                'help': 'Activities of Daily Living (0-10). Lower scores indicate greater impairment.'
            },
            'Disorientation': {
                'min': 0, 'max': 1, 'step': 1,
                'help': 'Disorientation: 0 = No, 1 = Yes'
            },
            'PersonalityChanges': {
                'min': 0, 'max': 1, 'step': 1,
                'help': 'Personality changes: 0 = No, 1 = Yes'
            },
            'DifficultyCompletingTasks': {
                'min': 0, 'max': 1, 'step': 1,
                'help': 'Difficulty completing tasks: 0 = No, 1 = Yes'
            },
            'Forgetfulness': {
                'min': 0, 'max': 1, 'step': 1,
                'help': 'Forgetfulness: 0 = No, 1 = Yes'
            },
            'SleepQuality': {
                'min': 4, 'max': 10, 'step': 1,
                'help': 'Sleep quality score (4-10). Higher scores indicate better sleep.'
            }
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
                        info = feature_info.get(feature, {'min': 0, 'max': 100, 'step': 1, 'help': f'Enter value for {feature}'})
                        
                        # Create input with proper range and help text
                        feature_values[feature] = st.number_input(
                            f"{feature} ({info['min']}-{info['max']})",
                            min_value=info['min'],
                            max_value=info['max'],
                            value=info['min'],
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

                        # Create prediction data with patient info
                        prediction_data = {
                            'Patient_ID': patient_id,
                            'Predicted_Diagnosis': int(prediction),
                            'Prediction_Probability': float(probability)
                        }
                        
                        # Add all feature values to prediction data
                        for feature, value in feature_values.items():
                            prediction_data[feature] = value

                       

                        # Create SHAP results
                        if create_shap_analysis_results is not None:
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

                        st.session_state.analysis_complete = True

                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)

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
