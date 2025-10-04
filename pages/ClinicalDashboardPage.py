# clinicalDashboardPage.py - Clinical Dashboard Page
"""
Streamlit Clinical Dashboard for Alzheimer's Disease Prediction

This module provides an interactive dashboard for analyzing and visualizing
predictions based on structured clinical data:

- **Overview Tab**: Summary statistics, risk stratification, and top predictive features.
- **Feature Description Tab**: Detailed explanations of clinical features, their measurement,
  and contribution to model predictions.
- **Individual Patient Profiles**: Displays patient-level predictions, MMSE and functional scores,
  SHAP-based top risk/protective factors, and personalized recommendations.
- **Patient Comparison & What-If Analysis**: Compare multiple patients and adjust features
  to explore potential changes in prediction outcomes.

The dashboard uses SHAP values to provide **interpretable, feature-level insights**
from the clinical machine learning model.

This file is intended to be run within the Streamlit application framework:
    streamlit run clinicaldashboard.py
"""

# ============================================================
# 📦 CORE IMPORTS & CONFIGURATION
# ============================================================

# ------------------------------
# Standard library imports
# ------------------------------
import os                   # File system operations
import sys                  # System-specific parameters & functions
import json                 # JSON serialization/deserialization
import warnings             # Filter or manage warnings
from datetime import datetime  # Handling timestamps

# ------------------------------
# Data processing & ML utilities
# ------------------------------
import pandas as pd         # type: ignore # DataFrame manipulation
import joblib               # type: ignore # Model serialization/deserialization
from pathlib import Path
# ------------------------------
# Visualization libraries
# ------------------------------
import matplotlib           # type: ignore # Core matplotlib package
matplotlib.use('Agg')       # Non-interactive backend for Streamlit rendering
import matplotlib.pyplot as plt  # type: ignore # Plotting
import plotly.express as px      # type: ignore # High-level Plotly visualization
import plotly.graph_objects as go  # type: ignore # Low-level Plotly visualization

# ------------------------------
# Web app framework & UI extras
# ------------------------------
import streamlit as st              # type: ignore # Streamlit web app framework
from streamlit_lottie import st_lottie  # type: ignore # Animated Lottie integration
import requests                     # type: ignore # HTTP requests (used for fetching Lottie JSONs)

# ------------------------------
# Local project-specific imports
# ------------------------------
from style import apply_custom_css                # Custom CSS for Streamlit app
from alzheimers_db_setup import AlzheimerPredictionStorage  # Database storage handler
from clinical_explanations import generate_patient_recommendations  # Clinical recommendations module

# ============================================================
# 🔕 SUPPRESS WARNINGS
# ============================================================
warnings.filterwarnings('ignore')

# ============================================================
# 🎨 APPLY CUSTOM STYLING
# ============================================================
apply_custom_css()

# ============================================================
# 🖥 STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI-Powered Alzheimer's Prediction Dashboard", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 📍 DATABASE PATH SETUP
# ============================================================
# Add this near the top of your file, after imports

# ============================================================
# 📍 DATABASE PATH SETUP
# ============================================================
def get_database_directory():
    """Find the Alzheimer_Database directory"""
    current_file = Path(__file__).resolve()
    
    # Check Alzheimer_Project folder structure
    for parent in list(current_file.parents):
        db_dir = parent / 'Alzheimer_Project' / 'Alzheimer_Database'
        if db_dir.exists() and (db_dir / 'alzheimer_predictions.db').exists():
            return db_dir
    
    # Fallback
    return current_file.parent / 'Alzheimer_Project' / 'Alzheimer_Database'

# Set database paths
DB_DIR = get_database_directory()
DB_PATH = DB_DIR / 'alzheimer_predictions.db'

# Verify database exists
if not DB_PATH.exists():
    st.error(f"Database not found at: {DB_PATH}")
    st.stop()
else:
    print(f"✓ Found database at: {DB_PATH}")

# ============================================================
# 🎞 LOTTIE ANIMATION HELPERS
# ============================================================

def load_lottie_url(url: str):
    """
    Load a Lottie animation from a remote URL.

    Parameters:
        url (str): The Lottie animation URL.

    Returns:
        dict | None: Lottie animation JSON data or None if failed.
    """
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def load_local_brain():
    """
    Load a local 'funny brain' Lottie animation.

    Returns:
        dict | None: Lottie JSON or None if file not found.
    """
    # Try multiple possible locations
    possible_paths = [
        "/Users/swehavenkateshwari/Alzheimer/lottie_brain.json",
        Path(__file__).parent / "lottie_brain.json",
        Path(__file__).parent / "assets" / "lottie_brain.json"
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            continue
    
    # If no local file found, try loading from URL
    try:
        brain_url = "https://assets5.lottiefiles.com/packages/lf20_touohxv0.json"
        return load_lottie_url(brain_url)
    except:
        return None

# ============================================================
# 🧠 HERO SECTION - Dashboard Header
# ============================================================
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">🧠 AI-Powered Alzheimer's Clinical Prediction Dashboard</h1>
    <p class="hero-subtitle">Advanced Machine Learning Analysis with SHAP Interpretability</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 📥 LOAD MODEL COMPONENTS
# ============================================================
@st.cache_data
def load_model_components():
    """
    Load all pre-trained model components needed for predictions.
    Downloads from GitHub if not present locally.
    Returns:
        tuple: (model, preprocessor, feature_names_original, explainer, feature_names_processed)
    """
    try:
        # Define paths
        BASE_DIR = Path("/tmp/alzheimer_app")
        MODEL_DIR = BASE_DIR / "alzheimers_model_files"
        
        # Ensure directories exist
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        
        # GitHub raw content base URL
        GITHUB_RAW_URL = "https://raw.githubusercontent.com/sv3112/Alzheimer_AI_Diagnosis_Dashboard/main/alzheimers_model_files"
        
        # List of model files
        MODEL_FILES = [
            'alzheimers_best_model.pkl',
            'alzheimers_preprocessor_top10.pkl',
            'alzheimers_top10_features.pkl',
            'alzheimers_shap_explainer.pkl',
            'alzheimers_feature_names_processed.pkl'
        ]
        
        # Download files if they don't exist
        import urllib.request
        import ssl
        ssl_context = ssl.create_default_context()
        
        for filename in MODEL_FILES:
            local_path = MODEL_DIR / filename
            
            if not local_path.exists():
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
        
        # Load all components
        model = joblib.load(MODEL_DIR / 'alzheimers_best_model.pkl')
        preprocessor = joblib.load(MODEL_DIR / 'alzheimers_preprocessor_top10.pkl')
        feature_names_original = joblib.load(MODEL_DIR / 'alzheimers_top10_features.pkl')
        explainer = joblib.load(MODEL_DIR / 'alzheimers_shap_explainer.pkl')
        feature_names_processed = joblib.load(MODEL_DIR / 'alzheimers_feature_names_processed.pkl')
        
        print("✓ All model components loaded successfully")
        return model, preprocessor, feature_names_original, explainer, feature_names_processed
    
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        print(f"❌ Detailed error: {str(e)}")
        return None, None, None, None, None

# ============================================================
# 📊 LOAD DASHBOARD DATA FROM DATABASE
# ============================================================
@st.cache_data(ttl=300)
def load_dashboard_data():
    """
    Load predictions and global feature importance from the database.

    Returns:
        dict: {
            'Individual_Predictions': DataFrame of per-patient predictions,
            'Global_Feature_Importance': DataFrame of global SHAP feature importance
        }
    """
    # CRITICAL FIX: Pass base_dir explicitly to avoid creating new database
    storage = AlzheimerPredictionStorage(base_dir=str(DB_DIR))
    try:
        individual_predictions = pd.DataFrame(storage.get_individual_predictions())
        global_importance = pd.DataFrame(storage.get_global_importance())
        
        return {
            'Individual_Predictions': individual_predictions,
            'Global_Feature_Importance': global_importance,
            'record_count': len(individual_predictions) + len(global_importance)
        }
    finally:
        storage.close()

# ============================================================
# 📂 INITIAL DATA LOAD
# ============================================================
with st.spinner("Loading clinical data from database..."):
    data_dict = load_dashboard_data()
    df_predictions = data_dict.get('Individual_Predictions', pd.DataFrame())
    df_global_importance = data_dict.get('Global_Feature_Importance', pd.DataFrame())
    record_count = data_dict.get('record_count', 0)


# ============================================================
# 🎛 SIDEBAR FILTERS & DEMOGRAPHICS
# ============================================================

# ------------------------------------------------------------
# 🔍 Sidebar Header
# ------------------------------------------------------------
st.sidebar.header("🔍 Filters")

# ------------------------------------------------------------
# 🧠 Sidebar Brain Animation (optional visual element)
# ------------------------------------------------------------
funny_brain_sidebar = load_local_brain()
if funny_brain_sidebar:
    with st.sidebar:
        st_lottie(
            funny_brain_sidebar,
            height=120,
            key="sidebar_funny_brain",
            speed=0.8,
            loop=True,
            quality="medium"
        )

# ------------------------------------------------------------
# ♻️ Reset Filters Mechanism
# ------------------------------------------------------------
# Initialize session state counter if not set
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# Reset button — increments counter to force widget re-creation
if st.sidebar.button("🎯 Reset All Filters", type="primary", use_container_width=True):
    st.session_state.reset_counter += 1
    st.rerun()

# Unique suffix for all filter widgets so they reset when counter changes
reset_suffix = f"_{st.session_state.reset_counter}"

# ------------------------------------------------------------
# 📊 Risk Level Filter
# ------------------------------------------------------------
risk_filter = st.sidebar.selectbox(
    "Risk Level",
    ["All", "No Risk", "Low Risk (<0.5)", "Medium Risk (0.5-0.7)", "High Risk (≥0.7)"],
    index=0,  # Always default to "All"
    key=f"risk_level_filter{reset_suffix}"
)

# Start with full dataset
df_filtered = df_predictions.copy()

# Apply risk filter logic
if risk_filter != "All":
    if risk_filter == "No Risk":
        # No risk → Prediction == 0
        df_filtered = df_filtered[df_filtered["Predicted_Diagnosis"] == 0]

    elif risk_filter == "Low Risk (<0.5)":
        # Low risk → Positive prediction with probability < 0.7
        df_filtered = df_filtered[
            (df_filtered["Predicted_Diagnosis"] == 1)
            & (df_filtered["Prediction_Probability"] < 0.7)
        ]

    elif risk_filter == "Medium Risk (0.5-0.7)":
        # Medium risk → Positive prediction with probability between 0.7 and 0.9
        df_filtered = df_filtered[
            (df_filtered["Predicted_Diagnosis"] == 1)
            & (df_filtered["Prediction_Probability"] >= 0.7)
            & (df_filtered["Prediction_Probability"] < 0.9)
        ]

    elif risk_filter == "High Risk (≥0.7)":
        # High risk → Positive prediction with probability >= 0.9
        df_filtered = df_filtered[
            (df_filtered["Predicted_Diagnosis"] == 1)
            & (df_filtered["Prediction_Probability"] >= 0.9)
        ]

# ------------------------------------------------------------
# 👥 Age Group Filter (if RAW_Age exists)
# ------------------------------------------------------------
age_group_filter = "All"
if "RAW_Age" in df_predictions.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("👥 Demographics")

    age_group_filter = st.sidebar.selectbox(
        "👤 Age Group",
        ["All", "<50", "50-59", "60-69", "70-79", "80+"],
        index=0,
        key=f"age_group_filter{reset_suffix}"
    )

    if age_group_filter != "All":
        def get_age_group(age):
            """Map numeric age to a categorical age group."""
            if pd.isna(age):
                return "Unknown"
            elif age < 50:
                return "<50"
            elif age < 60:
                return "50-59"
            elif age < 70:
                return "60-69"
            elif age < 80:
                return "70-79"
            else:
                return "80+"

        # Temporary Age_Group column for filtering
        df_filtered["Age_Group"] = df_filtered["RAW_Age"].apply(get_age_group)
        df_filtered = df_filtered[df_filtered["Age_Group"] == age_group_filter]
        df_filtered = df_filtered.drop("Age_Group", axis=1)

# ------------------------------------------------------------
# ⚧ Gender Filter (if RAW_Gender exists)
# ------------------------------------------------------------
gender_filter = "All"
if "RAW_Gender" in df_predictions.columns:
    # Add Demographics header if not already added by Age filter
    if "RAW_Age" not in df_predictions.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("👥 Demographics")

    available_genders = df_predictions["RAW_Gender"].unique()

    # Handle gender label format (string vs numeric)
    if any(isinstance(g, str) for g in available_genders):
        gender_options = ["All"] + sorted([str(g) for g in available_genders if pd.notna(g)])
    else:
        gender_options = ["All", "Female", "Male"]

    gender_filter = st.sidebar.selectbox(
        "⚧ Gender",
        gender_options,
        index=0,
        key=f"gender_filter{reset_suffix}"
    )

    if gender_filter != "All":
        if any(isinstance(g, str) for g in available_genders):
            df_filtered = df_filtered[df_filtered["RAW_Gender"] == gender_filter]
        else:
            gender_mapping = {"Female": 0, "Male": 1}
            if gender_filter in gender_mapping:
                df_filtered = df_filtered[df_filtered["RAW_Gender"] == gender_mapping[gender_filter]]

# ------------------------------------------------------------
# 📋 Filter Summary
# ------------------------------------------------------------
if len(df_filtered) < len(df_predictions):
    filtered_percentage = (len(df_filtered) / len(df_predictions)) * 100
    st.sidebar.info(
        f"📊 Showing {len(df_filtered):,} of {len(df_predictions):,} patients "
        f"({filtered_percentage:.1f}%)"
    )
else:
    st.sidebar.success(f"📊 Showing all {len(df_predictions):,} patients")

# ============================================================
# 📌 KEY METRICS DISPLAY
# ============================================================

# Create 5 equal-width metric columns
col1, col2, col3, col4, col5 = st.columns(5)

# ------------------------------------------------------------
# 👥 Total Patients
# ------------------------------------------------------------
with col1:
    total_patients = len(df_filtered)
    st.markdown(f"""
    <div class="metric-card_D">
        <div class="metric-value_D">{total_patients}</div>
        <div class="metric-label_d">👥 Total Patients</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# 🟢 No Risk
# ------------------------------------------------------------
with col2:
    no_risk = len(df_filtered[df_filtered["Predicted_Diagnosis"] == 0])
    st.markdown(f"""
    <div class="metric-card_D">
        <div class="metric-value_D">{no_risk}</div>
        <div class="metric-label_D">🟢 No Risk</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# 🟡 Low Risk
# ------------------------------------------------------------
with col3:
    low_risk = len(df_filtered[
        (df_filtered["Predicted_Diagnosis"] == 1)
        & (df_filtered["Prediction_Probability"] < 0.7)
    ])
    st.markdown(f"""
    <div class="metric-card_D">
        <div class="metric-value_D">{low_risk}</div>
        <div class="metric-label_D">🟡 Low Risk</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# 🟠 Medium Risk
# ------------------------------------------------------------
with col4:
    medium_risk = len(df_filtered[
        (df_filtered["Predicted_Diagnosis"] == 1)
        & (df_filtered["Prediction_Probability"] >= 0.7)
        & (df_filtered["Prediction_Probability"] < 0.9)
    ])
    st.markdown(f"""
    <div class="metric-card_D">
        <div class="metric-value_D">{medium_risk}</div>
        <div class="metric-label_D">🟠 Medium Risk</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# 🔴 High Risk
# ------------------------------------------------------------
with col5:
    high_risk = len(df_filtered[
        (df_filtered["Predicted_Diagnosis"] == 1)
        & (df_filtered["Prediction_Probability"] >= 0.9)
    ])
    st.markdown(f"""
    <div class="metric-card_D">
        <div class="metric-value_D">{high_risk}</div>
        <div class="metric-label_D">🔴 High Risk</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 🗂 TAB NAVIGATION
# ============================================================

("",)  # Optional spacer

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Overview",           # Tab 1: Overall statistics & charts
    "🔍 Clinical Feature Insights",    # Tab 2: Feature-level analysis
    "👥 Individual Patient Profile",   # Tab 3: Per-patient view
    "📈 Patient Cohort Comparison",    # Tab 4: Group analysis
    "🩺 Clinical Scenario Simulation"  # Tab 5: What-if analysis
])

# -----------------------------
# 📊 DASHBOARD OVERVIEW - TAB 1
# -----------------------------
with tab1:
    # --------------------------------
    # 1️⃣ Define Feature Categories
    # --------------------------------
    # These lists group dataset features into categories for display and selection.
    demographic_features = ['Age', 'Gender', 'Ethnicity', 'EducationLevel']
    lifestyle_features = ['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
    medical_history = ['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']
    cognitive_assessment = ['MemoryComplaints', 'BehavioralProblems', 'MMSE', 'FunctionalAssessment', 'ADL']
    symptoms = ['Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']

    # ---------------------------------------------------------
    # 2️⃣ Risk Level Distribution - Pie/Donut Chart (Left Col) - NUCLEAR OPTION
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 class="main-title">🎯 Risk Level Distribution</h3>', unsafe_allow_html=True)

        # Count individuals in each risk category based on prediction results
        risk_counts = pd.Series({
            'No Risk': len(df_filtered[df_filtered['Predicted_Diagnosis'] == 0]),
            'Low Risk': len(df_filtered[
                (df_filtered['Predicted_Diagnosis'] == 1) &
                (df_filtered['Prediction_Probability'] < 0.7)
            ]),
            'Medium Risk': len(df_filtered[
                (df_filtered['Predicted_Diagnosis'] == 1) &
                (df_filtered['Prediction_Probability'] >= 0.7) &
                (df_filtered['Prediction_Probability'] < 0.9)
            ]),
            'High Risk': len(df_filtered[
                (df_filtered['Predicted_Diagnosis'] == 1) &
                (df_filtered['Prediction_Probability'] >= 0.9)
            ])
        })

        # NUCLEAR OPTION: Manually create the data in EXACT order we want
        # This eliminates any ambiguity about ordering
        
        # Get actual counts
        no_risk_count = risk_counts['No Risk']
        low_risk_count = risk_counts['Low Risk'] 
        medium_risk_count = risk_counts['Medium Risk']
        high_risk_count = risk_counts['High Risk']
        
        # Create lists in EXACT order with EXACT colors
        final_labels = []
        final_values = []
        final_colors = []
        
        # Add each category only if it has data, in our preferred order
        if no_risk_count > 0:
            final_labels.append('No Risk')
            final_values.append(no_risk_count)
            final_colors.append('#10b981')  # GREEN
            
        if low_risk_count > 0:
            final_labels.append('Low Risk') 
            final_values.append(low_risk_count)
            final_colors.append('#fde047')  # YELLOW
            
        if medium_risk_count > 0:
            final_labels.append('Medium Risk')
            final_values.append(medium_risk_count) 
            final_colors.append('#f59e0b')  # ORANGE
            
        if high_risk_count > 0:
            final_labels.append('High Risk')
            final_values.append(high_risk_count)
            final_colors.append('#dc2626')  # RED - This should be RED!


        # Create the chart using go.Figure for maximum control
        fig_pie = go.Figure(data=[go.Pie(
            labels=final_labels,
            values=final_values,
            hole=0.5,
            marker=dict(
                colors=final_colors,  # DIRECTLY assign colors - no room for error
                line=dict(color='#FFFFFF', width=3)
            ),
            textposition='auto',
            textinfo='percent+label',
            textfont=dict(size=14, color='white', family='Inter, sans-serif'),
            texttemplate='<b>%{label}<br>%{percent}</b>',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.1 if label == 'No Risk' else 0.05 for label in final_labels]
        )])

        # Add center text
        fig_pie.add_annotation(
            text=f'<b>{len(df_filtered)}</b><br>Total',
            x=0.45, y=0.5,
            font=dict(size=22, color='#1f2937', family='Inter, sans-serif'),
            showarrow=False
        )

        # Layout
        fig_pie.update_layout(
            font=dict(family="Inter, sans-serif", size=16),
            legend=dict(
                orientation="v",
                yanchor="middle", y=0.5,
                xanchor="left", x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#E5E5E5', 
                borderwidth=1
            ),
            margin=dict(l=20, r=200, t=0, b=0),
            plot_bgcolor='white', 
            paper_bgcolor='white',
            height=400
        )

        # Display the chart
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---------------------------------------------------------
    # 3️⃣ Model Feature Selection Summary (Right Col)
    # ---------------------------------------------------------
    with col2:
        st.markdown('<h3 class="main-title">🧠 Model Feature Selection Overview</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Display total available features
        with col1:
            st.markdown("""
            <div class="feature-card_D" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div style="font-size: 2.5rem;">📊</div>
                <div class="feature-value_D" style="font-size: 3rem;">34</div>
                <div class="feature-label_D">Total Available Features</div>
                <div class="feature-sublabel_D">Complete feature set</div>
            </div>
            """, unsafe_allow_html=True)

        # Display number of top features used
        with col2:
            st.markdown("""
            <div class="feature-card_D" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div style="font-size: 2.5rem;">⭐</div>
                <div class="feature-value_D" style="font-size: 3rem;">10</div>
                <div class="feature-label_D">Selected Top Features</div>
                <div class="feature-sublabel_D">30% selection rate</div>
            </div>
            """, unsafe_allow_html=True)

        # Progress bar to show selection efficiency
        selection_percentage = (10 / 34) * 100
        st.markdown(f"""
        <div style="background: #f0f0f0; border-radius: 10px; padding: 1rem;">
            <div style="display: flex; justify-content: space-between;">
                <span><b>Feature Selection Efficiency</b></span>
                <span style="color: #667eea;">{selection_percentage:.1f}%</span>
            </div>
            <div style="background: #e0e0e0; border-radius: 20px; height: 20px;">
                <div style="background: linear-gradient(90deg, #667eea 0%, #f093fb 100%);
                            width: {selection_percentage}%; height: 100%; border-radius: 20px;">
                </div>
            </div>
            <div style="text-align: center; font-size: 0.9rem; color: #666;">
                Using only the most impactful features for optimal performance
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4️⃣ Feature Categories Selector & Display
    # ---------------------------------------------------------
    st.markdown('<h3 class="section-title">🏗️ Feature Categories Overview</h3>', unsafe_allow_html=True)

    category_mapping = {
        'Demographics': demographic_features,
        'Lifestyle': lifestyle_features,
        'Medical History': medical_history,
        'Cognitive Assessment': cognitive_assessment,
        'Symptoms': symptoms
    }
    category_options = list(category_mapping.keys())

    # Create dropdown selector with counts
    category_labels = [f"{cat} ({len(category_mapping[cat])} features)" for cat in category_options]
    selected_index = st.selectbox(
        "", options=range(len(category_options)),
        format_func=lambda x: category_labels[x], label_visibility="collapsed"
    )

    selected_category = category_options[selected_index]
    features_in_category = category_mapping[selected_category]

    # Display features in a 2-column grid with icons
    cols = st.columns(2)
    for i, feature in enumerate(features_in_category):
        with cols[i % 2]:
            icon_map = {
                "age": "🎂", "gender": "👤", "ethnicity": "🌍", "education": "🎓",
                "bmi": "⚖️", "smoking": "🚬", "alcohol": "🍷", "physical": "💪",
                "diet": "🥗", "sleep": "😴", "family history": "👨‍👩‍👧‍👦",
                "cardiovascular": "❤️", "diabetes": "🩸", "depression": "😔",
                "head injury": "🤕", "hypertension": "💉", "memory": "🧩",
                "behavioral": "🎭", "mmse": "📊", "functional": "🔧", "adl": "🏠",
                "confusion": "😵", "disorientation": "🧭", "personality": "🎨",
                "difficulty": "📝", "forgetfulness": "💭"
            }
            feature_icon = next((emoji for key, emoji in icon_map.items() if key in feature.lower()), "•")
            st.markdown(f"<div class='feature-item'><span>{feature_icon}</span> <b>{feature}</b></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 5️⃣ Top Feature Importance (SHAP Values)
    # ---------------------------------------------------------
    if not df_global_importance.empty:
        st.markdown('<h3 class="main-title">🏆 Top 10 Most Important Features</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #666;">
        These features have the greatest impact on model predictions, ranked by mean absolute SHAP values.
        </p>
        """, unsafe_allow_html=True)

        top_features = df_global_importance.head(10)
        fig_importance = go.Figure(go.Bar(
            x=top_features['Mean_Absolute_SHAP'],
            y=top_features['Feature'],
            orientation='h',
            marker=dict(
                color=top_features['Mean_Absolute_SHAP'],
                colorscale='Turbo',
                showscale=True
            ),
            text=top_features['Mean_Absolute_SHAP'].apply(lambda x: f"<b>{x:.3f}</b>"),
            textposition='outside'
        ))

        # ✅ Updated layout: larger and bright axis fonts
        fig_importance.update_layout(
            title={'text': "✨ <b>Feature Importance Analysis</b>", 'x': 0.5},
            yaxis={
                'categoryorder': 'total ascending',
                'title': '<b>Feature</b>',
                'titlefont': dict(size=18, color='white'),
                'tickfont': dict(size=16, color='white')
            },
            xaxis={
                'title': '<b>Mean Absolute SHAP Value</b>',
                'titlefont': dict(size=18, color='white'),
                'tickfont': dict(size=16, color='white')
            },
            height=650,
            plot_bgcolor='#1E1E1E',  # dark background to make white font pop
            paper_bgcolor='#1E1E1E',  # keep consistent background
            font=dict(color='white'),  # default text color
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    else:
        st.warning("⚠️ No feature importance data available. Please ensure SHAP values have been calculated.")

# ============================
# 📊 Clinical Feature Insights (Tab 2)
# ============================
with tab2:
    
    # --------------------------------
    # 1️⃣ FEATURE SELECTION
    # --------------------------------
    # Get top 10 features by global importance
    available_features = df_global_importance['Feature'].head(10).tolist()
    
    # UI: Select feature for analysis
    st.markdown('<h5 class="subsection-title">🎯 Select Feature for Analysis</h5>', unsafe_allow_html=True)
    selected_feature = st.selectbox(
        "",
        options=available_features,
        index=0,
        label_visibility="collapsed"
    )
    
    # --------------------------------
    # 2️⃣ FETCH FEATURE INFORMATION
    # --------------------------------
    feature_info = df_global_importance[
        df_global_importance['Feature'] == selected_feature
    ].iloc[0]
    
    # Extract SHAP metrics for the selected feature
    mean_abs = feature_info['Mean_Absolute_SHAP']   # Importance magnitude
    mean_shap = feature_info['Mean_SHAP']           # Average directional effect
    std_shap = feature_info['Std_SHAP']             # Variability in effect
    
    # Determine if feature is binary or continuous
    if selected_feature in df_filtered.columns:
        unique_values = df_filtered[selected_feature].nunique()
        is_binary = unique_values == 2
        feature_values = df_filtered[selected_feature].unique()
    else:
        is_binary = False
        feature_values = []
    
    # --------------------------------
    # 3️⃣ FEATURE IMPACT ANALYSIS FUNCTION
    # --------------------------------
    def analyze_feature_impact(df_filtered, selected_feature, mean_shap):
        """
        Analyze the clinical impact of a feature using SHAP values.
        Handles binary and continuous features differently.
        Falls back to global SHAP if individual SHAPs are missing.
        """
        if f'{selected_feature}_SHAP' in df_filtered.columns:
            shap_values = df_filtered[f'{selected_feature}_SHAP']
            
            # Binary features: Compare SHAP impact for present vs absent
            if is_binary and selected_feature in df_filtered.columns:
                present_mask = df_filtered[selected_feature] == 1
                absent_mask = df_filtered[selected_feature] == 0
                
                if present_mask.sum() > 0 and absent_mask.sum() > 0:
                    shap_when_present = shap_values[present_mask].mean()
                    shap_when_absent = shap_values[absent_mask].mean()
                    
                    return {
                        'direction': 'risk' if shap_when_present > shap_when_absent else 'protective',
                        'present_impact': shap_when_present,
                        'absent_impact': shap_when_absent,
                        'difference': abs(shap_when_present - shap_when_absent)
                    }
            
            # Continuous features: Correlation between feature and SHAP values
            elif not is_binary and selected_feature in df_filtered.columns:
                correlation = df_filtered[selected_feature].corr(shap_values)
                
                return {
                    'direction': 'risk' if correlation > 0 else 'protective',
                    'correlation': correlation,
                    'strength': abs(correlation)
                }
        
        # Fallback: Only global mean SHAP available
        return {
            'direction': 'risk' if mean_shap > 0 else 'protective' if mean_shap < 0 else 'neutral',
            'global_mean': mean_shap
        }
    
    # Run impact analysis for the selected feature
    impact_analysis = analyze_feature_impact(df_filtered, selected_feature, mean_shap)
    
    # --------------------------------
    # 4️⃣ FEATURE DESCRIPTIONS DATABASE
    # --------------------------------
    # Educational descriptions for each feature
    feature_data = {
    'ADL': {
    'full_name': 'Activities of Daily Living',
    'what_is_it': (
        'Activities of Daily Living (ADL) assess an individual’s capacity to independently perform essential self-care '
        'tasks that are fundamental for daily functioning. These include eating, dressing, bathing, toileting, transferring '
        '(e.g., moving from bed to chair), and maintaining continence. ADL provides critical insights into a person’s functional '
        'status and overall level of independence.'
    ),
    'how_measured': (
        'ADL is typically evaluated using standardized clinical scales or structured interviews with patients and caregivers. '
        'Scores usually range from 0 to 10 (or similar scales), with higher scores indicating greater independence and lower '
        'need for assistance. The assessment combines direct observation, self-reports, and caregiver feedback to form a comprehensive picture.'
    )
    },
  
    'MMSE': {
    'full_name': 'Mini-Mental State Examination',
    'what_is_it': (
        'The Mini-Mental State Examination (MMSE) is a brief, widely used cognitive screening tool designed to evaluate global '
        'cognitive function. It assesses multiple cognitive domains, including orientation to time and place, immediate and delayed memory, '
        'attention and calculation, language abilities (such as naming, repetition, and comprehension), and visuospatial skills (such as copying a figure). '
        'The MMSE is frequently used to detect cognitive impairment, monitor disease progression, and assess treatment response in conditions such as dementia.'
    ),
    'how_measured': (
        'The MMSE is administered in a structured format, typically taking about 5–10 minutes to complete. It is scored out of 30 points, '
        'with higher scores indicating better cognitive function. Generally, a score of 24 or higher is considered normal, while lower scores '
        'suggest varying degrees of cognitive impairment. Performance is evaluated based on standardized questions and tasks conducted by a clinician.'
    )
    },

    'Age': {
    'full_name': 'Chronological Age',
    'what_is_it': (
        'Chronological age refers to the number of years a person has lived since birth. It is a fundamental demographic variable '
        'and a crucial determinant in health assessments, as it strongly influences the risk of developing many medical conditions, '
        'including neurodegenerative diseases such as Alzheimer’s disease. Age is often used to contextualize other clinical and cognitive findings.'
    ),
    'how_measured': (
        'Age is measured in complete years since the date of birth, as recorded in official documents or reported by the individual. '
        'It is considered one of the simplest yet most informative clinical parameters.'
    )
    },
    'MemoryComplaints': {
    'full_name': 'Subjective Memory Complaints',
    'what_is_it': (
        'Subjective Memory Complaints (SMCs) refer to an individual’s perception or an informant’s report of declining memory function. '
        'These complaints may include difficulties remembering recent events, misplacing items, or struggling to recall names and appointments. '
        'SMCs are important because they can represent early indicators of cognitive decline, even in the absence of measurable deficits on formal cognitive tests.'
    ),
    'how_measured': (
        'Typically assessed as a binary variable (Yes/No) through structured clinical interviews, standardized questionnaires, or informant-based reports. '
        'Patients are directly asked about perceived changes in their memory, and caregivers or family members may provide corroborating information.'
    )
    },
    'BehavioralProblems': {
    'full_name': 'Behavioral and Psychological Symptoms',
    'what_is_it': (
        'Behavioral and Psychological Symptoms refer to a wide range of non-cognitive disturbances commonly observed in individuals '
        'with neurodegenerative disorders such as Alzheimer’s disease. These symptoms can include agitation, aggression, irritability, '
        'wandering, sleep disturbances, depression, anxiety, hallucinations, and apathy. They significantly impact the quality of life '
        'of patients and caregivers, and are important targets for clinical management.'
    ),
    'how_measured': (
        'Typically assessed as a binary variable (Present/Absent) using structured caregiver interviews, standardized symptom checklists, '
        'or clinician observations. Tools such as the Neuropsychiatric Inventory (NPI) are often used to systematically evaluate and quantify these symptoms.'
    )
    },
    'PersonalityChanges': {
    'full_name': 'Personality Changes',
    'what_is_it': (
        'Personality Changes refer to notable shifts in an individual’s typical behavior patterns, emotional responses, and social interactions. '
        'These changes can manifest as increased apathy, irritability, disinhibition, reduced empathy, or socially inappropriate behaviors. '
        'Such alterations often reflect underlying neurological changes and can be among the earliest signs of certain types of dementia, including Alzheimer’s disease.'
    ),
    'how_measured': (
        'Assessed as a binary variable (Present/Absent), typically based on detailed reports from family members or close contacts who can compare current behavior '
        'to the individual’s longstanding personality traits (premorbid personality). Structured questionnaires and clinical interviews are used to systematically evaluate these changes.'
    )
    },

    'DifficultyCompletingTasks': {
    'full_name': 'Difficulties with Task Completion',
    'what_is_it': (
        'Difficulties with Task Completion refer to impairments in an individual’s ability to carry out familiar or routine activities that require '
        'executive functions such as planning, organizing, sequencing, and problem-solving. Examples include trouble managing finances, preparing meals, '
        'or following multi-step instructions. These difficulties often signal early executive dysfunction and are characteristic of cognitive decline in conditions like Alzheimer’s disease.'
    ),
    'how_measured': (
        'Assessed as a binary variable (Present/Absent), typically based on clinician observation or detailed caregiver reports describing the individual’s ability '
        'to perform daily tasks. Structured functional assessments or informant questionnaires (e.g., Functional Activities Questionnaire) are commonly used to support evaluation.'
    )
    },

    'Forgetfulness': {
    'full_name': 'Forgetfulness',
    'what_is_it': (
        'Forgetfulness refers to frequent or severe memory lapses that exceed what is typically expected with normal aging. '
        'These episodes can include difficulty remembering recent events, appointments, conversations, or the location of personal belongings. '
        'When persistent, forgetfulness can interfere with daily activities and is often an early clinical indicator of mild cognitive impairment or dementia.'
    ),
    'how_measured': (
        'Assessed as a binary variable (Present/Absent), based on information gathered during clinical interviews and corroborated by caregiver or family member reports. '
        'Structured questionnaires or cognitive screening tools may be used to document the frequency, severity, and impact of these memory lapses on daily life.'
    )
    },

    'Hypertension': {
    'full_name': 'Hypertension',
    'what_is_it': (
        'Hypertension refers to persistently elevated blood pressure levels, which can have widespread negative effects on cardiovascular and brain health. '
        'It is a major modifiable risk factor for stroke, heart disease, and vascular contributions to cognitive impairment and dementia. '
        'In the context of cognitive decline, hypertension can lead to chronic cerebral hypoperfusion and microvascular damage, contributing to neurodegeneration.'
    ),
    'how_measured': (
        'Typically assessed as a binary variable (Present/Absent), based on a documented clinical diagnosis or repeated blood pressure measurements showing sustained '
        'readings above 130/80 mmHg, in line with current clinical guidelines. Blood pressure is measured using a sphygmomanometer or automated devices, and diagnoses may be '
        'confirmed through medical records or physician evaluation.'
    )
    },

    'FunctionalAssessment': {
    'full_name': 'Functional Assessment',
    'what_is_it': (
        'Functional Assessment evaluates an individual’s overall ability to perform activities essential for independent living, '
        'including both basic self-care (e.g., eating, dressing) and more complex instrumental tasks (e.g., managing finances, shopping, medication management). '
        'It provides a holistic measure of functional capacity and is critical for determining care needs and disease impact.'
    ),
    'how_measured': (
        'Scores generally range where higher values indicate better functional ability and independence. Score ranges from 0 to 10. Lower scores reflect greater impairment in the ability to perform daily and instrumental activities, indicating reduced functional independence.'
    )
}

}
    
    # --------------------------------
    # 5️⃣ FALLBACK DESCRIPTION (if feature not in database)
    # --------------------------------
    if selected_feature in feature_data:
        desc = feature_data[selected_feature]
        full_name = desc['full_name']
        what_is_it = desc['what_is_it']
        how_measured = desc['how_measured']
    else:
        full_name = selected_feature.replace('_', ' ').title()
        what_is_it = f"{selected_feature} is included in the model based on statistical association with the outcome."
        how_measured = f"Measurement depends on the nature of {selected_feature}."
    
    # --------------------------------
    # 6️⃣ IMPORTANCE LEVEL DESCRIPTION
    # --------------------------------
    feature_rank = available_features.index(selected_feature) + 1
    if mean_abs > 0.3:
        importance_desc = f"**Critical importance**, ranked #{feature_rank} of {len(available_features)} features."
        importance_level = "🌟 Critical Importance"
        importance_color = "#dc2626"
    elif mean_abs > 0.15:
        importance_desc = f"**High importance**, ranked #{feature_rank} of {len(available_features)} features."
        importance_level = "🔥 High Importance"
        importance_color = "#ea580c"
    elif mean_abs > 0.05:
        importance_desc = f"**Moderate importance**, ranked #{feature_rank} of {len(available_features)} features."
        importance_level = "⭐ Moderate Importance"
        importance_color = "#d97706"
    else:
        importance_desc = f"**Lower importance**, ranked #{feature_rank} of {len(available_features)} features."
        importance_level = "🔸 Lower Importance"
        importance_color = "#0891b2"
    
    # --------------------------------
    # 7️⃣ DISPLAY FEATURE ANALYSIS CARD
    # --------------------------------
    st.markdown(f"""
    <div class="feature-detail-card" style="border-top: 6px solid {importance_color};">
        <!-- Header -->
        <div class="feature-header">
            <div class="rank-badge" style="background: linear-gradient(135deg, {importance_color}, {importance_color}dd);">
                {feature_rank}
            </div>
            <div>
                <h2>{selected_feature}</h2>
                <div>{full_name}</div>
            </div>
        </div>
        <!-- What is it -->
        <div class="content-section">
            <h3>📊 What is this feature?</h3>
            <p>{what_is_it}</p>
        </div>
        <!-- Measurement -->
        <div class="content-section">
            <h3>📏 How is it measured?</h3>
            <p>{how_measured}</p>
        </div>
        <!-- Model Importance -->
        <div class="content-section">
            <h3>⭐ Model Importance</h3>
            <p>{importance_desc} (Mean Absolute SHAP: {mean_abs:.3f})</p>
        </div>
        <!-- Importance Metric -->
        <div class="metric-box_D" style="border: 2px solid {importance_color};">
            <div class="metric-value_D" style="color: {importance_color};">
                {mean_abs:.3f}
            </div>
            <div class="metric-label_D" style="color: {importance_color};">
                {importance_level}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab3:

    # Individual Patient Profile - Tab 3: Patient Selector & Profile Display
    # Display a title for patient selection
    st.markdown("<p style='font-size: 1.4rem; font-weight: bold; margin-bottom: -0.5rem;'>Select Patient ID:</p>", unsafe_allow_html=True)

    # Dropdown to select patient ID from filtered dataframe
    patient_id = st.selectbox(
        "Choose a Patient ID to view their profile and risk analysis:",
        options=df_filtered['Patient_ID'].tolist()
    )

    # Only proceed if a patient ID is selected
    if patient_id:
        # Get data for the selected patient
        patient_data = df_filtered[df_filtered['Patient_ID'] == patient_id].iloc[0]

        # Extract predicted probability and diagnosis
        prob = patient_data['Prediction_Probability']
        predicted_diagnosis = patient_data['Predicted_Diagnosis']

        # Determine risk level and display style based on predicted diagnosis and probability
        if predicted_diagnosis == 0:
            # Prediction = 0 → No risk
            risk_level = 'No Risk'
            risk_class = 'risk-none'
            risk_emoji = '✅'
        elif predicted_diagnosis == 1:
            # Prediction = 1 → classify by probability
            if prob < 0.7:
                risk_level = 'Low Risk'
                risk_class = 'risk-low'
                risk_emoji = '🟢'
            elif prob < 0.9:
                risk_level = 'Medium Risk'
                risk_class = 'risk-medium'
                risk_emoji = '🟡'
            else:
                risk_level = 'High Risk'
                risk_class = 'risk-high'
                risk_emoji = '🔴'

        # Human-readable diagnosis and confidence
        predicted_diagnosis = "Positive" if patient_data['Predicted_Diagnosis'] == 1 else "Negative"
        confidence = patient_data['Prediction_Confidence']

        # Split layout into two columns: left for summary card, right for details
        col1, col2 = st.columns([1, 2])

        with col1:
            # Display patient info card with diagnosis, probability, confidence, and risk
            st.markdown(f"""
            <div class="info-card" style="text-align: center; font-size: 1.2rem; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">
                <h2 style="font-size: 2.2rem; color: #1e293b; margin-bottom: 1rem; font-weight: 700;">Patient ID {patient_id}</h2>
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 0rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin-bottom: 0.5rem;"><strong>Predicted Diagnosis:</strong> <span style="font-weight: bold; font-size: 1.4rem; color: {'#ef4444' if predicted_diagnosis == 'Positive' else '#10b981'};">{predicted_diagnosis}</span></p>
                </div>
                <div style="background: #f1f5f9; padding: 0.5rem; border-radius: 1px; margin: 1rem 0; border-left: 4px solid #3b82f6;">
                    <p style="font-size: 1.1rem; color: #475569; margin: 0; font-style: italic;">This patient has a <strong>{risk_level.lower()}</strong> risk of developing Alzheimer's disease.</p>
                </div>  
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin: 0;"><strong>Prediction Probability:</strong> <span style="font-size: 1.4rem; font-weight: 700; color: #059669;">{prob:.1%}</span></p>
                </div>
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin: 0;"><strong>Confidence:</strong> <span style="font-size: 1.4rem; font-weight: 700; color: #059669;">{confidence:.1%}</span></p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <span class="{risk_class}" style="font-size: 1rem; font-weight: 700; padding: 0.5rem 0.3rem; border-radius: 5px; display: inline-block; text-transform: uppercase; letter-spacing: 1px;">{risk_emoji} {risk_level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Display header or placeholder for comprehensive patient profile
            st.markdown(f"View Comprehensive Patient Profile") 

            # Identify all RAW_* columns for the current patient for further detail display
            raw_cols = [col for col in patient_data.index if col.startswith('RAW_')]

            # Organize patient details into four tabs for clarity
            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
                "Demographics & Clinical", 
                "Medical Risk Factors", 
                "Lifestyle & Symptoms", 
                "Lab Results"
            ])

            # --------------------------------------
            # Tab 1: Demographics & Clinical
            # --------------------------------------
            with detail_tab1:
                # Split into two columns for side-by-side display
                col1_demo, col2_demo = st.columns(2)

                with col1_demo:
                    st.markdown("### 👤 Demographics")
                    # Mapping of RAW columns to display name and unit
                    demographics = {
                        'RAW_Age': ('Age', 'years'),
                        'RAW_Gender': ('Gender', None),
                        'RAW_Ethnicity': ('Ethnicity', None),
                        'RAW_EducationLevel': ('Education Level', None)
                    }

                    for col, (display_name, unit) in demographics.items():
                        if col in raw_cols:
                            val = patient_data[col]

                            # Convert coded gender to human-readable
                            if col == 'RAW_Gender' and pd.notna(val):
                                val_str = str(val).strip()
                                val = "Female" if val_str in ['0', '0.0'] else "Male"

                            # Convert coded ethnicity to descriptive string
                            if col == 'RAW_Ethnicity' and pd.notna(val):
                                val_str = str(val).strip()
                                mapping = {'0': 'Caucasian', '1': 'African American', '2': 'Asian', '3': 'Other'}
                                val = mapping.get(val_str, val)

                            # Convert education level code to readable text
                            if col == 'RAW_EducationLevel' and pd.notna(val):
                                val_str = str(val).strip()
                                mapping = {'0': "No formal education", '1': "High School Diploma", 
                                        '2': "Bachelor's Degree", '3': "Master's Degree"}
                                val = mapping.get(val_str, val)

                            # Format value with units if applicable
                            display_val = f"{val} {unit}" if unit and pd.notna(val) else val if pd.notna(val) else "Not available"

                            st.markdown(f"**{display_name}:** {display_val}")

                with col2_demo:
                    st.markdown("### 🏥 Clinical Assessment")
                    # Mapping for clinical metrics with display name, unit, and interpretation
                    clinical = {
                        'RAW_MMSE': ('MMSE Score', '/30', 'higher_better'),
                        'RAW_FunctionalAssessment': ('Functional Assessment', '/10', 'higher_better'),
                        'RAW_ADL': ('Activities of Daily Living', '/10', 'higher_better'),
                        'RAW_BMI': ('BMI', 'kg/m²', 'range')
                    }

                    for col, (display_name, unit, indicator) in clinical.items():
                        if col in raw_cols:
                            val = patient_data[col]

                            if pd.notna(val) and isinstance(val, (int, float)):
                                # Assign visual status based on thresholds
                                if col == 'RAW_MMSE':
                                    status = "🟢" if val >= 24 else "🟡" if val >= 18 else "🔴"
                                elif col == 'RAW_FunctionalAssessment':
                                    status = "🟢" if val >= 9 else "🟡" if val >= 4 else "🔴"
                                elif col == 'RAW_ADL':
                                    status = "🟢" if val >= 9 else "🟡" if val >= 4 else "🔴"
                                elif col == 'RAW_BMI':
                                    if val < 18.5: status = "🔴 Underweight"
                                    elif val <= 24.9: status = "🟢 Normal"
                                    elif val <= 29.9: status = "🟡 Overweight"
                                    else: status = "🔴 Obese"
                                else:
                                    status = ""

                                st.markdown(f"**{display_name}:** {val:.1f}{unit} {status}")
                            else:
                                st.markdown(f"**{display_name}:** Not available")

            # --------------------------------------
            # Tab 2: Medical Risk Factors
            # --------------------------------------
            with detail_tab2:
                st.markdown("### Medical Risk Factors")

                # Mapping of medical risk factor columns to display name and icon
                risk_factors = {
                    'RAW_FamilyHistoryAlzheimers': ('Family History of Alzheimer\'s', '👨‍👩‍👧‍👦'),
                    'RAW_CardiovascularDisease': ('Cardiovascular Disease', '❤️'),
                    'RAW_Diabetes': ('Diabetes', '🩺'),
                    'RAW_Depression': ('Depression', '😔'),
                    'RAW_HeadInjury': ('Head Injury', '🤕'),
                    'RAW_Hypertension': ('Hypertension', '💊')
                }

                risk_count = 0
                risk_details = []

                for col, (display_name, icon) in risk_factors.items():
                    if col in raw_cols:
                        val = patient_data[col]
                        if pd.notna(val):
                            if val == 1:
                                risk_count += 1
                                risk_details.append(f"{icon} **{display_name}**: ✅ Present")
                            else:
                                risk_details.append(f"{icon} **{display_name}**: ❌ Absent")

                # Determine overall risk color/level
                if risk_count == 0: risk_color, risk_level = "#10b981", "Low"
                elif risk_count <= 2: risk_color, risk_level = "#f59e0b", "Moderate"
                else: risk_color, risk_level = "#ef4444", "High"

                # Display risk summary
                st.markdown(f"""
                <div style="background-color: {risk_color}20; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Overall Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
                    <br><strong>Risk Factors Present:</strong> {risk_count} of {len(risk_factors)}
                </div>
                """, unsafe_allow_html=True)

                # Display individual risk factor statuses
                for detail in risk_details:
                    st.markdown(detail)

            # --------------------------------------
            # Tab 3: Lifestyle & Symptoms
            # --------------------------------------
            with detail_tab3:
                col1_life, col2_life = st.columns(2)
                
                # Left column: Lifestyle factors display
                with col1_life:
                    st.markdown("### 🏃 Lifestyle Factors")  # Heading for the lifestyle section
                    
                    # Dictionary mapping column names in patient data to:
                    # (Display name, icon emoji, value type for interpretation)
                    lifestyle = {
                        'RAW_Smoking': ('Smoking Status', '🚬', 'binary'),
                        'RAW_AlcoholConsumption': ('Alcohol Consumption', '🍷', 'units_per_week'),
                        'RAW_PhysicalActivity': ('Physical Activity', '🏃', 'hours_per_week'),
                        'RAW_DietQuality': ('Diet Quality', '🥗', 'diet_score'),
                        'RAW_SleepQuality': ('Sleep Quality', '😴', 'sleep_score')
                    }
                    
                    # Loop through each lifestyle factor and display patient-specific values
                    for col, (display_name, icon, val_type) in lifestyle.items():
                        if col in raw_cols:  # Ensure the column exists in the dataset
                            val = patient_data[col]  # Get patient value for this lifestyle factor
                            if pd.notna(val):  # Skip if value is NaN/missing
                                
                                # Case 1: Binary variable (Yes/No) e.g., Smoking Status
                                if val_type == 'binary':
                                    val_display = "Yes" if val == 1 else "No"
                                    # Color code: Red for unhealthy (Yes), Green for healthy (No)
                                    color = "#ef4444" if val == 1 else "#10b981"
                                    st.markdown(
                                        f"{icon} **{display_name}:** <span style='color: {color}'>{val_display}</span>",
                                        unsafe_allow_html=True
                                    )
                                
                                # Case 2: Alcohol consumption (units per week)
                                elif val_type == 'units_per_week':
                                    # Risk categories: <=14 low risk, <=17 moderate risk, else high risk
                                    if isinstance(val, (int, float)):
                                        if val <= 14:
                                            status = "🟢 Low risk"
                                        elif val <= 17:
                                            status = "🟡 Moderate risk"
                                        else:
                                            status = "🔴 High risk"
                                        # Normalize to max of 20 units/week for progress bar
                                        progress = min(max(val / 20.0, 0.0), 1.0)
                                        st.markdown(f"{icon} **{display_name}:** {val:.1f} units/week {status}")
                                        st.progress(progress)
                                
                                # Case 3: Physical activity (hours per week)
                                elif val_type == 'hours_per_week':
                                    # Recommended >= 2.5 hours per week
                                    if isinstance(val, (int, float)):
                                        if val >= 2.5:
                                            status = "🟢 Good"
                                        elif val >= 1:
                                            status = "🟡 Fair"
                                        else:
                                            status = "🔴 Low"
                                        # Normalize to max of 10 hours/week for progress bar
                                        progress = min(max(val / 10.0, 0.0), 1.0)
                                        st.markdown(f"{icon} **{display_name}:** {val:.1f} hours/week {status}")
                                        st.progress(progress)
                                
                                # Case 4: Diet quality score (0–10)
                                elif val_type == 'diet_score':
                                    if isinstance(val, (int, float)):
                                        if val >= 7:
                                            status = "🟢 Excellent"
                                        elif val >= 5:
                                            status = "🟡 Good"
                                        else:
                                            status = "🔴 Poor"
                                        # Normalize to max score of 10 for progress bar
                                        progress = min(max(val / 10.0, 0.0), 1.0)
                                        st.markdown(f"{icon} **{display_name}:** {val:.1f}/10 {status}")
                                        st.progress(progress)
                                
                                # Case 5: Sleep quality score (4–10)
                                elif val_type == 'sleep_score':
                                    if isinstance(val, (int, float)):
                                        if val >= 8:
                                            status = "🟢 Excellent"
                                        elif val >= 6:
                                            status = "🟡 Good"
                                        else:
                                            status = "🔴 Poor"
                                        # Normalize 4–10 to 0–1 scale for progress bar
                                        progress = min(max((val - 4) / 6.0, 0.0), 1.0)
                                        st.markdown(f"{icon} **{display_name}:** {val:.1f}/10 {status}")
                                        st.progress(progress)

                with col2_life:
                    st.markdown("### 🧠 Cognitive Symptoms")
                    symptoms = {
                        'RAW_MemoryComplaints': ('Memory Complaints', '🧩'),
                        'RAW_BehavioralProblems': ('Behavioral Problems', '😤'),
                        'RAW_Confusion': ('Confusion', '😵'),
                        'RAW_Disorientation': ('Disorientation', '🧭'),
                        'RAW_PersonalityChanges': ('Personality Changes', '🎭'),
                        'RAW_DifficultyCompletingTasks': ('Difficulty Completing Tasks', '📝'),
                        'RAW_Forgetfulness': ('Forgetfulness', '💭')
                    }

                    symptom_count = 0
                    for col, (display_name, icon) in symptoms.items():
                        if col in raw_cols:
                            val = patient_data[col]
                            if pd.notna(val):
                                if val == 1:
                                    symptom_count += 1
                                    st.markdown(f"{icon} **{display_name}:** ⚠️ Present")
                                else:
                                    st.markdown(f"{icon} **{display_name}:** ✅ Absent")

                    # Summary of cognitive symptoms
                    if symptom_count > 0:
                        st.markdown(f"""
                        <div style="background-color: #f59e0b20; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <strong>Total Symptoms:</strong> {symptom_count} of {len(symptoms)}
                        </div>
                        """, unsafe_allow_html=True)

            # --------------------------------------
            # Tab 4: Lab Results
            # --------------------------------------
            with detail_tab4:
                st.markdown("### 🔬 Clinical Measurements")
                col1_lab, col2_lab = st.columns(2)

                with col1_lab:
                    st.markdown("**Blood Pressure**")
                    if 'RAW_SystolicBP' in raw_cols and 'RAW_DiastolicBP' in raw_cols:
                        systolic = patient_data.get('RAW_SystolicBP', 'N/A')
                        diastolic = patient_data.get('RAW_DiastolicBP', 'N/A')

                        if pd.notna(systolic) and pd.notna(diastolic):
                            # Classify BP stage
                            if systolic < 120 and diastolic < 80: bp_status = "Normal 🟢"
                            elif systolic < 130 and diastolic < 80: bp_status = "Elevated 🟡"
                            elif systolic < 140 or diastolic < 90: bp_status = "Stage 1 HTN 🟠"
                            else: bp_status = "Stage 2 HTN 🔴"

                            st.metric(f"{systolic:.0f}/{diastolic:.0f} mmHg", bp_status)
                        else:
                            st.write("Blood Pressure: Not available")

                with col2_lab:
                    st.markdown("**Cholesterol Panel**")
                    cholesterol_values = {
                        'RAW_CholesterolTotal': ('Cholesterol Total', 'mg/dL', 200),
                        'RAW_CholesterolLDL': ('Cholesterol LDL', 'mg/dL', 100),
                        'RAW_CholesterolHDL': ('Cholesterol HDL', 'mg/dL', 40),
                        'RAW_CholesterolTriglycerides': ('Cholesterol Triglycerides', 'mg/dL', 150)
                    }

                    for col, (label, unit, threshold) in cholesterol_values.items():
                        if col in raw_cols:
                            val = patient_data[col]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                # HDL: higher better; others: lower better
                                color = "#10b981" if (val >= threshold if col=='RAW_CholesterolHDL' else val <= threshold) else "#ef4444"
                                st.markdown(f"**{label}:** <span style='color: {color}'>{val:.0f} {unit}</span>", unsafe_allow_html=True)

        # Actual vs Predicted Diagnosis (if available)

        col1 = st.columns(1)[0]  # Create a single column for SHAP analysis display

        # -----------------------------
        # Identify SHAP columns
        # -----------------------------
        # Columns that are NOT prefixed with RAW_ and NOT in excluded list are considered SHAP columns
        excluded_cols = [
            'Patient_ID', 'Predicted_Diagnosis', 'Prediction_Probability',
            'Prediction_Confidence', 'Actual_Diagnosis', 'Correct_Prediction',
            'Prediction_Error'
        ]

        # Add any ID-related columns (like patient_id, ID, id) to excluded list
        id_columns = [col for col in df_filtered.columns if 'id' in col.lower()]
        excluded_cols.extend(id_columns)

        # Get all RAW_ columns (original feature values)
        raw_columns = [col for col in df_filtered.columns if col.startswith('RAW_')]

        # SHAP columns are everything else except excluded and RAW_ columns
        shap_columns = [col for col in df_filtered.columns if col not in excluded_cols and col not in raw_columns]

        # Optional: Debug print to see which columns are being considered for SHAP
        # st.write("SHAP columns:", shap_columns)

        # -----------------------------
        # Define binary features for formatting
        # -----------------------------
        binary_features = [
            'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
            'Depression', 'HeadInjury', 'Hypertension', 'Smoking',
            'AlcoholConsumption', 'MemoryComplaints', 'BehavioralProblems',
            'Confusion', 'Disorientation', 'PersonalityChanges',
            'DifficultyCompletingTasks', 'Forgetfulness'
        ]

        # -----------------------------
        # Function to format feature values
        # -----------------------------
        def format_feature_value(feature_name, value, raw_value=None):
            """Format feature values for display in SHAP analysis"""
            
            # Handle missing values
            if pd.isna(value) or value == 'N/A':
                return 'N/A'
            
            # Use RAW value if available
            display_value = raw_value if raw_value is not None and pd.notna(raw_value) else value

            # Format binary features as Present / Absent
            if any(bf in feature_name for bf in binary_features):
                if isinstance(display_value, (int, float)):
                    return "Present" if int(display_value) == 1 else "Absent"
                elif isinstance(display_value, str):
                    try:
                        numeric_val = float(display_value)
                        return "Present" if int(numeric_val) == 1 else "Absent"
                    except:
                        return str(display_value)
            
            # Format Gender
            if 'Gender' in feature_name:
                if isinstance(display_value, (int, float)):
                    return "Female" if int(display_value) == 0 else "Male"
                elif isinstance(display_value, str):
                    try:
                        numeric_val = float(display_value)
                        return "Female" if int(numeric_val) == 0 else "Male"
                    except:
                        return str(display_value)
            
            # Format numeric features with units
            if isinstance(display_value, (int, float)):
                if 'Age' in feature_name:
                    return f"{int(display_value)} years"
                elif 'BMI' in feature_name:
                    return f"{display_value:.1f} kg/m²"
                elif 'BP' in feature_name:
                    return f"{display_value:.0f} mmHg"
                elif 'Cholesterol' in feature_name:
                    return f"{display_value:.0f} mg/dL"
                elif 'MMSE' in feature_name:
                    return f"{display_value:.0f}/30"
                elif 'ADL' in feature_name or 'FunctionalAssessment' in feature_name:
                    return f"{display_value:.0f}/10"
                elif any(scale in feature_name for scale in ['PhysicalActivity', 'DietQuality', 'SleepQuality']):
                    return f"{display_value:.0f}/10"
                else:
                    # General numeric formatting
                    if display_value == int(display_value):
                        return f"{int(display_value)}"
                    else:
                        return f"{display_value:.2f}"

            # Convert string numbers to numeric recursively
            elif isinstance(display_value, str):
                try:
                    numeric_val = float(display_value)
                    return format_feature_value(feature_name, numeric_val, None)
                except:
                    return str(display_value)
            
            return str(display_value)

        # -----------------------------
        # SHAP Analysis for Patient
        # -----------------------------
        if shap_columns:
            # Extract SHAP values for this patient
            patient_shap_data = patient_data[shap_columns]

            # Convert all SHAP values to numeric, ignore errors
            patient_shap = pd.Series(dtype=float)
            for col in shap_columns:
                try:
                    value = pd.to_numeric(patient_data[col], errors='coerce')
                    if pd.notna(value):
                        patient_shap[col] = value
                except:
                    continue

            if len(patient_shap) > 0:
                # Get top 15 features by absolute SHAP value
                top_features_by_abs = patient_shap.reindex(patient_shap.abs().sort_values(ascending=False).index).head(15)
                # Split into positive (risk) and negative (protective) features
                top_positive = top_features_by_abs[top_features_by_abs > 0]
                top_negative = top_features_by_abs[top_features_by_abs < 0]

                # SHAP Analysis Header
                st.markdown("## 📊 SHAP Feature Analysis")
                st.markdown("*Understanding which features contribute most to this patient's prediction*")

                # Two columns: risk factors and protective factors
                col_pos, col_neg = st.columns(2)

                # Features where higher values are better
                higher_is_better_features = ['ADL', 'MMSE', 'FunctionalAssessment', 'PhysicalActivity', 'DietQuality', 'SleepQuality']

                # -----------------------------
                # Display Top Risk Factors (Positive SHAP)
                # -----------------------------
                with col_pos:
                    st.markdown("### 🔴 Top Risk Factors")
                    st.markdown("*Features increasing Alzheimer's risk*")

                    for feature, shap_value in top_positive.items():
                        if shap_value > 0.0:
                            # Get actual value from RAW_ column if exists
                            raw_feature_name = f'RAW_{feature}'
                            if raw_feature_name in patient_data.index:
                                actual_value = patient_data[raw_feature_name]
                            else:
                                actual_value = patient_data.get(feature, 'N/A')

                            # Format the feature value
                            if any(bf in feature for bf in binary_features):
                                if pd.notna(actual_value):
                                    if str(actual_value).strip() in ['1', '1.0']:
                                        formatted_value = "Present"
                                    elif str(actual_value).strip() in ['0', '0.0']:
                                        formatted_value = "Absent"
                                    else:
                                        formatted_value = str(actual_value)
                                else:
                                    formatted_value = "N/A"
                            else:
                                formatted_value = format_feature_value(feature, actual_value)

                            # Add safety indicator for better-is-higher features
                            safety_indicator = ""
                            if any(hb in feature for hb in higher_is_better_features) and pd.notna(actual_value):
                                try:
                                    val = float(actual_value)
                                    # ADL scale
                                    if 'ADL' in feature:
                                        if val >= 9:
                                            safety_indicator = " ✅ (Minimal/No impairment)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Mild impairment)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # Functional Assessment scale
                                    elif 'FunctionalAssessment' in feature:
                                        if val >= 9:
                                            safety_indicator = " ✅ (Minimal/No impairment)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Mild impairment)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # MMSE scale
                                    elif 'MMSE' in feature:
                                        if val >= 27:
                                            safety_indicator = " ✅ (No cognitive impairment)"
                                        elif val >= 24:
                                            safety_indicator = " 🟡 (Mild cognitive impairment)"
                                        elif val >= 18:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # Other scales (PhysicalActivity, DietQuality, SleepQuality)
                                    else:
                                        if val >= 8:
                                            safety_indicator = " ✅ (Excellent)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Good)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Fair)"
                                        else:
                                            safety_indicator = " ❌ (Poor)"
                                except:
                                    pass

                            # Add info text for tooltip
                            info_text = ""
                            if 'ADL' in feature:
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Activities of Daily Living (0-10): Lower scores = greater impairment</span>"
                            elif 'FunctionalAssessment' in feature:
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Functional Assessment (0-10): Lower scores = greater impairment</span>"
                            elif 'MMSE' in feature:
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Mini-Mental State Exam (0-30): Lower scores = cognitive impairment</span>"
                            elif any(scale in feature for scale in ['PhysicalActivity', 'DietQuality', 'SleepQuality']):
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Scale (0-10): Higher scores are better</span>"

                            # Display feature box with SHAP impact
                            st.markdown(f"""
                            <div style="background-color: rgba(239, 68, 68, 0.1); padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #ef4444;">
                                <strong style="color: #991b1b;">{feature.replace('_', ' ')}</strong><br>
                                <span style="color: #666;">Value: {formatted_value}{safety_indicator}</span>{info_text}<br>
                                <span style="color: #ef4444; font-weight: bold;">Impact: +{shap_value:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                # -----------------------------
                # Display Top Protective Factors (Negative SHAP)
                # -----------------------------
                with col_neg:
                    st.markdown("### 🟢 Top Protective Factors")
                    st.markdown("*Features decreasing Alzheimer's risk*")

                    # Loop through top features with negative SHAP values
                    for feature, shap_value in top_negative.items():
                        if shap_value < 0.0:
                            # Get actual feature value from RAW_ column if available
                            raw_feature_name = f'RAW_{feature}'
                            if raw_feature_name in patient_data.index:
                                actual_value = patient_data[raw_feature_name]
                            else:
                                actual_value = patient_data.get(feature, 'N/A')

                            # Format binary features as Present / Absent
                            if any(bf in feature for bf in binary_features):
                                if pd.notna(actual_value):
                                    if str(actual_value).strip() in ['1', '1.0']:
                                        formatted_value = "Present"
                                    elif str(actual_value).strip() in ['0', '0.0']:
                                        formatted_value = "Absent"
                                    else:
                                        formatted_value = str(actual_value)
                                else:
                                    formatted_value = "N/A"
                            else:
                                # Format non-binary features using helper function
                                formatted_value = format_feature_value(feature, actual_value)

                            # Add safety indicator for features where higher values are better
                            safety_indicator = ""
                            if any(hb in feature for hb in higher_is_better_features) and pd.notna(actual_value):
                                try:
                                    val = float(actual_value)
                                    # ADL scale interpretation
                                    if 'ADL' in feature:
                                        if val >= 9:
                                            safety_indicator = " ✅ (Minimal/No impairment)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Mild impairment)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # Functional Assessment interpretation
                                    elif 'FunctionalAssessment' in feature:
                                        if val >= 9:
                                            safety_indicator = " ✅ (Minimal/No impairment)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Mild impairment)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # MMSE cognitive scale
                                    elif 'MMSE' in feature:
                                        if val >= 27:
                                            safety_indicator = " ✅ (No cognitive impairment)"
                                        elif val >= 24:
                                            safety_indicator = " 🟡 (Mild cognitive impairment)"
                                        elif val >= 18:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # Other scales (PhysicalActivity, DietQuality, SleepQuality)
                                    else:
                                        if val >= 8:
                                            safety_indicator = " ✅ (Excellent)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Good)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Fair)"
                                        else:
                                            safety_indicator = " ❌ (Poor)"
                                except:
                                    pass

                            # Add descriptive info text for tooltip / context
                            info_text = ""
                            if 'ADL' in feature:
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Activities of Daily Living (0-10): Lower scores = greater impairment</span>"
                            elif 'FunctionalAssessment' in feature:
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Functional Assessment (0-10): Lower scores = greater impairment</span>"
                            elif 'MMSE' in feature:
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Mini-Mental State Exam (0-30): Lower scores = cognitive impairment</span>"
                            elif any(scale in feature for scale in ['PhysicalActivity', 'DietQuality', 'SleepQuality']):
                                info_text = "<br><span style='font-size: 1em; color: #666; font-style: italic;'>Scale (0-10): Higher scores are better</span>"

                            # Render the feature box with SHAP impact
                            st.markdown(f"""
                            <div style="background-color: rgba(16, 185, 129, 0.1); padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #10b981;">
                                <strong style="color: #064e3b;">{feature.replace('_', ' ')}</strong><br>
                                <span style="color: #666;">Value: {formatted_value}{safety_indicator}</span>{info_text}<br>
                                <span style="color: #10b981; font-weight: bold;">Impact: {shap_value:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                # -----------------------------
                # Feature contribution waterfall chart
                # -----------------------------
                st.subheader("📊 Feature Contribution Visualization")

                # Select top 10 features by absolute SHAP value
                feature_contributions = patient_shap.sort_values(key=abs, ascending=False).head(10)

                # Prepare lists for y-axis labels and hover texts
                labels = []
                hover_texts = []

                for feat in feature_contributions.index:
                    raw_feat_name = f'RAW_{feat}'

                    # -----------------------------
                    # Retrieve actual feature value
                    # -----------------------------
                    if raw_feat_name in patient_data.index:
                        actual_val = patient_data[raw_feat_name]
                    else:
                        actual_val = patient_data.get(feat, 'N/A')

                    # -----------------------------
                    # Format feature values for display
                    # -----------------------------
                    if any(bf in feat for bf in binary_features):
                        # Binary feature formatting
                        if pd.notna(actual_val):
                            if str(actual_val).strip() in ['1', '1.0']:
                                formatted_val = "Present"
                                emoji = "✓"
                            elif str(actual_val).strip() in ['0', '0.0']:
                                formatted_val = "Absent"
                                emoji = "✗"
                            else:
                                formatted_val = str(actual_val)
                                emoji = ""
                        else:
                            formatted_val = "N/A"
                            emoji = ""
                    else:
                        # Non-binary feature formatting
                        formatted_val = format_feature_value(feat, actual_val)
                        emoji = ""

                        # Add interpretation for specific scales (MMSE, ADL, FunctionalAssessment)
                        if pd.notna(actual_val):
                            try:
                                val = float(actual_val)
                                if 'MMSE' in feat:
                                    if val >= 27:
                                        emoji = "✅"
                                    elif val >= 24:
                                        emoji = "🟡"
                                    elif val >= 18:
                                        emoji = "⚠️"
                                    else:
                                        emoji = "❌"
                                elif 'ADL' in feat or 'FunctionalAssessment' in feat:
                                    if val >= 9:
                                        emoji = "✅"
                                    elif val >= 6:
                                        emoji = "🟡"
                                    elif val >= 4:
                                        emoji = "⚠️"
                                    else:
                                        emoji = "❌"
                            except:
                                pass

                    # -----------------------------
                    # Format feature label with optional emoji
                    # -----------------------------
                    feat_display = feat.replace('_', ' ')
                    if emoji:
                        label = f"{feat_display}<br><span style='font-size: 0.9em; color: #666;'>{emoji} {formatted_val}</span>"
                    else:
                        label = f"{feat_display}<br><span style='font-size: 0.9em; color: #666;'>{formatted_val}</span>"

                    labels.append(label)

                    # -----------------------------
                    # Create hover text for detailed info
                    # -----------------------------
                    shap_val = feature_contributions[feat]
                    impact = "increases" if shap_val > 0 else "decreases"
                    hover_text = f"<b>{feat_display}</b><br>Value: {formatted_val}<br>SHAP: {shap_val:.4f}<br>This {impact} Alzheimer's risk"
                    hover_texts.append(hover_text)

                # -----------------------------
                # Set bar colors based on SHAP value
                # Red for risk-increasing, Green for protective
                # -----------------------------
                colors = []
                for x in feature_contributions.values:
                    if x > 0:
                        # Red gradient
                        intensity = min(abs(x) / 0.2, 1)  # normalize to 0-1
                        colors.append(f'rgba(239, 68, 68, {0.4 + 0.6 * intensity})')
                    else:
                        # Green gradient
                        intensity = min(abs(x) / 0.2, 1)
                        colors.append(f'rgba(16, 185, 129, {0.4 + 0.6 * intensity})')

                # -----------------------------
                # Create horizontal bar chart using Plotly
                # -----------------------------
                fig_waterfall = go.Figure(go.Bar(
                    x=feature_contributions.values,
                    y=labels,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        line=dict(color='rgba(0,0,0,0.1)', width=1)
                    ),
                    text=[f"<b>{x:+.4f}</b>" for x in feature_contributions.values],  # Bold text
                    textposition='outside',
                    textfont=dict(size=12, color='black', family='Arial Black'),
                    hovertext=hover_texts,
                    hoverinfo='text'
                ))

                # -----------------------------
                # Update chart layout
                # -----------------------------
                fig_waterfall.update_layout(
                    title=dict(
                        text=f"<b>Top 15 Feature Contributions for Patient {patient_id}</b><br><span style='font-size: 14px; color: #666;'>How each feature impacts the Alzheimer's risk prediction</span>",
                        font=dict(size=18),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title="<b>SHAP Value</b> (Impact on Prediction)",
                        title_font=dict(size=16),
                        tickfont=dict(size=14),
                        gridcolor='rgba(128,128,128,0.2)',
                        zerolinecolor='rgba(128,128,128,0.4)',
                        zerolinewidth=2,
                        range=[min(feature_contributions.values) * 1.2, max(feature_contributions.values) * 1.2]
                    ),
                    yaxis=dict(tickfont=dict(size=14, color="black"), linecolor="black", automargin=True),
                    height=700,
                    margin=dict(l=250, r=120, t=100, b=80),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                # -----------------------------
                # Add colored risk zones for reference
                # -----------------------------
                fig_waterfall.add_vrect(x0=0, x1=max(feature_contributions.values) * 1.2,
                                        fillcolor="rgba(239, 68, 68, 0.05)", layer="below", line_width=0)
                fig_waterfall.add_vrect(x0=min(feature_contributions.values) * 1.2, x1=0,
                                        fillcolor="rgba(16, 185, 129, 0.05)", layer="below", line_width=0)

                # -----------------------------
                # Add annotations for risk direction
                # -----------------------------
                fig_waterfall.add_annotation(text="<b>← Protective Factors</b><br><span style='font-size: 11px;'>Decrease Risk</span>",
                                            xref="paper", yref="paper", x=0.15, y=-0.12, showarrow=False,
                                            font=dict(size=12, color="#10b981"), align="center")
                fig_waterfall.add_annotation(text="<b>Risk Factors →</b><br><span style='font-size: 11px;'>Increase Risk</span>",
                                            xref="paper", yref="paper", x=0.85, y=-0.12, showarrow=False,
                                            font=dict(size=12, color="#ef4444"), align="center")
                fig_waterfall.add_annotation(text="Baseline", x=0, y=-1, xref="x", yref="y",
                                            showarrow=False, font=dict(size=10, color="gray"), yshift=-20)

                # -----------------------------
                # Display chart in Streamlit
                # -----------------------------
                st.plotly_chart(fig_waterfall, use_container_width=True, key=f"patient_waterfall_{patient_id}")

                st.subheader("📋 Detailed Feature Impact Summary")

                # -----------------------------
                # Prepare summary dataframe
                # -----------------------------
                summary_data = []

                for feature in patient_shap.index:
                    shap_value = patient_shap[feature]

                    # Get actual feature value from RAW_ column if available
                    raw_feature_name = f'RAW_{feature}'
                    actual_value = patient_data[raw_feature_name] if raw_feature_name in patient_data.index else patient_data.get(feature, 'N/A')

                    # -----------------------------
                    # Format feature value
                    # -----------------------------
                    if any(bf in feature for bf in binary_features):
                        if pd.notna(actual_value):
                            if str(actual_value).strip() in ['1', '1.0']:
                                formatted_value = "✅ Present"
                            elif str(actual_value).strip() in ['0', '0.0']:
                                formatted_value = "❌ Absent"
                            else:
                                formatted_value = f"❓ {str(actual_value)}"
                        else:
                            formatted_value = "❓ N/A"
                    else:
                        formatted_value = format_feature_value(feature, actual_value)

                    # -----------------------------
                    # Enhanced clinical interpretation
                    # -----------------------------
                    interpretation = ""
                    if pd.notna(actual_value):
                        try:
                            val = float(actual_value)
                            if 'ADL' in feature or 'FunctionalAssessment' in feature:
                                if val >= 9:
                                    interpretation = " (✅ Minimal impairment)"
                                elif val >= 6:
                                    interpretation = " (🟡 Mild impairment)"
                                elif val >= 4:
                                    interpretation = " (⚠️ Moderate impairment)"
                                else:
                                    interpretation = " (🔴 Severe impairment)"
                            elif 'MMSE' in feature:
                                if val >= 27:
                                    interpretation = " (✅ No cognitive impairment)"
                                elif val >= 24:
                                    interpretation = " (🟡 Mild cognitive impairment)"
                                elif val >= 18:
                                    interpretation = " (⚠️ Moderate cognitive impairment)"
                                else:
                                    interpretation = " (🔴 Severe cognitive impairment)"
                            elif 'Age' in feature:
                                if val < 65:
                                    interpretation = " (✅ Younger adult)"
                                elif val < 75:
                                    interpretation = " (🟡 Older adult)"
                                elif val < 85:
                                    interpretation = " (⚠️ Elderly)"
                                else:
                                    interpretation = " (🔴 Very elderly)"
                            elif 'BMI' in feature:
                                if 18.5 <= val <= 24.9:
                                    interpretation = " (✅ Normal weight)"
                                elif 25 <= val <= 29.9:
                                    interpretation = " (🟡 Overweight)"
                                elif val >= 30:
                                    interpretation = " (⚠️ Obese)"
                                else:
                                    interpretation = " (🟡 Underweight)"
                        except:
                            pass

                    display_value = formatted_value + interpretation

                    # -----------------------------
                    # Determine impact level from SHAP magnitude
                    # -----------------------------
                    abs_shap = abs(shap_value)
                    if abs_shap > 0.15:
                        impact_level = "🔴 Very High"
                    elif abs_shap > 0.1:
                        impact_level = "🟠 High"
                    elif abs_shap > 0.05:
                        impact_level = "🟡 Medium"
                    elif abs_shap > 0.02:
                        impact_level = "🟢 Low"
                    else:
                        impact_level = "🔵 Minimal"

                    # -----------------------------
                    # Define clinical categories
                    # -----------------------------
                    demographic_features = ['Age', 'Gender', 'Ethnicity', 'EducationLevel']
                    lifestyle_features = ['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
                    medical_history = ['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']
                    cognitive_assessment = ['MemoryComplaints', 'BehavioralProblems','MMSE', 'FunctionalAssessment','ADL']
                    symptoms = ['Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']

                    def get_clinical_category(feature):
                        if feature in demographic_features:
                            return "📅 Demographic"
                        elif feature in lifestyle_features:
                            return "🚶 Lifestyle"
                        elif feature in medical_history:
                            return "🏥 Medical History"
                        elif feature in cognitive_assessment:
                            return "🧠 Cognitive"
                        elif feature in symptoms:
                            return "⚠️ Symptoms"
                        else:
                            return "📊 Other"

                    clinical_category = get_clinical_category(feature)

                    # -----------------------------
                    # Append row to summary
                    # -----------------------------
                    summary_data.append({
                        'Category': clinical_category,
                        'Feature': feature,
                        'Current Value': display_value,
                        'SHAP Impact': f"{shap_value:+.4f}",
                        'Impact Level': impact_level
                    })

                # -----------------------------
                # Convert to DataFrame and sort
                # -----------------------------
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values(
                    'SHAP Impact',
                    key=lambda x: abs(x.str.replace('+', '').str.replace('-', '').astype(float)),
                    ascending=False
                ).reset_index(drop=True)

                # -----------------------------
                # Display top 10 features
                # -----------------------------
                st.dataframe(summary_df.head(10), use_container_width=True, hide_index=True)

                # -----------------------------
                # Interpretation guide
                # -----------------------------
                with st.expander("📖 How to Interpret SHAP Values"):
                    st.markdown("""
                    **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to the model's prediction:
                    
                    - **Positive SHAP values** (red bars) → Features that **increase** the probability of Alzheimer's
                    - **Negative SHAP values** (green bars) → Features that **decrease** the probability of Alzheimer's
                    - **Larger absolute values** → Features with **stronger impact** on the prediction
                    
                    The values are additive: starting from the baseline prediction, each feature's SHAP value is added/subtracted to reach the final prediction for this patient.
                    """)

        # -----------------------------
        # Patient Recommendations System
        # -----------------------------
        

        def display_patient_recommendations(patient_data, patient_id):
            """Display personalized recommendations for a patient in Streamlit."""
            
            st.markdown("## 🎯 Personalized Recommendations")
            st.markdown(f"*Tailored intervention strategies for Patient {patient_id}*")
            
            # Generate recommendations
            recommendations = generate_patient_recommendations(patient_data)
            
            # Create recommendation tabs
            tabs = st.tabs([
                "🚨 Immediate Actions", 
                "🏃 Lifestyle Changes", 
                "🏥 Medical Follow-up", 
                "🧠 Cognitive Health", 
                "📊 Monitoring Plan"
            ])
            
            rec_tab1, rec_tab2, rec_tab3, rec_tab4, rec_tab5 = tabs
            
            # -----------------------------
            # Immediate Actions
            # -----------------------------
            with rec_tab1:
                st.markdown("### Priority Actions to Take Now")
                if recommendations.get('immediate_actions'):
                    for action in recommendations['immediate_actions']:
                        st.markdown(f"• {action}")
                else:
                    st.info("No immediate actions required. Continue with regular preventive care.")
            
            # -----------------------------
            # Lifestyle Modifications
            # -----------------------------
            with rec_tab2:
                st.markdown("### Lifestyle Modifications")
                lifestyle_recs = recommendations.get('lifestyle_modifications', [])
                
                if lifestyle_recs:
                    # Categorize recommendations
                    exercise_recs = [r for r in lifestyle_recs if any(word in r.lower() for word in ['exercise', 'walking', 'physical', 'activity', 'strength'])]
                    diet_recs = [r for r in lifestyle_recs if any(word in r.lower() for word in ['diet', 'mediterranean', 'omega', 'food', 'nutrition'])]
                    sleep_recs = [r for r in lifestyle_recs if any(word in r.lower() for word in ['sleep', 'rest', 'bedtime'])]
                    other_recs = [r for r in lifestyle_recs if r not in exercise_recs + diet_recs + sleep_recs]
                    
                    if exercise_recs:
                        st.markdown("#### 🏃 Physical Activity")
                        for rec in exercise_recs:
                            st.markdown(f"• {rec}")
                    
                    if diet_recs:
                        st.markdown("#### 🥗 Nutrition")
                        for rec in diet_recs:
                            st.markdown(f"• {rec}")
                    
                    if sleep_recs:
                        st.markdown("#### 😴 Sleep")
                        for rec in sleep_recs:
                            st.markdown(f"• {rec}")
                    
                    if other_recs:
                        st.markdown("#### 🔄 Other Lifestyle Changes")
                        for rec in other_recs:
                            st.markdown(f"• {rec}")
                else:
                    st.info("Current lifestyle appears optimal. Continue maintaining healthy habits.")
            
            # -----------------------------
            # Medical Follow-up
            # -----------------------------
            with rec_tab3:
                st.markdown("### Medical Follow-up Required")
                followups = recommendations.get('medical_followup', [])
                if followups:
                    for followup in followups:
                        st.markdown(f"• {followup}")
                else:
                    st.info("No specific medical follow-up needed beyond routine care.")
            
            # -----------------------------
            # Cognitive Health Interventions
            # -----------------------------
            with rec_tab4:
                st.markdown("### Cognitive Health Interventions")
                cognitive_recs = recommendations.get('cognitive_interventions', [])
                if cognitive_recs:
                    for intervention in cognitive_recs:
                        st.markdown(f"• {intervention}")
                else:
                    st.info("Continue current cognitive activities. Consider adding new challenging mental exercises.")
            
            # -----------------------------
            # Monitoring Plan
            # -----------------------------
            with rec_tab5:
                st.markdown("### Ongoing Monitoring Plan")
                monitoring_recs = recommendations.get('monitoring_suggestions', [])
                if monitoring_recs:
                    for suggestion in monitoring_recs:
                        st.markdown(f"• {suggestion}")
                else:
                    st.info("Standard monitoring schedule appropriate.")

        # -----------------------------
        # Display Recommendations Section
        # -----------------------------
        st.markdown("---")  # Separator line
        display_patient_recommendations(patient_data, patient_id)

# -------------------------
# Compare with other patients (Tab 4)
# -------------------------
with tab4:
    
    # Title and description of the comparison feature
    st.markdown(
        "<p style='font-size: 1.4rem; font-weight: bold; margin-bottom: -0.5rem;'>"
        "Compare two patients side by side to understand their differences in features and risk factors"
        "</p>",
        unsafe_allow_html=True
    )
    
    # --- Patient selection ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Dropdown for selecting the first patient
        patient_id_1 = st.selectbox(
            "Select First Patient:",
            options=df_filtered['Patient_ID'].tolist(),
            key="patient_1_select"
        )
    
    with col2:
        # Filter out the first selected patient so the second patient is always different
        available_patients = [p for p in df_filtered['Patient_ID'].tolist() if p != patient_id_1]
        
        # Dropdown for selecting the second patient
        patient_id_2 = st.selectbox(
            "Select Second Patient:",
            options=available_patients,
            key="patient_2_select"
        )
    
    # Only proceed if both patients are selected
    if patient_id_1 and patient_id_2:
        # Retrieve the complete row of data for both patients
        patient1_data = df_filtered[df_filtered['Patient_ID'] == patient_id_1].iloc[0]
        patient2_data = df_filtered[df_filtered['Patient_ID'] == patient_id_2].iloc[0]
        
        # Visual divider
        st.markdown("---")
        
        # Layout: First patient card | VS | Second patient card
        col1, col2, col3 = st.columns([2, 1, 2])
        
        # --- First Patient Card ---
        with col1:
            prob1 = patient1_data['Prediction_Probability']
            predicted_diagnosis1 = patient1_data['Predicted_Diagnosis']
            
            # Assign risk level, CSS class, and emoji based on probability thresholds
            if predicted_diagnosis1 == 0:
                risk_level1 = 'No Risk'
                risk_class1 = 'risk-none'
                risk_emoji1 = '✅'
            elif predicted_diagnosis1 == 1:
                if prob1 < 0.7:
                    risk_level1 = 'Low Risk'
                    risk_class1 = 'risk-low'
                    risk_emoji1 = '🟢'
                elif prob1 < 0.9:
                    risk_level1 = 'Medium Risk'
                    risk_class1 = 'risk-medium'
                    risk_emoji1 = '🟡'
                else:
                    risk_level1 = 'High Risk'
                    risk_class1 = 'risk-high'
                    risk_emoji1 = '🔴'
            
            predicted1 = "Positive" if predicted_diagnosis1 == 1 else "Negative"
            confidence1 = patient1_data['Prediction_Confidence']
          
            # Styled HTML card for Patient 1
            st.markdown(f"""
            <div class="info-card" style="text-align: center; font-size: 1.2rem; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">
                <h2 style="font-size: 2.2rem; color: #1e293b; margin-bottom: 1rem; font-weight: 700;">Patient ID {patient_id_1}</h2>
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 0rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin-bottom: 0.5rem;"><strong>Predicted Diagnosis:</strong> <span style="font-weight: bold; font-size: 1.4rem; color: {'#ef4444' if predicted1 == 'Positive' else '#10b981'};">{predicted1}</span></p>
                </div>
                <div style="background: #f1f5f9; padding: 0.5rem; border-radius: 1px; margin: 1rem 0; border-left: 4px solid #3b82f6;">
                    <p style="font-size: 1.1rem; color: #475569; margin: 0; font-style: italic;">This patient has a <strong>{risk_level1.lower()}</strong> risk of developing Alzheimer's disease.</p>
                </div>  
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin: 0;"><strong>Prediction Probability:</strong> <span style="font-size: 1.4rem; font-weight: 700; color: #059669;">{prob1:.1%}</span></p>
                </div>
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin: 0;"><strong>Confidence:</strong> <span style="font-size: 1.4rem; font-weight: 700; color: #059669;">{confidence1:.1%}</span></p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <span class="{risk_class1}" style="font-size: 1rem; font-weight: 700; padding: 0.5rem 0.3rem; border-radius: 5px; display: inline-block; text-transform: uppercase; letter-spacing: 1px;">{risk_emoji1} {risk_level1}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Middle column with "VS"
        with col2:
            st.markdown("""
            <div style="text-align: center; padding-top: 50px;">
                <h1>VS</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Second Patient Card ---
        with col3:
            prob2 = patient2_data['Prediction_Probability']
            predicted_diagnosis2 = patient2_data['Predicted_Diagnosis']
            
            # Assign risk level, CSS class, and emoji based on probability thresholds
            if predicted_diagnosis2 == 0:
                risk_level2 = 'No Risk'
                risk_class2 = 'risk-none'
                risk_emoji2 = '✅'
            elif predicted_diagnosis2 == 1:
                if prob2 < 0.7:
                    risk_level2 = 'Low Risk'
                    risk_class2 = 'risk-low'
                    risk_emoji2 = '🟢'
                elif prob2 < 0.9:
                    risk_level2 = 'Medium Risk'
                    risk_class2 = 'risk-medium'
                    risk_emoji2 = '🟡'
                else:
                    risk_level2 = 'High Risk'
                    risk_class2 = 'risk-high'
                    risk_emoji2 = '🔴'
            
            predicted2 = "Positive" if predicted_diagnosis2 == 1 else "Negative"
            confidence2 = patient2_data['Prediction_Confidence']
            
            # Styled HTML card for Patient 2
            st.markdown(f"""
            <div class="info-card" style="text-align: center; font-size: 1.2rem; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">
                <h2 style="font-size: 2.2rem; color: #1e293b; margin-bottom: 1rem; font-weight: 700;">Patient ID {patient_id_2}</h2>
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 0rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin-bottom: 0.5rem;"><strong>Predicted Diagnosis:</strong> <span style="font-weight: bold; font-size: 1.4rem; color: {'#ef4444' if predicted2 == 'Positive' else '#10b981'};">{predicted2}</span></p>
                </div>
                <div style="background: #f1f5f9; padding: 0.5rem; border-radius: 1px; margin: 1rem 0; border-left: 4px solid #3b82f6;">
                    <p style="font-size: 1.1rem; color: #475569; margin: 0; font-style: italic;">This patient has a <strong>{risk_level2.lower()}</strong> risk of developing Alzheimer's disease.</p>
                </div>  
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin: 0;"><strong>Prediction Probability:</strong> <span style="font-size: 1.4rem; font-weight: 700; color: #059669;">{prob2:.1%}</span></p>
                </div>
                <div style="background: white; padding: 0rem; border-radius: 1px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                    <p style="font-size: 1.3rem; margin: 0;"><strong>Confidence:</strong> <span style="font-size: 1.4rem; font-weight: 700; color: #059669;">{confidence2:.1%}</span></p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <span class="{risk_class2}" style="font-size: 1rem; font-weight: 700; padding: 0.5rem 0.3rem; border-radius: 5px; display: inline-block; text-transform: uppercase; letter-spacing: 1px;">{ risk_emoji2} {risk_level2}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


        # Feature comparison tabs - matching your patient profile structure
        st.markdown("### 📊 Comprehensive Patient Comparison")

        # Create tabs for patient comparison categories
        comparison_tabs = st.tabs(["Demographics & Clinical", "Medical Risk Factors", "Lifestyle & Symptoms", "Lab Results", "SHAP Analysis"])

        # Get all RAW columns for both patients to identify available data
        raw_cols = [col for col in patient1_data.index if col.startswith('RAW_')]

        # ---------------- Tab 1: Demographics & Clinical ----------------
        with comparison_tabs[0]:
            # Split layout into two columns: Demographics (left) and Clinical (right)
            col1_comp, col2_comp = st.columns(2)

            # ---------------- Left column: Demographics ----------------
            with col1_comp:
                st.markdown("### 👤 Demographics")
                demographics = {
                    'RAW_Age': ('Age', 'years'),
                    'RAW_Gender': ('Gender', None),
                    'RAW_Ethnicity': ('Ethnicity', None),
                    'RAW_EducationLevel': ('Education Level', None)
                }

                # Loop over each demographic field
                for col, (display_name, unit) in demographics.items():
                    if col in raw_cols:
                        val1 = patient1_data[col]
                        val2 = patient2_data[col]

                        # Function to format demographic values for display
                        def format_demographic_value(val, col_name):
                            # Map gender codes to labels
                            if col_name == 'RAW_Gender':
                                if pd.notna(val):
                                    val_str = str(val).strip()
                                    if val_str in ['0', '0.0']:
                                        return "Female"
                                    elif val_str in ['1', '1.0']:
                                        return "Male"
                            # Map ethnicity codes to labels
                            elif col_name == 'RAW_Ethnicity':
                                if pd.notna(val):
                                    val_str = str(val).strip()
                                    if val_str in ['0']:
                                        return "Caucasian"
                                    elif val_str in ['1']:
                                        return "African American"
                                    elif val_str in ['2']:
                                        return "Asian"
                                    elif val_str in ['3']:
                                        return "Other"
                            # Map education level codes to labels
                            elif col_name == 'RAW_EducationLevel':
                                if pd.notna(val):
                                    val_str = str(val).strip()
                                    if val_str in ['0']:
                                        return "No formal education"
                                    elif val_str in ['1']:
                                        return "High School Diploma"
                                    elif val_str in ['2']:
                                        return "Bachelor's Degree"
                                    elif val_str in ['3']:
                                        return "Master's Degree"

                            # Default formatting with unit
                            if unit and pd.notna(val):
                                return f"{val} {unit}"
                            else:
                                return val if pd.notna(val) else "Not available"

                        # Format both patients' values
                        val1_display = format_demographic_value(val1, col)
                        val2_display = format_demographic_value(val2, col)

                        # Determine comparison icon (match vs difference)
                        comparison_icon = "✅" if val1 == val2 else "🔄"

                        # Display demographic comparison
                        st.markdown(f"""
                        <div style="background-color: #f8fafc; padding: 10px; margin: 5px 0; border-radius: 5px;">
                            <strong>{display_name}:</strong> {comparison_icon}<br>
                            <span style="color: #666;">Patient {patient_id_1}:</span> {val1_display}<br>
                            <span style="color: #666;">Patient {patient_id_2}:</span> {val2_display}
                        </div>
                        """, unsafe_allow_html=True)

            # ---------------- Right column: Clinical Assessment ----------------
            with col2_comp:
                st.markdown("### 🏥 Clinical Assessment")
                clinical = {
                    'RAW_MMSE': ('MMSE Score', '/30', 'higher_better'),
                    'RAW_FunctionalAssessment': ('Functional Assessment', '/10', 'higher_better'),
                    'RAW_ADL': ('Activities of Daily Living', '/10', 'higher_better'),
                    'RAW_BMI': ('BMI', 'kg/m²', 'range')
                }

                # Loop over each clinical field
                for col, (display_name, unit, indicator) in clinical.items():
                    if col in raw_cols:
                        val1 = patient1_data[col]
                        val2 = patient2_data[col]

                        # Function to format clinical values with status indicators
                        def format_clinical_value(val, col_name):
                            if pd.notna(val) and isinstance(val, (int, float)):
                                status = ""
                                # MMSE score interpretation
                                if col_name == 'RAW_MMSE':
                                    if val >= 24:
                                        status = "🟢"
                                    elif val >= 18:
                                        status = "🟡"
                                    else:
                                        status = "🔴"
                                # Functional assessment and ADL interpretation
                                elif col_name in ['RAW_FunctionalAssessment', 'RAW_ADL']:
                                    if val >= 9:
                                        status = "🟢"
                                    elif val >= 4:
                                        status = "🟡"
                                    else:
                                        status = "🔴"
                                # BMI categories
                                elif col_name == 'RAW_BMI':
                                    if val < 18.5:
                                        status = "🔴 Underweight"
                                    elif 18.5 <= val <= 24.9:
                                        status = "🟢 Normal"
                                    elif 25 <= val <= 29.9:
                                        status = "🟡 Overweight"
                                    elif val >= 30:
                                        status = "🔴 Obese"
                                return f"{val:.1f}{unit} {status}"
                            else:
                                return "Not available"

                        # Format both patients' clinical values
                        val1_display = format_clinical_value(val1, col)
                        val2_display = format_clinical_value(val2, col)

                        # Comparison icon
                        comparison_icon = "✅" if val1 == val2 else "🔄"

                        # Display clinical comparison
                        st.markdown(f"""
                        <div style="background-color: #f8fafc; padding: 10px; margin: 5px 0; border-radius: 5px;">
                            <strong>{display_name}:</strong> {comparison_icon}<br>
                            <span style="color: #666;">Patient {patient_id_1}:</span> {val1_display}<br>
                            <span style="color: #666;">Patient {patient_id_2}:</span> {val2_display}
                        </div>
                        """, unsafe_allow_html=True)

                # ---------------- Tab 2: Medical Risk Factors ----------------
                with comparison_tabs[1]:
                    st.markdown("### 🏥 Medical Risk Factors Comparison")
                    
                    # Define medical risk factors and associated icons
                    risk_factors = {
                        'RAW_FamilyHistoryAlzheimers': ('Family History of Alzheimer\'s', '👨‍👩‍👧‍👦'),
                        'RAW_CardiovascularDisease': ('Cardiovascular Disease', '❤️'),
                        'RAW_Diabetes': ('Diabetes', '🩺'),
                        'RAW_Depression': ('Depression', '😔'),
                        'RAW_HeadInjury': ('Head Injury', '🤕'),
                        'RAW_Hypertension': ('Hypertension', '💊')
                    }
                    
                    # Initialize counters for the number of present risk factors per patient
                    risk_count_1 = 0
                    risk_count_2 = 0
                    
                    # Split layout into two columns for side-by-side comparison
                    col1_risk, col2_risk = st.columns(2)
                    
                    # ---------------- Left column: Patient 1 ----------------
                    with col1_risk:
                        st.markdown(f"#### Patient {patient_id_1}")
                        for col, (display_name, icon) in risk_factors.items():
                            if col in raw_cols:
                                val = patient1_data[col]
                                if pd.notna(val):
                                    if val == 1:  # Risk factor present
                                        risk_count_1 += 1
                                        st.markdown(f"{icon} **{display_name}**: ✅ Present")
                                    else:  # Risk factor absent
                                        st.markdown(f"{icon} **{display_name}**: ❌ Absent")
                    
                    # ---------------- Right column: Patient 2 ----------------
                    with col2_risk:
                        st.markdown(f"#### Patient {patient_id_2}")
                        for col, (display_name, icon) in risk_factors.items():
                            if col in raw_cols:
                                val = patient2_data[col]
                                if pd.notna(val):
                                    if val == 1:  # Risk factor present
                                        risk_count_2 += 1
                                        st.markdown(f"{icon} **{display_name}**: ✅ Present")
                                    else:  # Risk factor absent
                                        st.markdown(f"{icon} **{display_name}**: ❌ Absent")
                    
                    # ---------------- Risk summary comparison ----------------
                    st.markdown("---")  # Horizontal separator
                    col1_summary, col2_summary = st.columns(2)
                    
                    # Summary for Patient 1
                    with col1_summary:
                        # Determine risk level color and label based on number of present risk factors
                        if risk_count_1 == 0:
                            risk_color_1 = "#10b981"  # Green
                            risk_level_text_1 = "Low"
                        elif risk_count_1 <= 2:
                            risk_color_1 = "#f59e0b"  # Orange
                            risk_level_text_1 = "Moderate"
                        else:
                            risk_color_1 = "#ef4444"  # Red
                            risk_level_text_1 = "High"
                        
                        # Display risk summary card for Patient 1
                        st.markdown(f"""
                        <div style="background-color: {risk_color_1}20; padding: 15px; border-radius: 10px; text-align: center;">
                            <h4>Patient {patient_id_1}</h4>
                            <p><strong>Risk Factors:</strong> {risk_count_1} of {len(risk_factors)}</p>
                            <p><strong>Risk Level:</strong> <span style="color: {risk_color_1}; font-weight: bold;">{risk_level_text_1}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Summary for Patient 2
                    with col2_summary:
                        if risk_count_2 == 0:
                            risk_color_2 = "#10b981"
                            risk_level_text_2 = "Low"
                        elif risk_count_2 <= 2:
                            risk_color_2 = "#f59e0b"
                            risk_level_text_2 = "Moderate"
                        else:
                            risk_color_2 = "#ef4444"
                            risk_level_text_2 = "High"
                        
                        # Display risk summary card for Patient 2
                        st.markdown(f"""
                        <div style="background-color: {risk_color_2}20; padding: 15px; border-radius: 10px; text-align: center;">
                            <h4>Patient {patient_id_2}</h4>
                            <p><strong>Risk Factors:</strong> {risk_count_2} of {len(risk_factors)}</p>
                            <p><strong>Risk Level:</strong> <span style="color: {risk_color_2}; font-weight: bold;">{risk_level_text_2}</span></p>
                        </div>
                        """, unsafe_allow_html=True)

                # ---------------- Tab 3: Lifestyle & Symptoms ----------------
        with comparison_tabs[2]:
            # Split layout into two columns: Lifestyle on left, Cognitive Symptoms on right
            col1_life, col2_life = st.columns(2)
            
            # ---------------- Left column: Lifestyle Factors ----------------
            with col1_life:
                st.markdown("### 🏃 Lifestyle Factors Comparison")
                
                # Define lifestyle factors with icons and value types
                lifestyle = {
                    'RAW_Smoking': ('Smoking Status', '🚬', 'binary'),
                    'RAW_AlcoholConsumption': ('Alcohol Consumption', '🍷', 'units_per_week'),
                    'RAW_PhysicalActivity': ('Physical Activity', '🏃', 'hours_per_week'),
                    'RAW_DietQuality': ('Diet Quality', '🥗', 'diet_score'),
                    'RAW_SleepQuality': ('Sleep Quality', '😴', 'sleep_score')
                }
                
                # Iterate through each lifestyle factor
                for col, (display_name, icon, val_type) in lifestyle.items():
                    if col in raw_cols:
                        val1 = patient1_data[col]
                        val2 = patient2_data[col]
                        
                        st.markdown(f"#### {icon} {display_name}")
                        
                        # Function to format values based on type and add status indicators
                        def format_lifestyle_value(val, val_type):
                            if pd.notna(val):
                                if val_type == 'binary':  # Yes/No
                                    return "Yes" if val == 1 else "No"
                                elif val_type == 'units_per_week':  # Alcohol consumption
                                    if isinstance(val, (int, float)):
                                        if val <= 14:
                                            status = "🟢 Low risk"
                                        elif val <= 17:
                                            status = "🟡 Moderate risk"
                                        else:
                                            status = "🔴 High risk"
                                        return f"{val:.1f} units/week ({status})"
                                elif val_type == 'hours_per_week':  # Physical activity
                                    if isinstance(val, (int, float)):
                                        if val >= 2.5:
                                            status = "🟢 Good"
                                        elif val >= 1:
                                            status = "🟡 Fair"
                                        else:
                                            status = "🔴 Low"
                                        return f"{val:.1f} hours/week ({status})"
                                elif val_type in ['diet_score', 'sleep_score']:  # Diet/Sleep quality
                                    if isinstance(val, (int, float)):
                                        if val >= 7:
                                            status = "🟢 Excellent"
                                        elif val >= 5:
                                            status = "🟡 Good"
                                        else:
                                            status = "🔴 Poor"
                                        return f"{val:.1f}/10 ({status})"
                            return "Not available"
                        
                        # Format values for both patients
                        val1_display = format_lifestyle_value(val1, val_type)
                        val2_display = format_lifestyle_value(val2, val_type)
                        
                        # Display formatted values in a styled container
                        st.markdown(f"""
                        <div style="background-color: #f8fafc; padding: 10px; margin: 10px 0; border-radius: 5px;">
                            <span style="color: #666;">Patient {patient_id_1}:</span> {val1_display}<br>
                            <span style="color: #666;">Patient {patient_id_2}:</span> {val2_display}
                        </div>
                        """, unsafe_allow_html=True)
            
            # ---------------- Right column: Cognitive Symptoms ----------------
            with col2_life:
                st.markdown("### 🧠 Cognitive Symptoms Comparison")
                
                # Define cognitive symptoms with icons
                symptoms = {
                    'RAW_MemoryComplaints': ('Memory Complaints', '🧩'),
                    'RAW_BehavioralProblems': ('Behavioral Problems', '😤'),
                    'RAW_Confusion': ('Confusion', '😵'),
                    'RAW_Disorientation': ('Disorientation', '🧭'),
                    'RAW_PersonalityChanges': ('Personality Changes', '🎭'),
                    'RAW_DifficultyCompletingTasks': ('Difficulty Completing Tasks', '📝'),
                    'RAW_Forgetfulness': ('Forgetfulness', '💭')
                }
                
                # Initialize symptom counters
                symptom_count_1 = 0
                symptom_count_2 = 0
                
                # Iterate through symptoms and display for both patients
                for col, (display_name, icon) in symptoms.items():
                    if col in raw_cols:
                        val1 = patient1_data[col]
                        val2 = patient2_data[col]
                        
                        # Count present symptoms
                        if pd.notna(val1) and val1 == 1:
                            symptom_count_1 += 1
                        if pd.notna(val2) and val2 == 1:
                            symptom_count_2 += 1
                        
                        # Format symptom display
                        def format_symptom(val):
                            if pd.notna(val):
                                return "⚠️ Present" if val == 1 else "✅ Absent"
                            return "N/A"
                        
                        val1_display = format_symptom(val1)
                        val2_display = format_symptom(val2)
                        
                        # Display formatted symptom values
                        st.markdown(f"""
                        <div style="background-color: #f8fafc; padding: 8px; margin: 5px 0; border-radius: 5px;">
                            <strong>{icon} {display_name}:</strong><br>
                            <span style="color: #666; font-size: 0.9em;">P{patient_id_1}:</span> {val1_display}<br>
                            <span style="color: #666; font-size: 0.9em;">P{patient_id_2}:</span> {val2_display}
                        </div>
                        """, unsafe_allow_html=True)
                
                # ---------------- Symptom summary ----------------
                st.markdown("---")  # Horizontal separator
                st.markdown(f"""
                <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px;">
                    <h4>Symptom Summary</h4>
                    <p><strong>Patient {patient_id_1}:</strong> {symptom_count_1} of {len(symptoms)} symptoms</p>
                    <p><strong>Patient {patient_id_2}:</strong> {symptom_count_2} of {len(symptoms)} symptoms</p>
                </div>
                """, unsafe_allow_html=True)

        # Tab 4: Lab Results comparison for the selected patients
        with comparison_tabs[3]:
            st.markdown("### 🔬 Clinical Measurements Comparison")  # Main heading for the tab
            
            # Create two columns: one for Blood Pressure, one for Cholesterol Panel
            col1_lab, col2_lab = st.columns(2)
            
            # ------------------------------
            # Column 1: Blood Pressure
            # ------------------------------
            with col1_lab:
                st.markdown("#### Blood Pressure")  # Subsection heading
                
                # Loop through both selected patients to display their BP
                for patient_id, patient_data in [(patient_id_1, patient1_data), (patient_id_2, patient2_data)]:
                    # Ensure systolic and diastolic BP columns exist
                    if 'RAW_SystolicBP' in raw_cols and 'RAW_DiastolicBP' in raw_cols:
                        systolic = patient_data.get('RAW_SystolicBP', 'N/A')
                        diastolic = patient_data.get('RAW_DiastolicBP', 'N/A')
                        
                        if pd.notna(systolic) and pd.notna(diastolic):
                            # Classify blood pressure based on standard clinical guidelines
                            if systolic < 120 and diastolic < 80:
                                bp_status = "Normal 🟢"
                            elif systolic < 130 and diastolic < 80:
                                bp_status = "Elevated 🟡"
                            elif systolic < 140 or diastolic < 90:
                                bp_status = "Stage 1 HTN 🟠"
                            else:
                                bp_status = "Stage 2 HTN 🔴"
                            
                            # Display BP with status in a styled info card
                            st.markdown(f"""
                            <div style="background-color: #f8fafc; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                <strong>Patient {patient_id}:</strong><br>
                                {systolic:.0f}/{diastolic:.0f} mmHg ({bp_status})
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Show message if BP data is missing
                            st.markdown(f"**Patient {patient_id}:** Blood Pressure not available")
            
            # ------------------------------
            # Column 2: Cholesterol Panel
            # ------------------------------
            with col2_lab:
                st.markdown("#### Cholesterol Panel")  # Subsection heading
                
                # Dictionary mapping cholesterol columns to (label, unit, threshold)
                # Thresholds used for color-coding normal vs high/low
                cholesterol_values = {
                    'RAW_CholesterolTotal': ('Total', 'mg/dL', 200),
                    'RAW_CholesterolLDL': ('LDL', 'mg/dL', 100),
                    'RAW_CholesterolHDL': ('HDL', 'mg/dL', 40),
                    'RAW_CholesterolTriglycerides': ('Triglycerides', 'mg/dL', 150)
                }
                
                # Loop through each cholesterol measure
                for col, (label, unit, threshold) in cholesterol_values.items():
                    if col in raw_cols:  # Ensure column exists
                        st.markdown(f"**{label}:**")  # Display measure name
                        
                        # Loop through both patients to show their values
                        for patient_id, patient_data in [(patient_id_1, patient1_data), (patient_id_2, patient2_data)]:
                            val = patient_data[col]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                # HDL is "better if higher", others are "better if lower"
                                if col == 'RAW_CholesterolHDL':
                                    color = "#10b981" if val >= threshold else "#ef4444"
                                    status = "Normal" if val >= threshold else "Low"
                                else:
                                    color = "#10b981" if val <= threshold else "#ef4444"
                                    status = "Normal" if val <= threshold else "High"
                                
                                # Display patient-specific value with color coding
                                st.markdown(f"""
                                <div style="background-color: #f8fafc; padding: 5px; margin: 2px 0; border-radius: 3px;">
                                    <span style="color: #666;">P{patient_id}:</span> 
                                    <span style="color: {color}; font-weight: bold;">{val:.0f} {unit}</span> ({status})
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Display message if cholesterol value is missing
                                st.markdown(f"<span style='color: #666;'>P{patient_id}:</span> Not available", unsafe_allow_html=True)

        # ---------------- Tab 5: SHAP Feature Impact Comparison ----------------
        with comparison_tabs[4]:
            st.markdown("### 📊 SHAP Feature Impact Comparison")
            
            # ---------------- Identify SHAP columns ----------------
            # Exclude columns that are IDs, predictions, or raw features
            excluded_cols = ['Patient_ID', 'Predicted_Diagnosis', 'Prediction_Probability',
                            'Prediction_Confidence', 'Actual_Diagnosis', 'Correct_Prediction',
                            'Prediction_Error']
            
            # Include any columns with 'id' in the name
            id_columns = [col for col in df_filtered.columns if 'id' in col.lower()]
            excluded_cols.extend(id_columns)
            
            # Separate raw and SHAP columns
            raw_columns = [col for col in df_filtered.columns if col.startswith('RAW_')]
            shap_columns = [col for col in df_filtered.columns 
                            if col not in excluded_cols and col not in raw_columns]
            
            # Define binary features for formatting
            binary_features = [
                'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
                'Depression', 'HeadInjury', 'Hypertension', 'Smoking', 
                'AlcoholConsumption', 'MemoryComplaints', 'BehavioralProblems',
                'Confusion', 'Disorientation', 'PersonalityChanges', 
                'DifficultyCompletingTasks', 'Forgetfulness'
            ]
            
            # ---------------- Function to format feature values ----------------
            def format_feature_value(feature_name, value, raw_value=None):
                """Format feature values for display (handles binary, numeric, and gender)"""
                if pd.isna(value) or value == 'N/A':
                    return 'N/A'
                
                display_value = raw_value if raw_value is not None and pd.notna(raw_value) else value
                
                # Binary features
                if any(bf in feature_name for bf in binary_features):
                    if isinstance(display_value, (int, float)):
                        return "Present" if int(display_value) == 1 else "Absent"
                    elif isinstance(display_value, str):
                        try:
                            numeric_val = float(display_value)
                            return "Present" if int(numeric_val) == 1 else "Absent"
                        except:
                            return str(display_value)
                
                # Gender
                if 'Gender' in feature_name:
                    if isinstance(display_value, (int, float)):
                        return "Female" if int(display_value) == 0 else "Male"
                    elif isinstance(display_value, str):
                        try:
                            numeric_val = float(display_value)
                            return "Female" if int(numeric_val) == 0 else "Male"
                        except:
                            return str(display_value)
                
                # Numeric features with units
                if isinstance(display_value, (int, float)):
                    if 'Age' in feature_name:
                        return f"{int(display_value)} years"
                    elif 'BMI' in feature_name:
                        return f"{display_value:.1f} kg/m²"
                    elif 'BP' in feature_name:
                        return f"{display_value:.0f} mmHg"
                    elif 'Cholesterol' in feature_name:
                        return f"{display_value:.0f} mg/dL"
                    elif 'MMSE' in feature_name:
                        return f"{display_value:.0f}/30"
                    elif 'ADL' in feature_name or 'FunctionalAssessment' in feature_name:
                        return f"{display_value:.0f}/10"
                    elif any(scale in feature_name for scale in ['PhysicalActivity', 'DietQuality', 'SleepQuality']):
                        return f"{display_value:.0f}/10"
                    else:
                        return f"{display_value:.2f}" if display_value != int(display_value) else f"{int(display_value)}"
                
                # Handle string numeric values
                elif isinstance(display_value, str):
                    try:
                        numeric_val = float(display_value)
                        return format_feature_value(feature_name, numeric_val, None)
                    except:
                        return str(display_value)
                
                return str(display_value)
            
            # ---------------- Extract SHAP values for both patients ----------------
            if shap_columns:
                patient1_shap = pd.Series(dtype=float)
                patient2_shap = pd.Series(dtype=float)
                
                for col in shap_columns:
                    try:
                        value1 = pd.to_numeric(patient1_data[col], errors='coerce')
                        value2 = pd.to_numeric(patient2_data[col], errors='coerce')
                        if pd.notna(value1):
                            patient1_shap[col] = value1
                        if pd.notna(value2):
                            patient2_shap[col] = value2
                    except:
                        continue
                
                if len(patient1_shap) > 0 and len(patient2_shap) > 0:
                    # ---------------- Risk Factors Comparison ----------------
                    st.markdown("#### 🔴 Risk Factors Comparison")
                    col1, col2 = st.columns(2)
                    
                    # ---------------- Patient 1 Risk Factors ----------------
                    with col1:
                        st.markdown(f"**Patient {patient_id_1} - Top Risk Factors**")
                        # Sort by absolute SHAP value
                        p1_top_features_by_abs = patient1_shap.reindex(patient1_shap.abs().sort_values(ascending=False).index).head(15)
                        top_risk_p1 = p1_top_features_by_abs[p1_top_features_by_abs > 0]
                        
                        for feature, shap_value in top_risk_p1.items():
                            # Get actual value
                            raw_feature_name = f'RAW_{feature}'
                            actual_value = patient1_data.get(raw_feature_name, patient1_data.get(feature, 'N/A'))
                            
                            # Format value
                            formatted_value = format_feature_value(feature, actual_value)
                            
                            # Add safety indicators for higher-is-better features
                            safety_indicator = ""
                            higher_is_better_features = ['ADL', 'MMSE', 'FunctionalAssessment', 'PhysicalActivity', 'DietQuality', 'SleepQuality']
                            if any(hb in feature for hb in higher_is_better_features) and pd.notna(actual_value):
                                try:
                                    val = float(actual_value)
                                    # MMSE grading
                                    if 'MMSE' in feature:
                                        if val >= 27:
                                            safety_indicator = " ✅ (No cognitive impairment)"
                                        elif val >= 24:
                                            safety_indicator = " 🟡 (Mild cognitive impairment)"
                                        elif val >= 18:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    # ADL / FunctionalAssessment grading
                                    elif 'ADL' in feature or 'FunctionalAssessment' in feature:
                                        if val >= 9:
                                            safety_indicator = " ✅ (Minimal/No impairment)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Mild impairment)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                except:
                                    pass
                            
                            # Display formatted SHAP info
                            st.markdown(f"""
                            <div style="background-color: rgba(239, 68, 68, 0.1); padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #ef4444;">
                                <strong style="color: #991b1b; font-size: 0.9em;">{feature.replace('_', ' ')}</strong><br>
                                <span style="color: #666; font-size: 0.85em;">Value: {formatted_value}{safety_indicator}</span><br>
                                <span style="color: #ef4444; font-weight: bold; font-size: 0.85em;">Impact: +{shap_value:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if len(top_risk_p1) == 0:
                            st.info("No significant risk factors")
                    
                    # ---------------- Patient 2 Risk Factors ----------------
                    with col2:
                        st.markdown(f"**Patient {patient_id_2} - Top Risk Factors**")
                        p2_top_features_by_abs = patient2_shap.reindex(patient2_shap.abs().sort_values(ascending=False).index).head(15)
                        top_risk_p2 = p2_top_features_by_abs[p2_top_features_by_abs > 0]
                        
                        # Same formatting logic as Patient 1
                        for feature, shap_value in top_risk_p2.items():
                            raw_feature_name = f'RAW_{feature}'
                            actual_value = patient2_data.get(raw_feature_name, patient2_data.get(feature, 'N/A'))
                            formatted_value = format_feature_value(feature, actual_value)
                            safety_indicator = ""
                            if any(hb in feature for hb in higher_is_better_features) and pd.notna(actual_value):
                                try:
                                    val = float(actual_value)
                                    if 'MMSE' in feature:
                                        if val >= 27:
                                            safety_indicator = " ✅ (No cognitive impairment)"
                                        elif val >= 24:
                                            safety_indicator = " 🟡 (Mild cognitive impairment)"
                                        elif val >= 18:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                    elif 'ADL' in feature or 'FunctionalAssessment' in feature:
                                        if val >= 9:
                                            safety_indicator = " ✅ (Minimal/No impairment)"
                                        elif val >= 6:
                                            safety_indicator = " 🟡 (Mild impairment)"
                                        elif val >= 4:
                                            safety_indicator = " ⚠️ (Moderate impairment)"
                                        else:
                                            safety_indicator = " ❌ (Severe impairment)"
                                except:
                                    pass
                            st.markdown(f"""
                            <div style="background-color: rgba(239, 68, 68, 0.1); padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #ef4444;">
                                <strong style="color: #991b1b; font-size: 0.9em;">{feature.replace('_', ' ')}</strong><br>
                                <span style="color: #666; font-size: 0.85em;">Value: {formatted_value}{safety_indicator}</span><br>
                                <span style="color: #ef4444; font-weight: bold; font-size: 0.85em;">Impact: +{shap_value:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if len(top_risk_p2) == 0:
                            st.info("No significant risk factors")
                    
                    # ---------------- Protective Factors Comparison ----------------
                    st.markdown("---")
                    st.markdown("#### 🟢 Protective Factors Comparison")
                    col3, col4 = st.columns(2)
                    
                    # Patient 1 Protective
                    with col3:
                        st.markdown(f"**Patient {patient_id_1} - Top Protective Factors**")
                        top_protective_p1 = p1_top_features_by_abs[p1_top_features_by_abs < 0]
                        for feature, shap_value in top_protective_p1.items():
                            raw_feature_name = f'RAW_{feature}'
                            actual_value = patient1_data.get(raw_feature_name, patient1_data.get(feature, 'N/A'))
                            formatted_value = format_feature_value(feature, actual_value)
                            st.markdown(f"""
                            <div style="background-color: rgba(16, 185, 129, 0.1); padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #10b981;">
                                <strong style="color: #064e3b; font-size: 0.9em;">{feature.replace('_', ' ')}</strong><br>
                                <span style="color: #666; font-size: 0.85em;">Value: {formatted_value}</span><br>
                                <span style="color: #10b981; font-weight: bold; font-size: 0.85em;">Impact: {shap_value:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(top_protective_p1) == 0:
                            st.info("No significant protective factors")
                    
                    # Patient 2 Protective
                    with col4:
                        st.markdown(f"**Patient {patient_id_2} - Top Protective Factors**")
                        top_protective_p2 = p2_top_features_by_abs[p2_top_features_by_abs < 0]
                        for feature, shap_value in top_protective_p2.items():
                            raw_feature_name = f'RAW_{feature}'
                            actual_value = patient2_data.get(raw_feature_name, patient2_data.get(feature, 'N/A'))
                            formatted_value = format_feature_value(feature, actual_value)
                            st.markdown(f"""
                            <div style="background-color: rgba(16, 185, 129, 0.1); padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #10b981;">
                                <strong style="color: #064e3b; font-size: 0.9em;">{feature.replace('_', ' ')}</strong><br>
                                <span style="color: #666; font-size: 0.85em;">Value: {formatted_value}</span><br>
                                <span style="color: #10b981; font-weight: bold; font-size: 0.85em;">Impact: {shap_value:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(top_protective_p2) == 0:
                            st.info("No significant protective factors")

                    # ---------------- Key Feature Differences ----------------
                    st.markdown("---")
                    st.markdown("#### 🔄 Key Feature Differences")

                    # Identify common SHAP features between both patients
                    common_features = set(patient1_shap.index) & set(patient2_shap.index)

                    if len(common_features) > 0:
                        # Find features where SHAP effects are opposite
                        opposite_effects = []
                        for feature in common_features:
                            shap1 = patient1_shap[feature]
                            shap2 = patient2_shap[feature]
                            
                            # Opposite effect if product < 0 and at least one impact is meaningful
                            if (shap1 * shap2 < 0) and (abs(shap1) > 0.01 or abs(shap2) > 0.01):
                                opposite_effects.append({
                                    'feature': feature,
                                    'shap1': shap1,
                                    'shap2': shap2,
                                    'diff': abs(shap1 - shap2)
                                })
                        
                        if opposite_effects:
                            st.success(f"Found {len(opposite_effects)} features with opposite effects:")
                            
                            # Display each opposite feature
                            for item in opposite_effects:
                                feature = item['feature']
                                shap1 = item['shap1']
                                shap2 = item['shap2']
                                
                                effect1 = "🔴 Risk Factor" if shap1 > 0 else "🟢 Protective Factor"
                                effect2 = "🔴 Risk Factor" if shap2 > 0 else "🟢 Protective Factor"
                                
                                st.markdown(f"""
                                <div style="background-color: #fff7ed; padding: 12px; margin: 8px 0; border-radius: 8px; border-left: 4px solid #f97316;">
                                    <strong>{feature.replace('_', ' ')}</strong>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 8px;">
                                        <div>
                                            <span style="color: #666; font-size: 0.9em;">Patient {patient_id_1}:</span><br>
                                            <span style="font-size: 0.85em;">{effect1} ({shap1:+.4f})</span>
                                        </div>
                                        <div style="border-left: 1px solid #d1d5db; padding-left: 15px;">
                                            <span style="color: #666; font-size: 0.9em;">Patient {patient_id_2}:</span><br>
                                            <span style="font-size: 0.85em;">{effect2} ({shap2:+.4f})</span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No features found with opposite effects between patients.")
                    else:
                        st.warning("No common features found between patients for SHAP comparison.")

                    # ---------------- Interpretation Guide ----------------
                    with st.expander("📖 How to Interpret SHAP Comparison"):
                        st.markdown("""
                        **Understanding SHAP Comparisons:**
                        
                        - **Risk Factors (Red)**: Features that increase the probability of Alzheimer's for a specific patient. Positive SHAP values push predictions toward higher risk.
                        - **Protective Factors (Green)**: Features that decrease the probability. Negative SHAP values push predictions toward lower risk.
                        - **Opposite Effects**: The same feature may be a risk factor for one patient but protective for another, depending on clinical context.
                        - **Feature Values**: Actual measurements or states (e.g., "Present", "Absent", scores) that drive SHAP impact.

                        **Key Insights:**
                        - Compare the balance of risk vs. protective factors between patients.
                        - Look for features with significant differences in SHAP contributions to explain prediction differences.
                        - Identify modifiable risk factors for potential interventions (e.g., lifestyle changes, cognitive activities).
                        - Even identical feature values can have different SHAP impacts based on overall patient context.
                        """)

                    # ---------------- Overall Comparison Summary ----------------
                    st.markdown("---")
                    st.markdown("## 📋 Overall Comparison Summary")

                    # Determine risk difference
                    prob_diff = abs(prob1 - prob2)
                    if prob_diff > 0.3:
                        risk_comparison = f"Significant difference in risk levels ({prob_diff:.1%} difference)"
                        risk_color = "#ef4444"
                    elif prob_diff > 0.1:
                        risk_comparison = f"Moderate difference in risk levels ({prob_diff:.1%} difference)"
                        risk_color = "#f59e0b"
                    else:
                        risk_comparison = f"Similar risk levels ({prob_diff:.1%} difference)"
                        risk_color = "#10b981"

                    # Display overall risk summary with color coding
                    st.markdown(f"""
                    <div style="background-color: {risk_color}20; padding: 20px; border-radius: 10px; border-left: 4px solid {risk_color};">
                        <h4>Risk Level Comparison</h4>
                        <p style="color: {risk_color}; font-weight: bold;">{risk_comparison}</p>
                        <p><strong>Patient {patient_id_1}:</strong> {prob1:.1%} probability ({risk_level1} risk)</p>
                        <p><strong>Patient {patient_id_2}:</strong> {prob2:.1%} probability ({risk_level2} risk)</p>
                    </div>
                    """, unsafe_allow_html=True)

# Clinical Scenario Tab
with tab5:
    # Make a copy of predictions for safe filtering and manipulation
    df_filtered = df_predictions.copy()
    
    # Load model components once at app start
    model, preprocessor, feature_names_original, explainer, feature_names_processed = load_model_components()

    # Define binary features for proper display ("Present"/"Absent")
    binary_features = [
        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
        'HeadInjury', 'Hypertension', 'Smoking', 'AlcoholConsumption',
        'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
        'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
    ]

    # Map internal feature names to readable labels for display
    feature_display_names = {
        'Age': 'Age',
        'BMI': 'Body Mass Index',
        'MMSE': 'Cognitive Score (MMSE)',
        'FunctionalAssessment': 'Functional Assessment',
        'ADL': 'Activities of Daily Living',
        'PhysicalActivity': 'Physical Activity Level',
        'DietQuality': 'Diet Quality',
        'SleepQuality': 'Sleep Quality',
        'SystolicBP': 'Systolic Blood Pressure',
        'DiastolicBP': 'Diastolic Blood Pressure',
        'CholesterolTotal': 'Total Cholesterol',
        'CardiovascularDisease': 'Cardiovascular Disease',
        'Diabetes': 'Diabetes',
        'Depression': 'Depression',
        'Hypertension': 'Hypertension',
        'FamilyHistoryAlzheimers': 'Family History of Alzheimer\'s',
        'HeadInjury': 'Head Injury History',
        'Smoking': 'Smoking',
        'AlcoholConsumption': 'Alcohol Consumption',
        'MemoryComplaints': 'Memory Complaints',
        'Forgetfulness': 'Forgetfulness',
        'Confusion': 'Confusion',
        'Disorientation': 'Disorientation',
        'PersonalityChanges': 'Personality Changes',
        'DifficultyCompletingTasks': 'Difficulty Completing Tasks',
        'BehavioralProblems': 'Behavioral Problems'
    }

    # Function to create a grouped bar chart showing SHAP value changes after feature modification
    def create_feature_impact_chart(current_shap_values, new_shap_values, feature_names, modifications):
        """Create an interactive chart showing feature impact changes"""
        fig = go.Figure()
        
        # Identify features that were modified
        modified_indices = []
        modified_features = []
        for i, feature in enumerate(feature_names):
            if any(mod in feature for mod in modifications.keys()):
                modified_indices.append(i)
                modified_features.append(feature)
        
        if modified_indices:
            # Get current and new SHAP values for modified features
            current_impacts = [current_shap_values[0][i] for i in modified_indices]
            new_impacts = [new_shap_values[0][i] for i in modified_indices]
            
            # Add original impact bars
            fig.add_trace(go.Bar(
                name='Original Impact',
                x=modified_features,
                y=current_impacts,
                marker_color='lightblue',
                text=[f'{val:.3f}' for val in current_impacts],
                textposition='auto',
            ))
            
            # Add new impact bars after modification
            fig.add_trace(go.Bar(
                name='New Impact',
                x=modified_features,
                y=new_impacts,
                marker_color='darkblue',
                text=[f'{val:.3f}' for val in new_impacts],
                textposition='auto',
            ))
            
            # Layout settings
            fig.update_layout(
                title='Feature Impact Changes',
                xaxis_title='Features',
                yaxis_title='SHAP Value (Impact on Prediction)',
                barmode='group',           # Display bars side by side
                height=400,
                showlegend=True,
                hovermode='x unified'      # Show all bars on hover
            )
        
        return fig
    
    # Function to create a gauge chart for predicted risk
    def create_risk_gauge(probability, title="Risk Level"):
        """Create a gauge chart for risk visualization"""
        percentage = round(probability * 100, 1)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",       # Show gauge and numeric value
            value = percentage,
            title={
                'text': f"{title} <span style='font-size:20px; color:black'>Probability</span>",
                'font': {'size': 20, 'color': 'black'}
            },
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'size': 40, 'color': 'black'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},           # Color of the main bar
                'bgcolor': "white",                      # Background color
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [                               # Risk level coloring
                    {'range': [0, 50], 'color': '#10b981'},     # Low
                    {'range': [50, 70], 'color': '#dcfce7'},    # Slightly elevated
                    {'range': [70, 90], 'color': '#f59e0b'},    # Moderate
                    {'range': [90, 100], 'color': '#ef4444'}    # High
                ],
                'threshold': {                           # Mark the exact probability
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': percentage
                }
            }
        ))
        
        # Layout adjustments
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    # Function to create a visual comparison for a single feature
    def create_feature_comparison_visual(feature, original_value, new_value, feature_type='continuous'):
        """Return display information and CSS classes for a feature comparison card"""
        
        # Determine if the change represents an improvement
        is_improvement = False
        if feature_type == 'binary':
            # For risk factors, going from 1 to 0 is an improvement
            if feature in ['CardiovascularDisease', 'Diabetes', 'Depression', 'Hypertension', 
                        'HeadInjury', 'Smoking', 'AlcoholConsumption']:
                is_improvement = new_value < original_value
            # For symptoms, going from 1 to 0 is an improvement
            elif feature in ['MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
                            'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']:
                is_improvement = new_value < original_value
            # Family history is static
            elif feature == 'FamilyHistoryAlzheimers':
                is_improvement = None
        else:
            # For continuous features
            if feature in ['MMSE', 'FunctionalAssessment', 'ADL', 'PhysicalActivity', 'DietQuality', 'SleepQuality']:
                is_improvement = new_value > original_value  # Higher is better
            elif feature in ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal']:
                is_improvement = new_value < original_value  # Lower is better
        
        # Determine the CSS class for the card based on improvement
        if is_improvement is None:
            card_class = "feature-card"
        elif is_improvement:
            card_class = "feature-card feature-improved"
        else:
            card_class = "feature-card feature-worsened"
        
        # Format values for display
        if feature_type == 'binary':
            original_display = '✓ Yes' if original_value == 1 else '✗ No'
            new_display = '✓ Yes' if new_value == 1 else '✗ No'
            change_text = "Changed" if original_value != new_value else "No change"
        else:
            # Format numeric continuous values
            if feature in ['BMI', 'PhysicalActivity', 'DietQuality', 'SleepQuality']:
                original_display = f"{original_value:.1f}"
                new_display = f"{new_value:.1f}"
            else:
                original_display = f"{int(original_value)}"
                new_display = f"{int(new_value)}"
            
            # Show the change numerically
            change = new_value - original_value
            if change > 0:
                change_text = f"+{change:.1f}" if feature in ['BMI'] else f"+{int(change)}"
            elif change < 0:
                change_text = f"{change:.1f}" if feature in ['BMI'] else f"{int(change)}"
            else:
                change_text = "No change"
        
        # Determine value CSS class based on improvement
        if is_improvement is None:
            value_class = "value-neutral"
        elif is_improvement:
            value_class = "value-improved"
        else:
            value_class = "value-worsened"
        
        return card_class, original_display, new_display, change_text, value_class


    # Enhanced What-If Analysis Function
    def run(df_filtered):
        """Run interactive What-If analysis for a selected patient"""
        
        # Title and introduction
        st.markdown("<p style='font-size: 1.4rem; font-weight: bold; margin-bottom: -0.5rem;'>Explore how changing feature values impacts the prediction for a patient with advanced visualizations and recommendations</p>", unsafe_allow_html=True)
        
        # Initialize session state for storing modifications and reset counter
        if 'reset_counter' not in st.session_state:
            st.session_state.reset_counter = 0
        if 'all_modifications' not in st.session_state:
            st.session_state.all_modifications = {}
        
        # Patient selection dropdown
        patient_id_whatif = st.selectbox(
            "Select Patient for What-If Analysis:",
            options=df_filtered['Patient_ID'].tolist(),
            key="whatif_patient_select"
        )
        
        if patient_id_whatif:
            # Use reset counter to ensure widgets are refreshed properly
            reset_key = st.session_state.reset_counter
            
            # Get the selected patient's data
            patient_data_whatif = df_filtered[df_filtered['Patient_ID'] == patient_id_whatif].iloc[0].copy()
            
            # Section header for risk visualization
            st.markdown("### 📊 Risk Assessment Dashboard")
            
            # Four-column layout: current gauge, current prediction, new gauge, new prediction
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                # Show current risk probability gauge
                current_prob = patient_data_whatif['Prediction_Probability']
                current_diagnosis = patient_data_whatif['Predicted_Diagnosis']
                fig_current = create_risk_gauge(current_prob, "Current Risk")
                st.plotly_chart(fig_current, use_container_width=True)
            
            with col2:
                # Display current prediction result and risk level
                current_prediction = "Positive" if current_diagnosis == 1 else "Negative"
                current_pred_color = "#ef4444" if current_diagnosis == 1 else "#10b981"

                # Determine descriptive risk level
                current_risk = ('No Risk' if current_diagnosis == 0 else 
                                'Low Risk' if current_diagnosis == 1 and current_prob < 0.7 else 
                                'Medium Risk' if current_diagnosis == 1 and current_prob < 0.9 else 
                                'High Risk' )
                
                # Color based on risk level
                current_risk_color = ('#10b981' if current_diagnosis == 0 else 
                                    '#f59e0b' if current_diagnosis == 1 and current_prob < 0.7 else 
                                    '#f97316' if current_diagnosis == 1 and current_prob < 0.9 else 
                                    '#ef4444' )
            
                # Display card with current prediction and risk
                st.markdown(f"""
                <div class="info-card" style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border: 2px solid {current_pred_color}40; animation: pulse 0.5s;">
                    <h4 style="margin: 0 0 10px 0;">Current Prediction</h4>
                    <div style="font-size: 1.5rem; color: {current_pred_color}; font-weight: bold;">{current_prediction}</div>
                    <div style="margin-top: 5px;">Risk Level: <span style="color: {current_risk_color}; font-weight: bold;">{current_risk}</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Placeholder for new risk gauge after feature modification
                new_prob_gauge = st.empty()
            
            with col4:
                # Placeholder for new prediction card
                new_pred_placeholder = st.empty()
                new_pred_placeholder.markdown("""
                <div style="text-align: center; padding: 20px; opacity: 0.5;">
                    <h4>New Prediction</h4>
                    <p>Modify features to see changes</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Feature modification section with enhanced organization
            with st.expander("📝 Modify Patient Features", expanded=True):
                # Create tabs to organize features by category for easier navigation
                feature_tabs = st.tabs(["Demographics & Clinical", "Risk Factors", "Symptoms", "Lifestyle", "Lab Results", "Quick Scenarios"])
                
                # Retrieve modifications stored in session state to persist changes across interactions
                all_modifications = st.session_state.all_modifications.copy()
                
                # =========================
                # Tab 1: Demographics & Clinical
                # =========================
                with feature_tabs[0]:
                    col1, col2 = st.columns(2)  # Split into two columns: demographics & clinical scores
                    
                    # ---- Column 1: Demographics ----
                    with col1:
                        st.markdown('<h3 style="font-size: 22px; font-weight: bold;">👤 Demographics</h3>', unsafe_allow_html=True)
                        
                        # --- Age ---
                        current_age = patient_data_whatif.get('RAW_Age', 65)  # Default 65 if missing
                        # Risk interpretation based on age
                        age_help = "🟢 Low risk" if current_age < 65 else "🟡 Moderate risk" if current_age < 75 else "🔴 High risk"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Age ({age_help})</p>', unsafe_allow_html=True)
                        # Slider for adjusting age
                        new_age = st.slider(
                            "Age",
                            min_value=40,
                            max_value=100,
                            value=int(current_age) if pd.notna(current_age) else 65,
                            key=f"age_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        # Track changes and display a small change card
                        if new_age != int(current_age):
                            all_modifications['Age'] = new_age
                            age_change = new_age - current_age
                            change_color = "#ef4444" if age_change > 0 else "#10b981"
                            st.markdown(f"""
                            <div style="padding: 5px 10px; background: {change_color}20; border-radius: 5px; margin-top: -10px;">
                                <small>Changed from {int(current_age)} → {new_age} years</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # --- BMI ---
                        current_bmi = patient_data_whatif.get('RAW_BMI', 25.0)
                        # Health status interpretation
                        bmi_status = "🟢 Normal" if 18.5 <= current_bmi < 25 else "🟡 Overweight" if 25 <= current_bmi < 30 else "🔴 Obese"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">BMI ({bmi_status})</p>', unsafe_allow_html=True)
                        # Slider for BMI adjustment
                        new_bmi = st.slider(
                            f"BMI ({bmi_status})",
                            min_value=15.0,
                            max_value=45.0,
                            value=float(current_bmi) if pd.notna(current_bmi) else 25.0,
                            step=0.1,
                            key=f"bmi_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        # Track BMI changes with visual feedback
                        if abs(new_bmi - float(current_bmi)) > 0.01:
                            all_modifications['BMI'] = new_bmi
                            bmi_change = new_bmi - current_bmi
                            change_color = "#ef4444" if bmi_change > 0 else "#10b981"
                            st.markdown(f"""
                            <div style="padding: 5px 10px; background: {change_color}20; border-radius: 5px; margin-top: -10px;">
                                <small>Changed from {current_bmi:.1f} → {new_bmi:.1f} kg/m²</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ---- Column 2: Clinical Scores ----
                    with col2:
                        st.markdown('<h3 style="font-size: 22px; font-weight: bold;">🧠 Clinical Scores</h3>', unsafe_allow_html=True)
                        
                        # --- MMSE ---
                        current_mmse = patient_data_whatif.get('RAW_MMSE', 27)
                        current_mmse_int = int(current_mmse) if pd.notna(current_mmse) else 27
                        # Interpretation of cognitive score
                        mmse_interpretation = "🟢 Normal" if current_mmse_int >= 24 else "🟡 Mild Impairment" if current_mmse_int >= 18 else "🔴 Moderate Impairment"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">MMSE Score ({mmse_interpretation})</p>', unsafe_allow_html=True)
                        st.markdown('<p style="font-size: 16px; color: #666; margin-bottom: 10px;">Mini-Mental State Exam (0-30): Higher is better</p>', unsafe_allow_html=True)
                        # Slider for MMSE
                        new_mmse = st.slider(
                            f"MMSE Score ({mmse_interpretation})",
                            min_value=0,
                            max_value=30,
                            value=current_mmse_int,
                            help="Mini-Mental State Exam (0-30): Higher is better",
                            key=f"mmse_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_mmse != current_mmse_int:
                            all_modifications['MMSE'] = new_mmse
                            mmse_change = new_mmse - current_mmse_int
                            change_color = "#10b981" if mmse_change > 0 else "#ef4444"
                            st.markdown(f"""
                            <div style="padding: 5px 10px; background: {change_color}20; border-radius: 5px; margin-top: -10px;">
                                <small>Changed from {current_mmse_int} → {new_mmse}/30</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # --- Functional Assessment ---
                        current_func = patient_data_whatif.get('RAW_FunctionalAssessment', 8)
                        current_func_int = int(current_func) if pd.notna(current_func) else 8
                        func_status = "🟢 Good" if current_func_int >= 9 else "🟡 Moderate" if current_func_int >= 4 else "🔴 Poor"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">FunctionalAssessment Score ({func_status})</p>', unsafe_allow_html=True)
                        st.markdown('<p style="font-size: 16px; color: #666; margin-bottom: 10px;">Functional Assessment Score(0-10): Higher is better</p>', unsafe_allow_html=True)
                        new_func = st.slider(
                            f"Functional Assessment ({func_status})",
                            min_value=0,
                            max_value=10,
                            value=current_func_int,
                            help="Functional Assessment (0-10): Higher is better",
                            key=f"func_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_func != current_func_int:
                            all_modifications['FunctionalAssessment'] = new_func
                            func_change = new_func - current_func_int
                            change_color = "#10b981" if func_change > 0 else "#ef4444"
                            st.markdown(f"""
                            <div style="padding: 5px 10px; background: {change_color}20; border-radius: 5px; margin-top: -10px;">
                                <small>Changed from {current_func_int} → {new_func}/10</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # --- ADL (Activities of Daily Living) ---
                        current_adl = patient_data_whatif.get('RAW_ADL', 8)
                        current_adl_int = int(current_adl) if pd.notna(current_adl) else 8
                        adl_status = "🟢 Independent" if current_adl_int >= 9 else "🟡 Needs Some Help" if current_adl_int >= 4 else "🔴 Dependent"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Activities of Daily Life ({adl_status})</p>', unsafe_allow_html=True)
                        st.markdown('<p style="font-size: 16px; color: #666; margin-bottom: 10px;">Activities of Daily Life Score(0-10): Higher is better</p>', unsafe_allow_html=True)
                        new_adl = st.slider(
                            f"Activities of Daily Living ({adl_status})",
                            min_value=0,
                            max_value=10,
                            value=current_adl_int,
                            help="ADL (0-10): Higher is better",
                            key=f"adl_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_adl != current_adl_int:
                            all_modifications['ADL'] = new_adl
                            adl_change = new_adl - current_adl_int
                            change_color = "#10b981" if adl_change > 0 else "#ef4444"
                            st.markdown(f"""
                            <div style="padding: 5px 10px; background: {change_color}20; border-radius: 5px; margin-top: -10px;">
                                <small>Changed from {current_adl_int} → {new_adl}/10</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # =========================
                # Tab 2: Risk Factors
                # =========================
                with feature_tabs[1]:
                    st.markdown("#### 🚨 Medical & Life Style Risk Factors")
                    
                    # Define medical condition features
                    medical_conditions = {
                        'CardiovascularDisease': 'Cardiovascular Disease',
                        'Diabetes': 'Diabetes',
                        'Depression': 'Depression',
                        'Hypertension': 'Hypertension',
                        'FamilyHistoryAlzheimers': 'Family History of Alzheimer\'s',
                        'HeadInjury': 'Head Injury'
                    }
                    
                    # Define lifestyle risk features
                    lifestyle_risks = {
                        'Smoking': 'Smoking',
                        'AlcoholConsumption': 'Alcohol Consumption'
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    # ---- Column 1: Medical Conditions ----
                    with col1:
                        st.markdown("**Medical Conditions**")
                        for feature, display_name in medical_conditions.items():
                            # Get current feature value
                            current_val = patient_data_whatif.get(f'RAW_{feature}', 0)
                            current_bool = bool(int(current_val)) if pd.notna(current_val) else False
                            # Checkbox to modify feature
                            new_val = st.checkbox(
                                display_name,
                                value=current_bool,
                                key=f"risk_checkbox_{feature}_{patient_id_whatif}_{reset_key}"
                            )
                            # Track changes in modifications dictionary
                            if new_val != current_bool:
                                all_modifications[feature] = 1 if new_val else 0
                    
                    # ---- Column 2: Lifestyle Risks ----
                    with col2:
                        st.markdown("**Lifestyle Factors**")
                        for feature, display_name in lifestyle_risks.items():
                            current_val = patient_data_whatif.get(f'RAW_{feature}', 0)
                            current_bool = bool(int(current_val)) if pd.notna(current_val) else False
                            new_val = st.checkbox(
                                display_name,
                                value=current_bool,
                                key=f"risk_checkbox_{feature}_{patient_id_whatif}_{reset_key}"
                            )
                            if new_val != current_bool:
                                all_modifications[feature] = 1 if new_val else 0


                # =========================
                # Tab 3: Symptoms
                # =========================
                with feature_tabs[2]:
                    st.markdown("#### 🧩 Cognitive Symptoms")
                    
                    # Early stage cognitive symptoms
                    early_symptoms = {
                        'MemoryComplaints': 'Memory Complaints',
                        'Forgetfulness': 'Forgetfulness',
                        'Confusion': 'Confusion'
                    }
                    
                    # Advanced stage cognitive symptoms
                    advanced_symptoms = {
                        'Disorientation': 'Disorientation',
                        'PersonalityChanges': 'Personality Changes',
                        'DifficultyCompletingTasks': 'Difficulty Completing Tasks',
                        'BehavioralProblems': 'Behavioral Problems'
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    # ---- Column 1: Early Symptoms ----
                    with col1:
                        st.markdown("**Early Stage Symptoms**")
                        for feature, display_name in early_symptoms.items():
                            current_val = patient_data_whatif.get(f'RAW_{feature}', 0)
                            current_bool = bool(int(current_val)) if pd.notna(current_val) else False
                            new_val = st.checkbox(
                                display_name,
                                value=current_bool,
                                key=f"symptom_checkbox_{feature}_{patient_id_whatif}_{reset_key}"
                            )
                            # Track changes to session state modifications
                            if new_val != current_bool:
                                all_modifications[feature] = 1 if new_val else 0
                    
                    # ---- Column 2: Advanced Symptoms ----
                    with col2:
                        st.markdown("**Advanced Symptoms**")
                        for feature, display_name in advanced_symptoms.items():
                            current_val = patient_data_whatif.get(f'RAW_{feature}', 0)
                            current_bool = bool(int(current_val)) if pd.notna(current_val) else False
                            new_val = st.checkbox(
                                display_name,
                                value=current_bool,
                                key=f"symptom_checkbox_{feature}_{patient_id_whatif}_{reset_key}"
                            )
                            if new_val != current_bool:
                                all_modifications[feature] = 1 if new_val else 0

                # =========================
                # Tab 4: Lifestyle Factors
                # =========================
                with feature_tabs[3]:
                    st.markdown("#### 🏃‍♂️ Modifiable Lifestyle Factors")
                    st.info("These factors can be improved through lifestyle changes")
                    
                    col1, col2 = st.columns(2)
                    
                    # ---- Column 1: Physical Activity & Diet ----
                    with col1:
                        # Physical Activity
                        current_pa = patient_data_whatif.get('RAW_PhysicalActivity', 5)
                        pa_recommendation = "🟢 Excellent" if current_pa >= 8 else "🟡 Good" if current_pa >= 5 else "🔴 Needs Improvement"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Physical Activity Level ({pa_recommendation})</p>', unsafe_allow_html=True)
                        new_pa = st.number_input(
                            f"Physical Activity Level ({pa_recommendation})",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(current_pa) if pd.notna(current_pa) else 5.0,
                            step=0.1,
                            help="0 = Sedentary, 10 = Very Active (150+ min/week)",
                            key=f"pa_input_{patient_id_whatif}_{reset_key}"
                        )
                        if abs(new_pa - current_pa) > 0.001:
                            all_modifications['PhysicalActivity'] = new_pa
                        
                        # Diet Quality
                        current_diet = patient_data_whatif.get('RAW_DietQuality', 5)
                        diet_recommendation = "🟢 Healthy" if current_diet >= 8 else "🟡 Fair" if current_diet >= 5 else "🔴 Poor"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Diet Quality Level ({diet_recommendation})</p>', unsafe_allow_html=True)
                        new_diet = st.number_input(
                            f"Diet Quality ({diet_recommendation})",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(current_diet) if pd.notna(current_diet) else 5.0,
                            step=0.1,
                            help="0 = Poor Diet, 10 = Mediterranean/MIND Diet",
                            key=f"diet_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_diet != current_diet:
                            all_modifications['DietQuality'] = new_diet
                    
                    # ---- Column 2: Sleep Quality ----
                    with col2:
                        current_sleep = patient_data_whatif.get('RAW_SleepQuality', 5)
                        sleep_recommendation = "🟢 Good" if current_sleep >= 7 else "🟡 Fair" if current_sleep >= 5 else "🔴 Poor"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Sleep Quality Level ({sleep_recommendation})</p>', unsafe_allow_html=True)
                        new_sleep = st.number_input(
                            f"Sleep Quality ({sleep_recommendation})",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(current_sleep) if pd.notna(current_sleep) else 5.0,
                            step=0.1,
                            help="0 = Poor Sleep, 10 = 7-9 hours quality sleep",
                            key=f"sleep_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_sleep != current_sleep:
                            all_modifications['SleepQuality'] = new_sleep
                            if new_sleep < 5:
                                st.warning("⚠️ Poor sleep increases dementia risk")

                # =========================
                # Tab 5: Lab Results
                # =========================
                with feature_tabs[4]:
                    st.markdown("#### 🔬 Laboratory Values")
                    
                    col1, col2 = st.columns(2)
                    
                    # ---- Column 1: Blood Pressure ----
                    with col1:
                        st.markdown("**Blood Pressure**")
                        
                        # Systolic
                        current_sbp = patient_data_whatif.get('RAW_SystolicBP', 120)
                        sbp_category = "🟢 Normal" if current_sbp < 120 else "🟡 Elevated" if current_sbp < 130 else "🔴 High"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Systolic Blood Pressure ({sbp_category})</p>', unsafe_allow_html=True)
                        new_sbp = st.slider(
                            f"Systolic BP ({sbp_category})",
                            min_value=90,
                            max_value=200,
                            value=int(current_sbp) if pd.notna(current_sbp) else 120,
                            help="Normal: <120 mmHg, Elevated: 120-129, High: ≥130",
                            key=f"sbp_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_sbp != current_sbp:
                            all_modifications['SystolicBP'] = new_sbp
                        
                        # Diastolic
                        current_dbp = patient_data_whatif.get('RAW_DiastolicBP', 80)
                        dbp_category = "🟢 Normal" if current_dbp < 80 else "🟡 Elevated" if current_dbp < 90 else "🔴 High"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Diastolic Blood Pressure ({dbp_category})</p>', unsafe_allow_html=True)
                        new_dbp = st.slider(
                            f"Diastolic BP ({dbp_category})",
                            min_value=60,
                            max_value=120,
                            value=int(current_dbp) if pd.notna(current_dbp) else 80,
                            help="Normal: <80 mmHg, High: ≥80",
                            key=f"dbp_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_dbp != current_dbp:
                            all_modifications['DiastolicBP'] = new_dbp
                    
                    # ---- Column 2: Cholesterol ----
                    with col2:
                        st.markdown("**Cholesterol**")
                        current_chol = patient_data_whatif.get('RAW_CholesterolTotal', 200)
                        chol_category = "🟢 Desirable" if current_chol < 200 else "🟡 Borderline" if current_chol < 240 else "🔴 High"
                        st.markdown(f'<p style="font-size: 20px; font-weight: 600; margin-bottom: 5px;">Cholesterol Total ({chol_category})</p>', unsafe_allow_html=True)
                        new_chol = st.number_input(
                            f"Total Cholesterol ({chol_category})",
                            min_value=100.0,
                            max_value=400.0,
                            value=float(current_chol) if pd.notna(current_chol) else 200,
                            help="Desirable: <200 mg/dL, Borderline: 200-239, High: ≥240",
                            step=1.0,
                            key=f"chol_slider_{patient_id_whatif}_{reset_key}",
                            label_visibility="collapsed"
                        )
                        if new_chol != current_chol:
                            all_modifications['CholesterolTotal'] = new_chol

                # =========================
                # Tab 6: Quick Scenarios
                # =========================
                with feature_tabs[5]:
                    st.markdown("#### 🎯 Quick Intervention Scenarios")
                    st.info("Click a scenario to apply preset modifications that demonstrate common intervention strategies.")
                    
                    # Preset scenarios
                    scenarios = {
                        "🏃‍♂️ Healthy Lifestyle": {
                            'description': "Improves physical activity, diet, sleep, and eliminates harmful habits",
                            'modifications': {
                                'PhysicalActivity': 8,
                                'DietQuality': 8,
                                'SleepQuality': 8,
                                'Smoking': 0,
                                'AlcoholConsumption': 0
                            }
                        },
                        "💊 Medical Management": {
                            'description': "Optimizes blood pressure and cholesterol to healthy levels",
                            'modifications': {
                                'SystolicBP': 118,
                                'DiastolicBP': 78,
                                'CholesterolTotal': 180
                            }
                        },
                        "🧠 Cognitive Enhancement": {
                            'description': "Improves cognitive and functional scores through interventions",
                            'modifications': {
                                'MMSE': min(30, int(patient_data_whatif.get('RAW_MMSE', 27)) + 2),
                                'FunctionalAssessment': min(10, int(patient_data_whatif.get('RAW_FunctionalAssessment', 8)) + 1),
                                'ADL': min(10, int(patient_data_whatif.get('RAW_ADL', 8)) + 1),
                                'PhysicalActivity': 7,
                                'SleepQuality': 8
                            }
                        },
                        "🎯 Comprehensive Intervention": {
                            'description': "Combines multiple interventions for maximum risk reduction",
                            'modifications': {
                                'PhysicalActivity': 8,
                                'DietQuality': 8,
                                'SleepQuality': 8,
                                'SystolicBP': 118,
                                'DiastolicBP': 78,
                                'CholesterolTotal': 180,
                                'Smoking': 0,
                                'AlcoholConsumption': 0,
                                'BMI': 23.0
                            }
                        }
                    }
                    
                    # Display buttons in 2x2 grid
                    col1, col2 = st.columns(2)
                    for idx, (scenario_name, scenario_data) in enumerate(scenarios.items()):
                        col = col1 if idx % 2 == 0 else col2
                        with col:
                            st.markdown(f'<p style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{scenario_name}</p>', unsafe_allow_html=True)
                            st.caption(scenario_data['description'])
                            
                            if st.button(f"Apply {scenario_name}", use_container_width=True, key=f"scenario_{scenario_name}_{reset_key}"):
                                # Apply preset modifications to session state
                                st.session_state.all_modifications = scenario_data['modifications'].copy()
                                st.success(f"✅ {scenario_name} scenario applied!")
                                st.rerun()  # Refresh app to reflect changes

        # Display current modifications if any
        if st.session_state.all_modifications:
            st.markdown("---")
            st.markdown("### 📋 Currently Applied Modifications")
            
            # Count total modifications and show count
            mod_count = len(st.session_state.all_modifications)
            st.info(f"Total modifications: {mod_count}")
            
            # Button to clear all modifications
            if st.button("🗑️ Clear All Modifications", key=f"clear_all_mods_{reset_key}"):
                st.session_state.all_modifications = {}  # Reset modifications
                st.rerun()  # Refresh the page

        # Update session state with new modifications from this run
        st.session_state.all_modifications.update(all_modifications)

        # Separator for calculation section
        st.markdown("---")

        # Load current modifications from session state
        all_modifications = st.session_state.all_modifications

        # Layout for calculation and visualization
        col1, col2 = st.columns([3, 1])

        with col1:
            # Only allow calculation if modifications exist
            if all_modifications:
                if st.button(
                    "🔄 Calculate New Prediction", 
                    type="primary", 
                    use_container_width=True,
                    key=f"calculate_prediction_btn_{patient_id_whatif}_{reset_key}"
                ):
                    # Validate model components
                    if model is None or preprocessor is None:
                        st.error("Model components not loaded. Please check the model files.")
                    else:
                        with st.spinner("Calculating new prediction and generating insights..."):
                            try:
                                # Copy patient data to avoid overwriting original
                                modified_patient_data = patient_data_whatif.copy()
                                
                                # Apply modifications to both RAW_ and normal feature columns
                                for feature, value in all_modifications.items():
                                    raw_feature = f'RAW_{feature}'
                                    if raw_feature in modified_patient_data.index:
                                        modified_patient_data[raw_feature] = value
                                    if feature in modified_patient_data.index:
                                        modified_patient_data[feature] = value
                                
                                # Create DataFrame for top 10 features (modified)
                                features_for_prediction = pd.DataFrame()
                                for feature in feature_names_original:
                                    if f'RAW_{feature}' in modified_patient_data.index:
                                        features_for_prediction[feature] = [modified_patient_data[f'RAW_{feature}']]
                                    elif feature in modified_patient_data.index:
                                        features_for_prediction[feature] = [modified_patient_data[feature]]
                                    else:
                                        if f'RAW_{feature}' in patient_data_whatif.index:
                                            features_for_prediction[feature] = [patient_data_whatif[f'RAW_{feature}']]
                                        else:
                                            features_for_prediction[feature] = [0]
                                
                                # Get original feature set (unmodified) for SHAP comparison
                                original_features = pd.DataFrame()
                                for feature in feature_names_original:
                                    if f'RAW_{feature}' in patient_data_whatif.index:
                                        original_features[feature] = [patient_data_whatif[f'RAW_{feature}']]
                                    elif feature in patient_data_whatif.index:
                                        original_features[feature] = [patient_data_whatif[feature]]
                                    else:
                                        original_features[feature] = [0]
                                
                                # Preprocess both modified and original features
                                features_preprocessed = preprocessor.transform(features_for_prediction)
                                original_preprocessed = preprocessor.transform(original_features)
                                
                                # Predict probability with modified features
                                new_probability = model.predict_proba(features_preprocessed)[0][1]
                                
                                # Calculate SHAP values for modified and original features
                                new_shap_values = explainer.shap_values(features_preprocessed)
                                original_shap_values = explainer.shap_values(original_preprocessed)
                                
                                # If SHAP returns a list, select the class-1 explanation
                                if isinstance(new_shap_values, list):
                                    new_shap_values = new_shap_values[1]
                                if isinstance(original_shap_values, list):
                                    original_shap_values = original_shap_values[1]
                                
                                # Create updated risk gauge chart
                                fig_new = create_risk_gauge(new_probability, "New Risk")
                                new_prob_gauge.plotly_chart(fig_new, use_container_width=True)
                                
                                # Calculate probability change from original
                                prob_change = new_probability - current_prob
                                
                                # Determine new risk category and prediction label
                                new_risk = 'High' if new_probability >= 0.7 else 'Medium' if new_probability >= 0.3 else 'Low'
                                new_risk_color = '#ef4444' if new_probability >= 0.7 else '#f59e0b' if new_probability >= 0.3 else '#10b981'
                                new_prediction = "Positive" if new_probability >= 0.5 else "Negative"
                                new_pred_color = "#ef4444" if new_probability >= 0.5 else "#10b981"
                                
                                # Display new prediction card
                                new_pred_placeholder.markdown(f"""
                                <div class="info-card" style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border: 2px solid {new_risk_color}40; animation: pulse 0.5s;">
                                    <h4 style="margin: 0 0 10px 0;">New Prediction</h4>
                                    <div style="font-size: 1.5rem; color: {new_pred_color}; font-weight: bold;">{new_prediction}</div>
                                    <div style="margin-top: 5px;">Risk Level: <span style="color: {new_risk_color}; font-weight: bold;">{new_risk}</span></div>
                                </div>
                                <style>
                                @keyframes pulse {{
                                    0% {{ transform: scale(1); }}
                                    50% {{ transform: scale(1.05); }}
                                    100% {{ transform: scale(1); }}
                                }}
                                </style>
                                """, unsafe_allow_html=True)

                                # Display results in an expandable section for detailed analysis
                                with st.expander(" 📊 Detailed Analysis Results", expanded=True):
                                    # Show risk change compared to current probability
                                    if prob_change < 0:
                                        st.success(f"✅ Risk DECREASED by {abs(prob_change):.1%}") 
                                    elif prob_change > 0:
                                        st.error(f"⚠️ Risk INCREASED by {abs(prob_change):.1%} ")
                                    else:
                                        st.info("No change in risk level")
                                    
                                    # Header for modified features summary
                                    st.markdown("##### 📝 Modified Features Summary")
                                    
                                    if all_modifications:
                                        # Categorize modifications for organized display
                                        demographics = {}
                                        clinical = {}
                                        lifestyle = {}
                                        medical = {}
                                        symptoms = {}
                                        
                                        # Assign each feature modification to a category
                                        for feature, new_value in all_modifications.items():
                                            if feature in ['Age', 'BMI']:
                                                demographics[feature] = new_value
                                            elif feature in ['MMSE', 'FunctionalAssessment', 'ADL']:
                                                clinical[feature] = new_value
                                            elif feature in ['PhysicalActivity', 'DietQuality', 'SleepQuality']:
                                                lifestyle[feature] = new_value
                                            elif feature in ['SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CardiovascularDisease', 
                                                            'Diabetes', 'Depression', 'Hypertension', 'Smoking', 'AlcoholConsumption']:
                                                medical[feature] = new_value
                                            else:
                                                symptoms[feature] = new_value
                                        
                                        # List of categories with labels and their feature dicts
                                        categories = [
                                            ("👤 Demographics", demographics),
                                            ("🧠 Clinical Scores", clinical),
                                            ("🏃‍♂️ Lifestyle Factors", lifestyle),
                                            ("💊 Medical Conditions", medical),
                                            ("🧩 Symptoms", symptoms)
                                        ]
                                        
                                        # Iterate through each category to display feature cards
                                        for category_name, category_features in categories:
                                            if category_features:
                                                st.markdown(f"**{category_name}**")
                                                
                                                for feature, new_value in category_features.items():
                                                    # Fetch original value for comparison
                                                    original_value = patient_data_whatif.get(f'RAW_{feature}', 
                                                                                            patient_data_whatif.get(feature, 0))
                                                    
                                                    # Determine feature type (binary vs continuous) for visualization
                                                    feature_type = 'binary' if feature in binary_features else 'continuous'
                                                    
                                                    # Generate visual comparison card
                                                    card_class, orig_display, new_display, change_text, value_class = \
                                                        create_feature_comparison_visual(feature, original_value, new_value, feature_type)
                                                    
                                                    # Get display name for the feature
                                                    display_name = feature_display_names.get(feature, feature)
                                                    
                                                    # Render HTML card showing original vs modified values
                                                    st.markdown(f"""
                                                    <div class="{card_class}">
                                                        <div class="feature-name">{display_name}</div>
                                                        <div class="feature-values">
                                                            <span>Original: <strong>{orig_display}</strong></span>
                                                            <span>→</span>
                                                            <span>New: <strong>{new_display}</strong></span>
                                                            <span class="value-change {value_class}">{change_text}</span>
                                                        </div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                    else:
                                        # Message when no modifications are applied
                                        st.info("No modifications made yet.")

                                # Import function for calculating personalized intervention recommendations
                                from clinical_explanations import calculate_intervention_recommendations

                                # Display personalized recommendations based on modifications
                                with st.expander("💡 Personalized Intervention Recommendations", expanded=True):
                                    recommendations = calculate_intervention_recommendations(all_modifications, new_shap_values)
                                    
                                    if recommendations:
                                        # Separate recommendations by priority
                                        critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
                                        high_recs = [r for r in recommendations if r['priority'] == 'High']
                                        
                                        # Display critical recommendations
                                        if critical_recs:
                                            st.markdown("### 🚨 Critical Actions")
                                            for rec in critical_recs:
                                                st.error(f"**{rec['category']}**: {rec['recommendation']}")
                                                st.caption(f"Impact: {rec['impact']}")
                                        
                                        # Display high priority recommendations
                                        if high_recs:
                                            st.markdown("### ⚠️ High Priority Actions")
                                            for rec in high_recs:
                                                st.warning(f"**{rec['category']}**: {rec['recommendation']}")
                                                st.caption(f"Impact: {rec['impact']}")
                                    
                                    # General recommendations based on predicted risk
                                    st.markdown("### 📋 General Recommendations")
                                    if new_probability >= 0.7:
                                        # High risk general advice
                                        st.markdown("""
                                        <ul style="list-style-type: none; padding-left: 0;">
                                            <li>⚠️ <strong>Immediate comprehensive medical evaluation recommended</strong></li>
                                            <li>🧠 Consider cognitive assessment and neuroimaging</li>
                                            <li>💪 Implement aggressive risk factor modification</li>
                                            <li>⏰ Regular monitoring every 3-6 months</li>
                                        </ul>
                                        """, unsafe_allow_html=True)
                                    elif new_probability >= 0.3:
                                        # Medium risk general advice
                                        st.markdown("""
                                        <ul style="list-style-type: none; padding-left: 0;">
                                            <li>📅 Schedule cognitive screening with primary care provider</li>
                                            <li>🍎 Focus on modifiable risk factor improvement</li>
                                            <li>🤝 Consider joining brain health programs</li>
                                            <li>🗓️ Annual cognitive assessments recommended</li>
                                        </ul>
                                        """, unsafe_allow_html=True)
                                    else:
                                        # Low risk general advice
                                        st.markdown("""
                                        <ul style="list-style-type: none; padding-left: 0;">
                                            <li>✅ Maintain current healthy lifestyle habits</li>
                                            <li>🏥 Continue regular health check-ups</li>
                                            <li>🧩 Stay mentally and socially active</li>
                                            <li>🔍 Monitor for any cognitive changes</li>
                                        </ul>
                                        """, unsafe_allow_html=True)

                            # Exception handling for prediction calculation
                            except Exception as e:
                                st.error(f"Error calculating new prediction: {str(e)}")
                                st.write("Debug info:")
                                st.write(f"Features expected: {feature_names_original}")
                                st.write(f"Modifications: {all_modifications}")

                                # Message when no modifications are applied
                else:
                    st.info("👆 Modify some features above to see how the prediction changes")

            with col2:
                # Button to reset all modifications and start fresh
                if st.button(
                    "🔄 Reset All", 
                    use_container_width=True,
                    key=f"reset_values_btn_{patient_id_whatif}_{reset_key}"
                ):
                    # Increment a counter to track resets (could be used to trigger UI updates)
                    st.session_state.reset_counter += 1
                    
                    # Clear all stored modifications in session state
                    st.session_state.all_modifications = {}
                    
                    # Remove any patient-specific modified values from session state
                    if f'modified_values_{patient_id_whatif}' in st.session_state:
                        del st.session_state[f'modified_values_{patient_id_whatif}']
                    
                    # Rerun the Streamlit app to refresh UI with reset values
                    st.rerun()

            # Export functionality for modified analysis
            if all_modifications:
                # Expandable section for exporting analysis results
                with st.expander("📥 Export Analysis", expanded=False):
                    # Prepare data to export
                    export_data = {
                        'Patient_ID': patient_id_whatif,  # Track which patient the analysis is for
                        'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Timestamp of export
                        'Original_Risk': f"{current_prob:.1%}",  # Original risk before modifications
                        'Modifications': all_modifications  # Dictionary of all feature changes applied
                    }
                    
                    # Convert export data to JSON string for download
                    json_str = json.dumps(export_data, indent=2)
                    
                    # Streamlit download button to save analysis report as a JSON file
                    st.download_button(
                        label="📥 Download Analysis Report (JSON)",
                        data=json_str,
                        file_name=f"whatif_analysis_{patient_id_whatif}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

    # Run the main function to execute the What-If analysis on the filtered dataset
    run(df_filtered)


