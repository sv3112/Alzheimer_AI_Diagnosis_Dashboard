# homepage.py - Main Entry Point
"""
Streamlit Application Homepage for Multimodal Explainable AI for Alzheimer's Diagnosis

This module serves as the main landing page for the AI-powered Alzheimer's detection system.
It provides an overview of the system architecture, performance metrics, technology stack,
and navigation to other sections of the application.

This file is intended to be run directly using Streamlit:
streamlit run homepage.py
"""

import streamlit as st
from style import apply_custom_css

# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------
# Sets up the Streamlit page with title, icon, layout, and default sidebar state
# This configuration affects the browser tab title and overall page layout
st.set_page_config(
    page_title="Multimodal Explainable AI for Alzheimer's Diagnosis",  # Browser tab title
    page_icon="🧠",  # Browser tab icon (brain emoji)
    layout="wide",   # Use full width of browser window
    initial_sidebar_state="collapsed"  # Start with sidebar hidden for cleaner look
)

# -----------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------
def init_session_state():
    """
    Initialize session state variables if they don't exist yet.
    
    Session state in Streamlit persists data across user interactions and page reloads.
    This function ensures that key application state variables are properly initialized
    with default values to prevent KeyError exceptions and maintain consistent behavior.
    
    Session State Variables:
    - page: Tracks the current page/section being viewed
    - uploaded_data: Stores any dataset uploaded by the user
    - model_trained: Boolean flag tracking whether models have been trained
    """
    defaults = {
        'page': 'main',               # Default to main homepage
        'uploaded_data': None,        # No data uploaded initially
        'model_trained': False        # Models not trained initially
    }
    
    # Only set default values for keys that don't already exist
    # This prevents overwriting existing session state data
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------------------------------------
# UI Components
# -----------------------------------------------------------

def create_hero_section():
    """
    Render the hero/intro section with title, subtitle, and feature badges.
    
    This function creates the main header section that users see first when loading
    the homepage. It includes:
    - Main title highlighting the AI-powered nature of the system
    - Subtitle emphasizing transparency and clinical decision-making
    - Feature badges showcasing key capabilities
    
    The badges are rendered as styled HTML elements using custom CSS classes
    defined in the style.py module.
    """
    # Define the key features/capabilities to display as badges
    badges = [
        "🎯 Clinical Binary Classification",      # Clinical data → Demented/Non-Demented
        "🧠 MRI-Based 4-Stage Classification",   # MRI images → 4 severity stages
        "📊 Explainability Suite (SHAP & LIME)" # Interpretability tools
    ]
    
    # Convert badges list into HTML span elements with custom styling
    badges_html = "".join([f'<span class="badge">{badge}</span>' for badge in badges])
    
    # Render the complete hero section using markdown with HTML
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">🧠 AI-Powered Multimodal Framework for Alzheimer's Detection and Interpretation</h1>
        <p class="hero-subtitle">From Black Box to Glass Box: Transparent AI for Clinical Decision-Making</p>
        <div style="margin-top: 1rem;">{badges_html}</div>
    </div>
    """, unsafe_allow_html=True)

def create_architecture_card(icon: str, title: str, description: str):
    """
    Create a card element for architecture visualization.
    
    Args:
        icon (str): Emoji or icon character to display at the top of the card
        title (str): Card title/heading text
        description (str): Detailed description of the architecture component
    
    Returns:
        str: HTML string representing a styled card component
        
    This function generates individual cards that explain different aspects of the
    system architecture. Each card visually represents one key component or concept
    in an easily digestible format.
    """
    return f"""
    <div class="card">
        <div class="card-icon">{icon}</div>
        <h3 class="card-title">{title}</h3>
        <div class="card-description">{description}</div>
    </div>
    """

def create_metric_card(value: str, label: str, tooltip: str):
    """
    Create a metric card styled as a clickable button with tooltip.
    
    Args:
        value (str): The main metric value to display prominently (e.g., "94.42%")
        label (str): The metric name/label (e.g., "Accuracy")
        tooltip (str): Explanatory text shown on hover
    
    Returns:
        str: HTML string for a styled metric card with hover tooltip
        
    These cards display key performance metrics in an attractive, interactive format.
    The button-like styling makes them visually appealing while tooltips provide
    additional context without cluttering the interface.
    """
    return f"""
    <div class="metric-card-button" title="{tooltip}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_feature_item(step: str, description: str, tooltip: str):
    """
    Create a single step/feature description with tooltip support.
    
    Args:
        step (str): The step number and name (e.g., "1️⃣ Data Preprocessing")
        description (str): Brief description of what this step does
        tooltip (str): Detailed explanation shown on hover
    
    Returns:
        str: HTML string for a workflow step with interactive tooltip
        
    This function is used to create individual items in the ML workflow pipelines.
    Each item represents one step in the process, with concise descriptions and
    detailed tooltips for users who want more information.
    """
    return f"""
    <div class="feature-item">
        <div class="tooltip">
            <strong>{step}</strong>: {description}
            <span class="tooltip-text">{tooltip}</span>
        </div>
    </div>
    """

def create_section_header(title: str, subtitle: str = ""):
    """
    Render a section header with optional subtitle.
    
    Args:
        title (str): Main section title
        subtitle (str, optional): Additional subtitle text. Defaults to empty string.
    
    Returns:
        str: HTML string for a consistently styled section header
        
    This function ensures consistent styling across all major sections of the homepage.
    It provides visual hierarchy and clear separation between different content areas.
    """
    # Only include subtitle HTML if subtitle text is provided
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin: 0rem 0;">
        <h2 class="section-title">{title}</h2>
        {subtitle_html}
    </div>
    """

# -----------------------------------------------------------
# Data Definitions for Display Components
# -----------------------------------------------------------

# Architecture cards data - explains the four key aspects of the system
ARCHITECTURE_CARDS = [
    # Each dictionary represents one card explaining a key part of the system's architecture
    {
        "icon": "📊",
        "title": "Independent Data Modalities",
        "description": "Two separate prediction pathways: clinical data (34 features) or MRI imaging data. Each pathway can work independently, allowing Alzheimer's prediction even if only one data type is available.",
    },
    {
        "icon": "🧠", 
        "title": "Parallel Processing",
        "description": "Dedicated models for each data type: a traditional machine learning model (CatBoost) for clinical data and a deep convolutional neural network (Inception V3) for MRI images.",
    },
    {
        "icon": "🎯",
        "title": "Dual Classification System", 
        "description": "Clinical data enables binary classification (Demented vs. Non-Demented). MRI images provide a detailed 4-stage classification (Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented).",
    },
    {
        "icon": "🔍",
        "title": "Explainable AI Suite",
        "description": "<strong>SHAP</strong>: Quantifies feature contributions to predictions<br><strong>Grad-CAM</strong>: Shows which brain areas the model looks at the most on MRI images.<br> Gives a score for how important each brain region is in the model's decision",
    }
]

# Model performance metrics - these would typically come from model evaluation results
# In a production system, these might be loaded from a database or config file
clinical_accuracy = 94.42  # Clinical model accuracy percentage
clinical_f1 = 0.95         # Clinical model F1-score (0-1 scale)
clinical_auc = 0.9429      # Clinical model AUC-ROC score

mri_accuracy = 95.0        # MRI model accuracy percentage  
mri_f1 = 0.946            # MRI model F1-score (0-1 scale)
mri_kappa = 0.9285        # MRI model Cohen's Kappa (agreement measure)

# Calculate average metrics across both models for summary display
average_accuracy = round((clinical_accuracy + mri_accuracy) / 2, 2)
average_f1 = round(((clinical_f1 * 100) + (mri_f1 * 100)) / 2, 2)  # Convert to percentage
average_auc = round((clinical_auc + 0.9959) / 2, 3)  # 0.9959 is MRI model AUC

# Metrics to display in the performance section
# Each tuple contains: (display_value, metric_name, tooltip_explanation)
METRICS = [
    (f"{average_accuracy}%", "Average Accuracy", "The overall percentage of correct predictions made by the Clinical and MRI models."),
    (f"{average_auc}", "Average AUC-ROC", "How well the model can tell apart different classes (higher is better)."),
    (f"{average_f1}%", "Average F1-Score", "A balance between how many correct positive results the model finds and how many it misses."),
    ("92.85%", "Cohen's Kappa (MRI Only)", "How much the MRI model's predictions agree with the true results beyond chance."),
    ("93.48%", "Average Precision (Clinical Only)", "The accuracy of the Clinical model when identifying positive cases.")
]

# Technology stack used in the project
# Each tuple contains: (technology_name, description_for_tooltip)
TECH_STACK = [
    # Programming Language
    ("Python 3.9+", "Primary programming language"),
    # ML/DL frameworks  
    ("scikit-learn", "Machine learning algorithms and tools"),
    ("CatBoost", "Gradient boosting algorithm used for clinical data"),
    ("TensorFlow / Keras", "Deep learning framework (used for CNNs)"),
    ("PyTorch", "Alternative deep learning framework (if applicable)"),
    # Explainability tools
    ("SHAP", "Model interpretability for tabular data"),
    ("Grad-CAM", "CNN-based visualization of important MRI regions"),
    ("LIME", "Local interpretable model-agnostic explanations (optional)"),
    # Data handling libraries
    ("Pandas", "Data manipulation and analysis"),
    ("NumPy", "Numerical computing and array operations"),
    ("OpenCV", "Image processing and computer vision tasks"),
    # Visualization tools
    ("Plotly", "Interactive data visualizations in dashboard"),
    ("matplotlib", "Basic plotting for analysis"),
    ("seaborn", "Statistical visualizations and distributions"),
    # Deployment framework
    ("Streamlit", "Web application framework for interactive dashboard"),
    ("SQLite3", "Lightweight database for storing predictions and explanations"),
    # Utility libraries
    ("joblib", "Model and pipeline serialization"),
]

# -----------------------------------------------------------
# Main Application
# -----------------------------------------------------------
def main():
    """
    Main Streamlit app function.
    
    This is the primary function that orchestrates the entire homepage display.
    It handles:
    1. Session state initialization
    2. Custom CSS styling application  
    3. Rendering of all homepage sections in order
    4. Navigation between different pages
    
    The function is structured to render sections from top to bottom as they
    appear on the webpage, maintaining a logical flow for users.
    """
    # Initialize session state variables and apply custom CSS styling
    init_session_state()
    apply_custom_css()
    
    # === HERO SECTION ===
    # Main header with title, subtitle, and feature badges
    create_hero_section()
    
    # === ARCHITECTURE OVERVIEW SECTION ===
    # Explain the system's approach to multimodal AI
    st.markdown(create_section_header(
        "Breaking the Black Box: From Opaque AI to Transparent Healthcare",
        "Empowering clinicians with AI decisions that are clear, reliable, and clinically actionable"
    ), unsafe_allow_html=True)
    
    # Display architecture cards in a 4-column layout
    # Each card explains one key aspect of the system architecture
    cols = st.columns(4)
    for i, card_data in enumerate(ARCHITECTURE_CARDS):
        with cols[i]:
            st.markdown(create_architecture_card(**card_data), unsafe_allow_html=True)
    
    # === MACHINE LEARNING WORKFLOW SECTION ===
    # Visual separator and section header
    st.markdown("--- ")
    st.markdown(create_section_header("🚀 Machine Learning Workflow & Pipeline"), unsafe_allow_html=True)
    
    # Define detailed workflow steps for both clinical and MRI pipelines
    # These lists explain the step-by-step process for each data modality
    
    # Clinical data processing pipeline (6 steps)
    clinical_workflow = [
        ("1️⃣ Data Preprocessing", "Handle missing values, normalize numerical features, and encode categorical variables.", "Prepares and cleans raw clinical data to ensure accuracy and consistency before modeling."),
        ("2️⃣ Feature Selection", "Combine Correlation analysis, Information Gain, and Chi-Square filtering to rank features.", "Selects the most relevant and predictive clinical variables to enhance model performance."),
        ("3️⃣ Top 10 Features", "Choose the top 10 features based on the averaged ranks from Correlation, Information Gain, and Chi-Square scores.", "Focuses the analysis on the most informative features, reducing complexity and improving interpretability."),
        ("4️⃣ Voting Classifier", "Ensemble approach combining multiple algorithms to produce robust and stable predictions.", "Leverages the strengths of different models to increase reliability and reduce overfitting."),
        ("5️⃣ CatBoost Model", "Final optimized CatBoost model for binary classification (Demented vs. Non-Demented).", "A powerful gradient boosting algorithm designed for tabular data, offering high accuracy and efficiency."),
        ("6️⃣ SHAP Analysis", "Generate global feature importance and individual-level explanations using SHAP values.", "Provides transparent, interpretable insights into how each feature influences the model's decisions.")
    ]

    # MRI image processing pipeline (6 steps) 
    mri_workflow = [
        ("1️⃣ Image Preprocessing", "Resize, normalize, and augment MRI scans.", "Standardizes brain images to improve model performance and generalizability."),
        ("2️⃣ Feature Extraction", "Automatically extract features through convolutional neural network (CNN) layers.", "Enables the model to learn complex patterns and structural markers directly from brain images."),
        ("3️⃣ Inception V3", "Use a pre-trained Inception V3 model fine-tuned for 4-stage dementia classification.", "Leverages a powerful deep learning architecture specifically adapted for brain scan analysis."),
        ("4️⃣ Model Prediction", "Generate probability scores for each dementia stage (Non-Demented, Very Mild, Mild, Moderate Demented).", "Provides confidence levels for staging and supports personalized clinical decision-making."),
        ("5️⃣ Grad-CAM", "Visualize critical brain regions that influence model predictions using Grad-CAM heatmaps.", "Highlights areas of focus, enhancing transparency and interpretability for clinicians."),
        ("6️⃣ LIME Visualization", "Segments the brain scan into interpretable regions and quantifies each region's importance score for a specific prediction, allowing detailed region-wise contribution analysis.", "Enabling numerical analysis of which brain areas contributed most to each individual prediction.")
    ]
    
    # Display both pipelines side-by-side in a two-column layout
    pipe_col1, pipe_col2 = st.columns(2)
    with pipe_col1:
        st.markdown("### 📊 Clinical Data Pipeline")
        # Render each step in the clinical workflow
        for step in clinical_workflow:
            st.markdown(create_feature_item(*step), unsafe_allow_html=True)
    with pipe_col2:
        st.markdown("### 🧠 MRI Image Pipeline") 
        # Render each step in the MRI workflow
        for step in mri_workflow:
            st.markdown(create_feature_item(*step), unsafe_allow_html=True)
    
    # === PERFORMANCE METRICS SECTION ===
    # Display key performance indicators in an attractive card layout
    st.markdown(create_section_header("🏆 Model Performance & Clinical Impact"), unsafe_allow_html=True)
    cols = st.columns(5)  # Create 5 columns for the 5 metrics
    for i, metric in enumerate(METRICS):
        with cols[i]:
            st.markdown(create_metric_card(*metric), unsafe_allow_html=True)
    
    # === NAVIGATION BUTTONS SECTION ===
    # Provide easy access to other sections of the application
    st.markdown(create_section_header("Ready to Explore the System?"), unsafe_allow_html=True)
    
    # Center the navigation buttons using column layout
    col1, col2, col3 = st.columns([1, 3, 1])  # [left_spacer, content, right_spacer]
    with col2:  # Use the middle column for content
        subcol1, subcol2, subcol3 = st.columns(3)  # Three equal columns for buttons
        
        # Upload & Analyze Data button
        with subcol1:
            if st.button("📤 Upload & Analyze Data", key="main_upload_btn", use_container_width=True, help="Upload new patient data for analysis"):
                st.switch_page("pages/uploaddataPage.py")  # Navigate to upload page
        
        # Clinical Dashboard button        
        with subcol2:
            if st.button("📊 Clinical Dashboard", key="main_dashboard_btn", use_container_width=True, help="View predictions and analytics"):
                st.switch_page("pages/ClinicalDashboardPage.py")  # Navigate to clinical dashboard
        
        # MRI Dashboard button
        with subcol3:
            if st.button("🧠 MRI Dashboard", key="main_mri_dashboard_btn", use_container_width=True, help="View MRI predictions and visualizations"):
                st.switch_page("pages/MRIDashboardPage.py")  # Navigate to MRI dashboard
    
    # === TECHNOLOGY STACK SECTION ===
    # Display all technologies used in the project as interactive badges
    tech_badges = "".join([f'<span class="tech-badge tooltip">{tech[0]}<span class="tooltip-text">{tech[1]}</span></span>' for tech in TECH_STACK])
    st.markdown(f"""
    <div style="background: var(--bg-light); padding: 1rem; border-radius: 10px; margin: 2rem 0; text-align: center;">
        <h3 style="color: var(--text-primary); margin-bottom: 1rem;">🛠️ Technology Stack</h3>
        <div>{tech_badges}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === FOOTER SECTION ===
    # Project attribution and copyright information
    st.markdown("---")  # Horizontal separator line
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
        <p style="font-size: 1.1rem; font-weight: 500;">Master's Project in Artificial Intelligence & Healthcare</p>
        <p style="font-size: 0.9rem;">© 2025 | Multimodal Explainable AI for Alzheimer's Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    
    main()