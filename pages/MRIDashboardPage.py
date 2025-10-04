# MRIDashboardPage.py - MRI Dashboard Using Existing GitHub Database

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

warnings.filterwarnings('ignore')

from style import apply_custom_css
import re
import time

apply_custom_css()

# ------------------------------
# DATABASE PATH SETUP
# ------------------------------

def get_database_directory():
    """Find the Alzheimer_Database directory in the repository"""
    current_file = Path(__file__).resolve()
    
    # Strategy 1: Check Alzheimer_Project folder structure
    for parent in list(current_file.parents):
        db_dir = parent / 'Alzheimer_Project' / 'Alzheimer_Database'
        if db_dir.exists() and (db_dir / 'alzheimer_predictions.db').exists():
            return db_dir
    
    # Strategy 2: Look for repository root (has .git folder)
    for parent in list(current_file.parents):
        if (parent / '.git').exists():
            db_dir = parent / 'Alzheimer_Database'
            if db_dir.exists():
                return db_dir
            # Also check inside Alzheimer_Project
            db_dir = parent / 'Alzheimer_Project' / 'Alzheimer_Database'
            if db_dir.exists():
                return db_dir
    
    # If not found, return expected location
    return current_file.parent / 'Alzheimer_Database'

# Set database directory
DB_DIR = get_database_directory()
DB_PATH = DB_DIR / 'alzheimer_predictions.db'

# Add repository root to Python path for imports
repo_root = DB_DIR.parent
sys.path.insert(0, str(repo_root))

# Also try Alzheimer_Project if it exists
if (repo_root / 'Alzheimer_Project').exists():
    sys.path.insert(0, str(repo_root / 'Alzheimer_Project'))

# Import database storage class
try:
    from alzheimers_db_setup import AlzheimerPredictionStorage
    DB_AVAILABLE = True
except ImportError as e:
    st.error(f"Cannot import database module: {e}")
    DB_AVAILABLE = False

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="AI-Powered Alzheimer's MRI Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# CLINICAL EXPLANATIONS
# ------------------------------
try:
    from clinical_explanations import (
        generate_region_explanation, 
        generate_patient_narrative, 
        create_interactive_region_selector,
        get_clinical_insights,
        generate_comparative_narrative
    )
    CLINICAL_FEATURES_AVAILABLE = True
except ImportError:
    CLINICAL_FEATURES_AVAILABLE = False

# ------------------------------
# SESSION STATE
# ------------------------------
if 'prevent_rerun' not in st.session_state:
    st.session_state.prevent_rerun = False
if 'last_data_check' not in st.session_state:
    st.session_state.last_data_check = 0

# ------------------------------
# HERO SECTION
# ------------------------------
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">🧠 AI-Powered Alzheimer's MRI Analysis Dashboard</h1>
    <p class="hero-subtitle">Advanced Deep Learning Analysis with Clinical Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# DATA LOADING
# ------------------------------

@st.cache_data(ttl=60, show_spinner="Loading dashboard data...")
def load_mri_data_optimized(force_refresh=False):
    """Load MRI data from existing GitHub repository database"""
    
    if not DB_AVAILABLE:
        return {
            'batch_predictions': pd.DataFrame(),
            'batch_regions': pd.DataFrame(),
            'stored_images': pd.DataFrame(),
            'load_timestamp': datetime.now(),
            'total_records': 0,
            'error': 'Database module not available'
        }
    
    if not DB_PATH.exists():
        return {
            'batch_predictions': pd.DataFrame(),
            'batch_regions': pd.DataFrame(),
            'stored_images': pd.DataFrame(),
            'load_timestamp': datetime.now(),
            'total_records': 0,
            'error': f'Database not found at: {DB_PATH}'
        }
    
    start_time = time.time()
    original_dir = os.getcwd()
    
    try:
        # Change to database directory
        os.chdir(str(DB_DIR))
        
        # Create storage handler with explicit base directory
        storage = AlzheimerPredictionStorage(base_dir=str(DB_DIR))
        
        # Debug: Check what tables exist in the database
        import sqlite3
        conn = sqlite3.connect('alzheimer_predictions.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Debug: Check row counts for each table
        table_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_info[table_name] = count
        
        conn.close()
        
        # Load predictions with better error handling
        batch_predictions_raw = storage.get_batch_predictions()
        
        # Handle different return types
        if batch_predictions_raw:
            if isinstance(batch_predictions_raw, pd.DataFrame):
                batch_predictions_df = batch_predictions_raw
            elif isinstance(batch_predictions_raw, list) and len(batch_predictions_raw) > 0:
                # Check if it's a list of dicts or list of tuples
                if isinstance(batch_predictions_raw[0], dict):
                    batch_predictions_df = pd.DataFrame(batch_predictions_raw)
                else:
                    # It's likely tuples - need to get column names from cursor
                    cursor.execute("PRAGMA table_info(batch_predictions)")
                    columns = [col[1] for col in cursor.fetchall()]
                    batch_predictions_df = pd.DataFrame(batch_predictions_raw, columns=columns)
            else:
                batch_predictions_df = pd.DataFrame()
        else:
            batch_predictions_df = pd.DataFrame()
        
        # Debug info
        debug_info = {
            'tables': table_info,
            'predictions_raw_type': type(batch_predictions_raw).__name__,
            'predictions_raw_count': len(batch_predictions_raw) if batch_predictions_raw else 0,
            'predictions_df_shape': batch_predictions_df.shape if not batch_predictions_df.empty else (0, 0),
            'predictions_df_columns': list(batch_predictions_df.columns) if not batch_predictions_df.empty else []
        }
        
        # Clean confidence values
        if not batch_predictions_df.empty and 'Confidence' in batch_predictions_df.columns:
            def clean_confidence(val):
                if pd.isna(val):
                    return 0.0
                if isinstance(val, str):
                    try:
                        cleaned = float(val.replace('%', '').strip())
                        return cleaned / 100.0 if cleaned > 1 else cleaned
                    except:
                        return 0.0
                return float(val) if val <= 1 else float(val) / 100.0
            
            batch_predictions_df['Confidence'] = batch_predictions_df['Confidence'].apply(clean_confidence)
            
            # Convert probability columns
            prob_columns = ['Mild_Demented_Probability', 'Moderate_Demented_Probability', 
                          'Non_Demented_Probability', 'Very_Mild_Demented_Probability']
            for col in prob_columns:
                if col in batch_predictions_df.columns:
                    batch_predictions_df[col] = pd.to_numeric(batch_predictions_df[col], errors='coerce').fillna(0)
        
        # Load regions with better error handling
        batch_regions_raw = storage.get_batch_region()
        
        if batch_regions_raw:
            if isinstance(batch_regions_raw, pd.DataFrame):
                batch_regions_df = batch_regions_raw
            elif isinstance(batch_regions_raw, list) and len(batch_regions_raw) > 0:
                if isinstance(batch_regions_raw[0], dict):
                    batch_regions_df = pd.DataFrame(batch_regions_raw)
                else:
                    cursor.execute("PRAGMA table_info(batch_region)")
                    columns = [col[1] for col in cursor.fetchall()]
                    batch_regions_df = pd.DataFrame(batch_regions_raw, columns=columns)
            else:
                batch_regions_df = pd.DataFrame()
        else:
            batch_regions_df = pd.DataFrame()
        
        if not batch_regions_df.empty:
            numeric_cols = ['ScoreCAM_Importance_Score', 'ScoreCAM_Importance_Percentage']
            for col in numeric_cols:
                if col in batch_regions_df.columns:
                    batch_regions_df[col] = pd.to_numeric(batch_regions_df[col], errors='coerce').fillna(0)
        
        # Load images
        stored_images_raw = storage.get_stored_images()
        stored_images_df = pd.DataFrame(stored_images_raw) if stored_images_raw else pd.DataFrame()
        
        # Fix image paths to work with repository structure
        if not stored_images_df.empty and 'file_path' in stored_images_df.columns:
            def fix_image_path(path_str):
                if not path_str:
                    return path_str
                
                # If it's already a valid absolute path, use it
                p = Path(path_str)
                if p.is_absolute() and p.exists():
                    return str(p)
                
                # Extract just the filename from the path
                filename = Path(path_str).name
                
                # Look for the file in stored_images directory
                stored_images_dir = DB_DIR / 'stored_images'
                if stored_images_dir.exists():
                    image_path = stored_images_dir / filename
                    if image_path.exists():
                        return str(image_path)
                
                # Try direct path relative to database directory
                rel_path = DB_DIR / path_str
                if rel_path.exists():
                    return str(rel_path)
                
                # Return original if nothing works (will show error in display)
                return path_str
            
            stored_images_df['file_path'] = stored_images_df['file_path'].apply(fix_image_path)
            
            # Add debug: count how many images actually exist
            debug_info['images_found'] = stored_images_df['file_path'].apply(lambda x: Path(x).exists()).sum()
            debug_info['images_total'] = len(stored_images_df)
        
        storage.close()
        
        return {
            'batch_predictions': batch_predictions_df,
            'batch_regions': batch_regions_df,  # FIXED: Changed from 'region_predictions'
            'stored_images': stored_images_df,
            'load_timestamp': datetime.now(),
            'total_records': len(batch_predictions_df) + len(batch_regions_df) + len(stored_images_df),
            'error': None,
            'debug_info': debug_info
        }
        
    except Exception as e:
        return {
            'batch_predictions': pd.DataFrame(),
            'batch_regions': pd.DataFrame(),  # FIXED: Changed from 'region_predictions'
            'stored_images': pd.DataFrame(),
            'load_timestamp': datetime.now(),
            'total_records': 0,
            'error': str(e)
        }
    finally:
        os.chdir(original_dir)

# ------------------------------
# IMAGE DISPLAY
# ------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def display_image_from_db_cached(image_path, caption=""):
    """Cache image existence checks"""
    try:
        if isinstance(image_path, str) and ('/' in image_path or '\\' in image_path):
            if os.path.exists(image_path):
                return image_path, True, os.path.getmtime(image_path)
        return None, False, 0
    except:
        return None, False, 0

def display_image_from_db(image_data, caption="Image"):
    """Display image from database"""
    if image_data:
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            
            # File path
            if isinstance(image_data, str) and ('/' in image_data or '\\' in image_data):
                cached_path, exists, mod_time = display_image_from_db_cached(image_data, caption)
                if exists and cached_path:
                    st.image(cached_path, caption=caption, use_container_width=True)
                    return True
                else:
                    st.error(f"Image not found: {image_data}")
                    return False
            
            # Data URL
            elif isinstance(image_data, str) and image_data.startswith('data:image'):
                st.image(image_data, caption=caption, use_container_width=True)
                return True
            
            # Base64 string
            elif isinstance(image_data, str):
                try:
                    img_data = base64.b64decode(image_data)
                    img = Image.open(BytesIO(img_data))
                    st.image(img, caption=caption, use_container_width=True)
                    return True
                except:
                    st.error("Invalid image data")
                    return False
            
            # Bytes
            elif isinstance(image_data, bytes):
                img = Image.open(BytesIO(image_data))
                st.image(img, caption=caption, use_container_width=True)
                return True
            
            else:
                st.image(image_data, caption=caption, use_container_width=True)
                return True
                
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            return False
    return False

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def extract_patient_id(filename):
    """Extract patient ID from filename"""
    patterns = [
        r'patient[_\s]*(\d+)',
        r'P(\d+)',
        r'scan[_\s]*(\d+)',
        r'(\d+)',
        r'([A-Za-z]+\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1) if pattern != r'([A-Za-z]+\d+)' else match.group(1)
    return None

@st.cache_data(ttl=120)
def get_patient_images_optimized(patient_filename, df_images):
    """Get images for a patient"""
    if df_images.empty:
        return pd.DataFrame()
    
    patient_id = extract_patient_id(patient_filename)
    
    if patient_id and 'patient_id' in df_images.columns:
        scan_images = df_images[df_images['patient_id'].astype(str) == str(patient_id)]
        if not scan_images.empty:
            return scan_images
    
    scan_base = patient_filename.split('.')[0] if '.' in patient_filename else patient_filename
    scan_images = df_images[df_images['filename'].str.contains(scan_base, case=False, na=False)]
    
    return scan_images

# ------------------------------
# LOAD DATA
# ------------------------------

with st.spinner("Loading dashboard data..."):
    data_dict = load_mri_data_optimized()
    
    df_predictions = data_dict.get('batch_predictions', pd.DataFrame())
    df_regions = data_dict.get('batch_regions', pd.DataFrame())  # FIXED: Changed from 'region_predictions'
    df_images = data_dict.get('stored_images', pd.DataFrame())
    total_records = data_dict.get('total_records', 0)
    error = data_dict.get('error')

# Show status
if error:
    st.error(f"Error loading data: {error}")
    
    # Show detailed debugging info
    with st.expander("Debug Information"):
        st.write("**Current file location:**", Path(__file__).resolve())
        st.write("**Looking for database at:**", DB_PATH)
        st.write("**Database directory:**", DB_DIR)
        st.write("**Database exists:**", DB_PATH.exists())
        
        st.write("\n**Checking parent directories:**")
        current = Path(__file__).resolve()
        for i, parent in enumerate(current.parents):
            st.write(f"Level {i}: {parent}")
            db_check = parent / 'Alzheimer_Database' / 'alzheimer_predictions.db'
            st.write(f"  - Has database? {db_check.exists()}")
            if (parent / '.git').exists():
                st.write(f"  - Repository root!")
    
    st.stop()
elif total_records > 0:
    st.success(f"Connected to database | {total_records} records loaded from {DB_DIR.name}")
else:
    st.warning("No data found in database")
    st.stop()

# Add debug display to verify data loaded correctly
with st.expander("Data Loading Summary"):
    st.write(f"**Batch Predictions:** {len(df_predictions)} records")
    st.write(f"**Brain Regions:** {len(df_regions)} records")
    st.write(f"**Stored Images:** {len(df_images)} records")
    
    if not df_regions.empty:
        st.write("\n**Sample Region Data:**")
        st.dataframe(df_regions.head())

@st.cache_data(ttl=60)
def calculate_metrics_fast(df_predictions):
    """Calculate dashboard metrics"""
    if df_predictions.empty:
        return 0, 0, 0, 0, 0
    
    total_scans = len(df_predictions)
    class_counts = df_predictions['Predicted_Class'].value_counts()
    
    return (
        total_scans,
        class_counts.get('Non Demented', 0),
        class_counts.get('Very Mild Demented', 0),
        class_counts.get('Mild Demented', 0),
        class_counts.get('Moderate Demented', 0)
    )

total_scans, non_demented, very_mild_demented, mild_demented, moderate_demented = calculate_metrics_fast(df_predictions)

# Metrics display
col1, col2, col3, col4, col5 = st.columns(5)

metrics_data = [
    (col1, total_scans, "Total Scans"),
    (col2, non_demented, "Non-Demented"),
    (col3, very_mild_demented, "Very Mild"),
    (col4, mild_demented, "Mild"),
    (col5, moderate_demented, "Moderate")
]

for col, value, label in metrics_data:
    with col:
        st.markdown(f"""
        <div class="metric-card_D">
            <div class="metric-value_D">{value}</div>
            <div class="metric-label_D">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("Dashboard is now connected to your existing database and images!")
st.info("The remaining visualization code from your original file continues here...")
# ------------------------------
# 🗂 Tabs Section
# ------------------------------
st.markdown("")

# Initialize session state for selected tab and last action
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "📊 Overview Dashboard"
if 'last_action' not in st.session_state:
    st.session_state.last_action = None

st.markdown(" ")

# ------------------------------
# 🖱 Tab navigation buttons
# ------------------------------
col1, col2, col3 = st.columns(3)

# Overview Dashboard tab
with col1:
    if st.button(
        "📊 Overview Dashboard", 
        key="tab_nav_overview",
        use_container_width=True,
        type="primary" if st.session_state.selected_tab == "📊 Overview Dashboard" else "secondary"
    ):
        st.session_state.selected_tab = "📊 Overview Dashboard"
        st.session_state.last_action = 'tab_change'
        st.rerun()  # Force re-render with new tab

# Individual Patient Analysis tab
with col2:
    if st.button(
        "🧬 Individual Patient Analysis", 
        key="tab_nav_individual",
        use_container_width=True,
        type="primary" if st.session_state.selected_tab == "🧬 Individual Patient Analysis" else "secondary"
    ):
        st.session_state.selected_tab = "🧬 Individual Patient Analysis"
        st.session_state.last_action = 'tab_change'
        st.rerun()

# Comparative Patient Insights tab
with col3:
    if st.button(
        "⚖️ Comparative Patient Insights", 
        key="tab_nav_comparative",
        use_container_width=True,
        type="primary" if st.session_state.selected_tab == "⚖️ Comparative Patient Insights" else "secondary"
    ):
        st.session_state.selected_tab = "⚖️ Comparative Patient Insights"
        st.session_state.last_action = 'tab_change'
        st.rerun()

# If the currently selected tab in the dashboard is the "Overview Dashboard"
if st.session_state.selected_tab == "📊 Overview Dashboard":
    # Create two equal-width columns for layout
    col1, col2 = st.columns([2, 2])


    # =========================
    # LEFT COLUMN: Classification Distribution Pie Chart
    # =========================
    with col1:
        # Section title for classification distribution
        st.markdown('<h4 class="subsection-title">🎯 Classification Distribution</h4>', unsafe_allow_html=True)
        
        # Only display the chart if prediction data is available
        if not df_predictions.empty:
            # Count how many predictions fall into each class
            class_counts = df_predictions['Predicted_Class'].value_counts()
            
            # Define the exact same colors as your Risk Assessment cards
            color_map = {
                'Non Demented': '#10b981',      # Green - matches "No Risk"
                'Very Mild Demented': '#fde047', # Yellow - matches "Low Risk"  
                'Mild Demented': '#f59e0b',     # Orange - matches "Moderate Risk"
                'Moderate Demented': '#dc2626'  # Red - matches "High Risk"
            }
            
            # Get colors in the same order as the data
            colors = [color_map.get(class_name, '#888888') for class_name in class_counts.index]
            
            # Create pie chart using go.Figure for direct color control
            fig_pie = go.Figure()
            
            fig_pie.add_trace(go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.6,  # Creates donut effect
                marker=dict(
                    colors=colors,  # Apply our custom colors directly
                    line=dict(color='#FFFFFF', width=3)
                ),
                textposition='auto',
                textinfo='percent+label',
                textfont=dict(size=14, color='white', family='Inter, sans-serif'),
                texttemplate='<b>%{label}</b><br>%{percent}',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                pull=[0.1 if 'Non Demented' in label else 0.05 for label in class_counts.index]
            ))

            # Add center text showing total
            fig_pie.add_annotation(
                text=f'<b>{len(df_predictions)}</b><br>Total',
                x=0.5, y=0.5,
                font=dict(size=20, color='#1f2937', family='Inter, sans-serif'),
                showarrow=False
            )

            # Update layout to match your style
            fig_pie.update_layout(
                font=dict(family="Inter, sans-serif", size=16),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle", y=0.5,
                    xanchor="left", x=0.85,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#E5E5E5',
                    borderwidth=1,
                    font=dict(size=13, color='#000000')
                ),
                margin=dict(l=0, r=200, t=30, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=450
            )

            # Display the chart
            st.plotly_chart(fig_pie, use_container_width=True)
            
        else:
            # Inform the user if no data is available
            st.info("📊 No scan data available for visualization")



    # =========================
    # RIGHT COLUMN: Risk Assessment Summary
    # =========================
    with col2:
        # Section title for risk summary
        st.markdown('<h4 class="subsection-title">🔍 Risk Assessment Summary</h4>', unsafe_allow_html=True)
        
        # Only display risk summary if data is available
        if not df_predictions.empty:
            # Dictionary mapping risk categories to their counts
            risk_categories = {
                'No Risk': non_demented,
                'Low Risk': very_mild_demented,
                'Moderate Risk': mild_demented,
                'High Risk': moderate_demented
            }
            
            # Loop through each risk category and create a styled card
            for risk_level, count in risk_categories.items():
                # Calculate percentage of total scans for this risk level
                percentage = (count / total_scans) * 100 if total_scans > 0 else 0
                
                # Assign colors and icons based on risk level
                if risk_level == 'No Risk':
                    color = '#10b981'  # Green
                    icon = '🟢 '
                elif risk_level == 'Low Risk':
                    color = '#fde047'  # Yellow
                    icon = '🟡'
                elif risk_level == 'Moderate Risk':
                    color = '#f59e0b'  # Orange
                    icon = '🟠'
                else:
                    color = '#dc2626'  # Red
                    icon = '🔴'
                
                # Display a custom HTML card for each risk level
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}15 0%, {color}25 100%);
                    border: 1px solid {color}40;
                    border-radius: 12px;
                    padding: 16px;
                    margin: 8px 0;
                    border-left: 4px solid {color};
                ">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <div style="font-size: 14px; font-weight: 600; color: #374151;">{icon} {risk_level}</div>
                            <div style="font-size: 24px; font-weight: 700; color: {color}; margin: 4px 0;">{count:,}</div>
                            <div style="font-size: 12px; color: #6b7280;">{percentage:.1f}% of all scans</div>
                        </div>
                        <div style="font-size: 32px; opacity: 0.3;">{icon}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Row 2: Brain Region Analysis with PERCENTAGE DISPLAY
    if not df_regions.empty:  # Only run if brain region data exists
        st.markdown('<h3 class="section-title">🧠 Brain Region Analysis</h3>', unsafe_allow_html=True)
        
        # Add descriptive text to explain the purpose of the chart
        st.markdown("""
        <p style="font-family: 'Inter', sans-serif; color: #666; margin-bottom: 1.5rem;">
        The chart below shows brain regions that have the greatest impact on dementia classification,
        ranked by their average importance scores displayed as percentages. Higher percentages indicate regions with stronger predictive power.
        </p>
        """, unsafe_allow_html=True)
        
        # Cache the aggregation to avoid recomputing on every run
        @st.cache_data(ttl=60)
        def aggregate_region_stats(df_regions):
            """Cache expensive aggregation operations for speed"""
            return df_regions.groupby('Brain_Region').agg({
                'ScoreCAM_Importance_Score': ['mean', 'std', 'count'],  # Mean, standard deviation, and count
                'ScoreCAM_Importance_Percentage': 'mean'  # Average percentage importance
            }).round(3)
        
        # Aggregate region data using the full dataset
        region_stats = aggregate_region_stats(df_regions)
        region_stats.columns = ['Mean_Score', 'Std_Score', 'Count', 'Mean_Percentage']  # Rename columns
        region_stats = region_stats.reset_index()
        region_stats = region_stats.sort_values('Mean_Score', ascending=False)  # Sort by mean importance score
        
        # Split chart and potential table into two columns (chart gets more space)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if len(region_stats) > 0:
                # Select only the top 15 most important brain regions
                top_regions = region_stats.head(15)
                
                # Sort for horizontal bar chart so most important is on top
                top_regions_sorted = top_regions.sort_values('Mean_Score', ascending=True)
                
                # Convert importance scores to percentage of the total importance
                total_importance = region_stats['Mean_Score'].sum() 
                top_regions_sorted['Mean_Score_Percent'] = (top_regions_sorted['Mean_Score'] / total_importance) * 100
                
                # Create the bar chart (horizontal layout)
                fig_regions = go.Figure(go.Bar(
                    x=top_regions_sorted['Mean_Score_Percent'],  # X-axis: percentage
                    y=top_regions_sorted['Brain_Region'],  # Y-axis: brain region names
                    orientation='h',  # Horizontal bars
                    marker=dict(
                        color=top_regions_sorted['Mean_Score_Percent'],  # Color based on importance %
                        colorscale='Turbo',  # Vibrant color scale
                        showscale=True,  # Show color scale legend
                        line=dict(color='white', width=1.5),  # White border for clarity
                        colorbar=dict(
                            title="<b>Importance<br>Score (%)</b>",  # Colorbar label
                            tickmode="linear",
                            tick0=0,
                            dtick=top_regions_sorted['Mean_Score_Percent'].max() / 5,  # Spacing between ticks
                            bgcolor='rgba(255,255,255,0.8)',
                            borderwidth=1,
                            bordercolor='#ddd',
                            title_font=dict(family='Inter, sans-serif', size=14),
                            tickfont=dict(family='Inter, sans-serif', size=12),
                            ticksuffix='%'
                        )
                    ),
                    # Add % labels outside bars
                    text=top_regions_sorted['Mean_Score_Percent'].apply(lambda x: f'<b>{x:.1f}%</b>'),
                    textposition='outside',
                    textfont=dict(size=13, color='#000000', family='Inter, sans-serif'),
                    # Custom hover info (includes count and raw score)
                    hovertemplate='<b>%{y}</b><br>' +
                        '<b>Overall Percentage:</b> %{x:.1f}%<br>' +
                        '<b>Raw Score:</b> %{customdata[1]:.4f}<br>' +
                        '<b>Samples:</b> %{customdata[0]}<br>' +
                        '<extra></extra>',
                    customdata=list(zip(top_regions_sorted['Count'], top_regions_sorted['Mean_Score']))
                ))

                # Chart layout customization
                fig_regions.update_layout(
                    title={
                        'text': "✨ <b>Brain Region Overall Importance Distribution</b>",  # Chart title
                        'x': 0.5,  # Centered title
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#000000', 'family': 'Inter, sans-serif'}
                    },
                    xaxis_title="<b>Overall Percentage of Total Importance (%)</b>",  # X-axis label
                    xaxis_title_font=dict(size=14, family='Inter, sans-serif'),
                    yaxis={
                        'categoryorder': 'total ascending',  # Bars ordered by importance
                        'tickfont': {'size': 13, 'color': '#000000', 'family': 'Inter, sans-serif'},
                        'linecolor': '#000000',
                        'linewidth': 2,
                        'autorange': True
                    },
                    height=650,
                    font=dict(family="Inter, sans-serif", size=12),
                    showlegend=False,
                    plot_bgcolor='#F5F5F5',
                    paper_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255,255,255,0.8)',
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='#000000',
                        tickfont={'size': 12, 'color': '#000000', 'family': 'Inter, sans-serif'},
                        linecolor='#000000',
                        linewidth=2,
                        range=[0, top_regions_sorted['Mean_Score_Percent'].max() * 1.1],  # Add some padding
                        ticksuffix='%'
                    ),
                    margin=dict(l=100, r=50, t=80, b=100)
                )
                
                # Add interpretation note below chart
                fig_regions.add_annotation(
                    text="<b>Higher percentages indicate brain regions with greater importance in dementia detection</b>",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.17,
                    showarrow=False,
                    font=dict(size=12, color="black", family='Inter, sans-serif'),
                    xanchor='center',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#000000',
                    borderwidth=1,
                    borderpad=5
                )
                
                # Render the chart in Streamlit
                st.plotly_chart(fig_regions, use_container_width=True)

        with col2:
            # Sidebar section showing detailed statistics for each brain region
            st.markdown('<h4 class="subsection-title">📊 Regional Statistics</h4>', unsafe_allow_html=True)

            # Group data by brain region to calculate descriptive statistics
            region_detailed = df_regions.groupby('Brain_Region').agg({
                'ScoreCAM_Importance_Score': ['mean', 'std', 'count', 'min', 'max'],  # Importance score stats
                'Patient_ID': 'nunique'  # Number of unique patients
            }).round(3)

            # Dictionary mapping certain brain regions to importance & explanation text
            regions_info = {
                "Ventricular": {
                    "importance": 0.72,
                    "explanation": "Fluid-filled cavities affecting brain function."
                },
                "Frontal": {
                    "importance": 0.45,
                    "explanation": "Controls reasoning, planning, and movement."
                },
                "Occipital": {
                    "importance": 0.58,
                    "explanation": "Processes visual information."
                },
                "Temporal": {
                    "importance": 0.83,
                    "explanation": "Involved in memory and hearing."
                },
                "Hippocampus": {
                    "importance": 0.91,
                    "explanation": "Key for memory formation and spatial navigation."
                },
                "Parietal": {
                    "importance": 0.39,
                    "explanation": "Processes sensory information and spatial awareness."
                },
            }

            # Rename column levels after aggregation for easier reference
            region_detailed.columns = ['Mean_Score', 'Std_Score', 'Sample_Count', 'Min_Score', 'Max_Score', 'Unique_Patients']
            region_detailed = region_detailed.reset_index()

            # Calculate variability (relative std deviation) per brain region
            region_detailed['Variability'] = region_detailed['Std_Score'] / region_detailed['Mean_Score']

            # Identify top 3 most variable brain regions
            most_variable = region_detailed.nlargest(3, 'Variability')
            st.markdown("**🔄 Most Variable Regions:**")
            for _, region in most_variable.iterrows():
                mean_percent = region['Mean_Score'] * 100  # Convert mean score to %
                st.markdown(f"""
                <div style="
                    background: #fef3c7;
                    border-left: 3px solid #f59e0b;
                    padding: 8px;
                    margin: 4px 0;
                    border-radius: 4px;
                ">
                    <strong>{region['Brain_Region']} — {regions_info.get(region['Brain_Region'], {}).get('explanation', 'No explanation available')}</strong><br>
                    <span style="font-size: 11px; color: #92400e;">
                        Variability: {region['Variability']:.2f} • Mean: {mean_percent:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Identify top 3 most consistent brain regions (lowest variability)
            st.markdown("**🎯 Most Consistent Regions:**")
            most_consistent = region_detailed.nsmallest(3, 'Variability')
            for _, region in most_consistent.iterrows():
                mean_percent = region['Mean_Score'] * 100  # Convert mean score to %
                st.markdown(f"""
                <div style="
                    background: #dcfce7;
                    border-left: 3px solid #16a34a;
                    padding: 8px;
                    margin: 4px 0;
                    border-radius: 4px;
                ">
                    <strong>{region['Brain_Region']} — {regions_info.get(region['Brain_Region'], {}).get('explanation', 'No explanation available')}</strong><br>
                    <span style="font-size: 11px; color: #14532d;">
                        Variability: {region['Variability']:.2f} • Mean: {mean_percent:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Section for recent model predictions and classifications
        st.markdown('<h3 class="section-title">📋 Recent Analysis Activity</h3>', unsafe_allow_html=True)

        if not df_predictions.empty:
            # Dropdown filter to view only a specific class or all predictions
            activity_filter = st.selectbox(
                "Filter by class:",
                ["All"] + sorted(df_predictions['Predicted_Class'].unique().tolist()),
                key="activity_class_filter_tab1"
            )

            # Apply class filter to prediction activity dataframe
            activity_df = df_predictions.copy()
            if activity_filter != "All":
                activity_df = activity_df[activity_df['Predicted_Class'] == activity_filter]

            if not activity_df.empty:
                display_data = []

                # Iterate through the top 15 recent predictions
                for _, row in activity_df.head(15).iterrows():
                    pred_class = str(row['Predicted_Class']).replace('_', ' ')
                    confidence = row['Confidence']

                    # Find top brain region for this scan & convert score to %
                    region_info = "N/A"
                    patient_regions = df_regions[df_regions['Filename'] == row['Filename']]
                    if not patient_regions.empty:
                        top_region = patient_regions.nlargest(1, 'ScoreCAM_Importance_Score').iloc[0]
                        top_score_percentage = top_region['ScoreCAM_Importance_Score'] * 100
                        region_info = f"{top_region['Brain_Region']} ({top_score_percentage:.1f}%)"

                    # Assign risk level and priority based on prediction class
                    if 'Non Demented' in row['Predicted_Class']:
                        status, risk, priority = '🟢', 'Low', 'Normal'
                    elif 'Very Mild Demented' in row['Predicted_Class']:
                        status, risk, priority = '🟡', 'Low-Medium', 'Monitor'
                    elif 'Mild Demented' in row['Predicted_Class']:
                        status, risk, priority = '🟠', 'Medium', 'Follow-up'
                    else:
                        status, risk, priority = '🔴', 'High', 'Urgent'

                    # Assign confidence text color based on threshold
                    if confidence >= 0.9:
                        conf_color = '#16a34a'
                    elif confidence >= 0.7:
                        conf_color = '#eab308'
                    else:
                        conf_color = '#dc2626'

                    # Append row data for table display
                    display_data.append({
                        'Status': status,
                        'Scan ID': row['Filename'][:20] + '...' if len(row['Filename']) > 20 else row['Filename'],
                        'Classification': pred_class,
                        'Confidence': f'<span style="color: {conf_color}; font-weight: bold;">{confidence:.1%}</span>',
                        'Risk Level': risk,
                        'Top Region': region_info,
                        'Action': priority
                    })

                # Convert to DataFrame for HTML rendering
                display_df = pd.DataFrame(display_data)

                # Custom CSS for neat table formatting
                st.markdown("""
                    <style>
                    table {
                        width: 100% !important;
                        table-layout: fixed;
                    }
                    th {
                        text-align: center !important;
                        padding: 10px;
                    }
                    td {
                        padding: 10px;
                        text-align: center;
                        word-wrap: break-word;
                    }
                    </style>
                """, unsafe_allow_html=True)

                # Display recent activity table
                st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

                # Button to export all filtered activity to CSV
                if st.button("📥 Export Full Activity Report", key="export_activity_tab1"):
                    csv = activity_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"mri_activity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
            else:
                # If no data after filtering
                st.info("No activity matches the selected filters.")
        else:
            # If no predictions available at all
            st.info("📊 No recent analysis data available.")

# TAB 2: Individual Analysis 
elif st.session_state.selected_tab == "🧬 Individual Patient Analysis":
    tab2_container = st.container()  # Create a container for all elements in Tab 2

    with tab2_container:
        if not df_predictions.empty:  # Proceed only if predictions dataframe is not empty
            # header
            st.markdown(" ")  # Spacer
            st.markdown("""
                <div style="background: #4e54c8; 
                            padding: 2px; border-radius: 5px; margin-bottom: 5px; 
                            color: white; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
                    <h4 style="margin: 0; color: white;">🧬 Individual Analysis + Clinical Intelligence</h4>
                </div>
                """, unsafe_allow_html=True)  # Custom HTML/CSS styling for section title

            # Patient selector (using session state for persistence)
            col1 = st.columns(2)[0]  # Create two columns and use the first for patient selection
            
            with col1:
                scan_options = []  # List to store (row index, display label) pairs
                filename_col = 'Filename' if 'Filename' in df_predictions.columns else 'filename'  
                # Check column naming variations and choose the correct one

                # Iterate over each prediction row to prepare display-friendly patient names
                for idx, row in df_predictions.iterrows():
                    filename = row[filename_col]  # Get filename from dataframe
                    # Remove file extension if present
                    clean_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
                    
                    # Try to extract patient number using regex (matches "patient_12" or "patient 12")
                    match = re.search(r'patient[_\s]*(\d+)', clean_filename, re.IGNORECASE)
                    if match:
                        patient_display = f"Patient {match.group(1)}"
                    else:
                        # If no patient number, clean up underscores and title-case the string
                        patient_display = clean_filename.replace('_', ' ').title()
                    
                    scan_label = f"{patient_display}"  # Final display label
                    scan_options.append((idx, scan_label))  # Store both index and label
                
                # Initialize session state for selected scan
                if 'tab2_selected_scan' not in st.session_state:
                    # Default to the first scan if available, else set to 0
                    st.session_state.tab2_selected_scan = scan_options[0][0] if scan_options else 0
                
                # Create a select box for scan selection
                selected_scan_idx = st.selectbox(
                    "🔍 Select MRI Scan:",
                    options=[opt[0] for opt in scan_options],  # Use dataframe indices as values
                    format_func=lambda x: next(opt[1] for opt in scan_options if opt[0] == x),  # Map index to label
                    key="tab2_scan_selector_fixed",  # FIXED KEY for Streamlit to persist selection
                    index=[opt[0] for opt in scan_options].index(st.session_state.tab2_selected_scan) 
                          if st.session_state.tab2_selected_scan in [opt[0] for opt in scan_options] else 0,
                    help="Select a scan for comprehensive analysis with clinical intelligence",  # Tooltip
                    # On change, update session state to keep track of the selected scan
                    on_change=lambda: setattr(st.session_state, 'tab2_selected_scan', st.session_state.tab2_scan_selector_fixed)
                )
            
            # Update session state with the newly selected scan index
            st.session_state.tab2_selected_scan = selected_scan_idx

            # Get the selected scan row from dataframe for further analysis
            selected_scan = df_predictions.loc[selected_scan_idx]

        # Main Analysis Container
        st.markdown("---")  # Horizontal separator to visually break sections
        
        # 1: Patient Information + Clinical Assessment
        # Section header styled with HTML to give a clear title for this analysis block
        st.markdown('<h4 class="subsection-title">📋 Patient Diagnostic Information & Clinical Assessment</h4>', unsafe_allow_html=True)

        # Create two columns for layout: left for patient info card, right for other content
        col1, col2 = st.columns(2)

        with col1:
            # Patient Info Card (keeping your original design)
            # Get the filename column name depending on how it is labeled in selected_scan
            filename_col = 'Filename' if 'Filename' in selected_scan.index else 'filename'
            scan_id = selected_scan.get(filename_col, 'Unknown')  # Retrieve the filename or fallback to 'Unknown'
            
            # Helper function to extract a readable patient ID from the filename
            def extract_patient_display_id(filename):
                match = re.search(r'(patient[_\s]*\d+)', filename, re.IGNORECASE)
                return match.group(1).replace('_', ' ').title() if match else filename
            
            display_patient_id = extract_patient_display_id(scan_id)  # Cleaned-up patient ID
            
            pred_class = selected_scan['Predicted_Class']  # Model’s predicted dementia category
            confidence = selected_scan['Confidence']       # Model’s prediction confidence score
            
            # Styling based on prediction category
            if 'Non Demented' in pred_class:
                primary_color = '#10b981'  # Green tone for healthy
                bg_gradient = 'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)'
                status_icon = '✅'
                status_text = 'Healthy Brain'
                
            elif 'Very Mild Demented' in pred_class:
                primary_color = '#eab308'  # Yellow tone for early warning
                bg_gradient = 'linear-gradient(135deg, #fefce8 0%, #fef3c7 100%)'
                status_icon = '⚠️'
                status_text = 'Early Signs Detected'
                
            elif 'Mild Demented' in pred_class:
                primary_color = '#f97316'  # Orange tone for moderate concern
                bg_gradient = 'linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%)'
                status_icon = '🔶'
                status_text = 'Mild Dementia'
                
            else:
                primary_color = '#ef4444'  # Red tone for high concern
                bg_gradient = 'linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)'
                status_icon = '🚨'
                status_text = 'Moderate Dementia'
                
            # Render the custom HTML patient info card with prediction details and confidence
            st.markdown(f"""
            <div style="
                background: {bg_gradient};
                border: 2px solid {primary_color}40;
                border-radius: 12px;
                padding: 12px 16px;
                position: relative;
                overflow: hidden;
                height: auto;
                box-shadow: 0 3px 8px rgba(0,0,0,0.08);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <!-- Decorative background icon for visual emphasis -->
                <div style="
                    position: absolute;
                    top: -8px;
                    right: -8px;
                    font-size: 80px;
                    opacity: 0.1;
                    transform: rotate(15deg);
                    color: {primary_color};
                    pointer-events: none;
                    user-select: none;
                ">{status_icon}</div>
                <!-- Section Title -->
                <h3 style="
                    margin: 0 0 8px 0; 
                    color: #1e293b; 
                    font-weight: 700; 
                    font-size: 1.1rem; 
                    position: relative; 
                    z-index: 1;
                ">
                    🎯 AI Classification Result
                </h3>
                <!-- Prediction Display Row -->
                <div style="
                    display: flex; 
                    align-items: center; 
                    margin-bottom: 12px; 
                    position: relative; 
                    z-index: 1;
                ">
                    <div style="font-size: 36px; margin-right: 12px;">{status_icon}</div>
                    <div>
                        <div style="
                            font-size: 1.1rem; 
                            font-weight: 700; 
                            color: {primary_color}; 
                            margin-bottom: 2px;
                        ">
                            {pred_class.replace('_', ' ')}
                        </div>
                        <div style="
                            font-size: 0.85rem; 
                            color: #6b7280; 
                            font-weight: 500;
                        ">
                            {status_text}
                        </div>
                    </div>
                </div>
                <!-- Risk Level Badge -->
                <div style="
                    background: {primary_color}10;
                    border-radius: 16px;
                    padding: 4px 12px;
                    font-size: 0.8rem;
                    color: {primary_color};
                    font-weight: 600;
                    display: inline-block;
                    margin-bottom: 12px;
                ">
                    <span style="font-weight: 700;">Risk Level:</span>
                    {'Low' if 'Non Demented' in pred_class else 'Low-Medium' if 'Very Mild' in pred_class else 'Medium' if 'Mild' in pred_class else 'High'}
                </div>
                <!-- Confidence Section -->
                <div style="
                    background: rgba(255,255,255,0.9);
                    border-radius: 10px;
                    padding: 6px 10px;
                    margin: 12px 0 0 0;
                    position: relative;
                    z-index: 1;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
                ">
                    <!-- Confidence Label and Category -->
                    <div style="
                        display: flex; 
                        justify-content: space-between; 
                        align-items: center; 
                        margin-bottom: 6px;
                        font-size: 0.8rem; 
                        color: #6b7280; 
                        font-weight: 500;
                    ">
                        <div>Model Confidence</div>
                        <div style="
                            background: {primary_color}20;
                            border-radius: 16px;
                            padding: 2px 8px;
                            border: 1px solid {primary_color}40;
                            font-size: 0.75rem;
                            color: {primary_color};
                            font-weight: 600;
                            white-space: nowrap;
                        ">
                            {'HIGH' if confidence >= 0.8 else 'MEDIUM' if confidence >= 0.5 else 'LOW'}
                        </div>
                    </div>
                    <!-- Confidence Percentage -->
                    <div style="
                        font-size: 1.5rem; 
                        font-weight: 700; 
                        color: {primary_color}; 
                        margin-bottom: 6px;
                        line-height: 1;
                    ">
                        {confidence:.1%}
                    </div>
                    <!-- Confidence Progress Bar -->
                    <div style="
                        background: #e5e7eb; 
                        border-radius: 4px; 
                        height: 5px; 
                        overflow: hidden;
                    ">
                        <div style="
                            background: {primary_color}; 
                            height: 100%; 
                            width: {confidence*100}%; 
                            border-radius: 4px; 
                            transition: width 0.7s ease;
                        "></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)  # Allow HTML/CSS rendering in Streamlit

        with col2:
            # CLINICAL ASSESSMENT SECTION
            if CLINICAL_FEATURES_AVAILABLE:
                st.markdown("**📋 Clinical Assessment**")  # Section heading for clinical assessment
                patient_regions = df_regions[df_regions[filename_col] == scan_id]  # Filter brain region data for the current scan
                
                try:
                    # Generate a narrative summary of the patient's scan using both clinical and imaging data
                    narrative = generate_patient_narrative(
                        patient_data=selected_scan,
                        region_data=patient_regions,
                        analysis_type="individual"
                    )
                    
                    # Retrieve predicted class and determine UI styling based on diagnosis severity
                    pred_class = selected_scan['Predicted_Class']
                    if "Non Demented" in pred_class:
                        status_color = "#10b981"  # Green for healthy
                        severity_level = "Normal"
                        severity_icon = "✅"
                        risk_level = "Low"
                    elif "Very Mild" in pred_class:
                        status_color = "#f59e0b"  # Amber for early warning
                        severity_level = "Very Mild"
                        severity_icon = "⚠️"
                        risk_level = "Low-Medium"
                    elif "Mild" in pred_class:
                        status_color = "#f97316"  # Orange for moderate concern
                        severity_level = "Mild"
                        severity_icon = "🔶"
                        risk_level = "Medium"
                    else:
                        status_color = "#ef4444"  # Red for serious concern
                        severity_level = "Moderate"
                        severity_icon = "🚨"
                        risk_level = "High"

                    # Display clinical narrative inside a styled info box
                    # 🧠 Display clinical narrative inside a styled info box
                    clean_narrative = (
                        narrative.replace('<br>', '\n')
                                .replace('<p>', '')
                                .replace('</p>', '')
                                .replace('\\n\\n', '\n')
                                .strip()
                    )

                    st.markdown(f"""
                    <div style="
                        background: {status_color}10;
                        border-left: 4px solid {status_color};
                        padding: 12px 16px;
                        border-radius: 8px;
                        font-weight: 500;
                        line-height: 1.6;
                        color: #374151;
                        white-space: pre-wrap;
                    ">
                        {clean_narrative}
                    </div>
                    """, unsafe_allow_html=True)

                            
                except Exception as e:
                    # Display a styled error box if narrative generation fails
                    st.markdown(f"""
                    <div style="
                        background: #fef2f2;
                        color: #b91c1c;
                        border-left: 4px solid #ef4444;
                        padding: 12px;
                        border-radius: 8px;
                        font-weight: 500;
                    ">
                        ❗ Error generating clinical narrative: {e}
                    </div>
                    """, unsafe_allow_html=True)

            else:
                # When no clinical features are available, show a basic summary card instead
                st.markdown("**📋 Patient Overview**")
                st.markdown(f"""
                <div style="
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 12px;
                ">
                    <div><strong>Scan ID:</strong> {scan_id}</div>
                    <div><strong>Classification:</strong> {pred_class}</div>
                    <div><strong>Confidence:</strong> {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        # Heading for the probability distribution chart
        st.markdown('<h4 class="subsection-title">📊 Probability Distribution Analysis</h4>', unsafe_allow_html=True)

        # Columns that contain probability scores for each possible classification
        prob_columns = ['Mild_Demented_Probability', 'Moderate_Demented_Probability', 
                        'Non_Demented_Probability', 'Very_Mild_Demented_Probability']

        # Gather available probability data for the selected scan
        prob_data = []
        for col in prob_columns:
            if col in selected_scan and pd.notna(selected_scan[col]):
                class_name = col.replace('_Probability', '').replace('_', ' ')  # Clean up column name
                prob_value = float(selected_scan[col])
                prob_data.append({
                    'Class': class_name,
                    'Probability': prob_value
                })

        if prob_data:
            # Create DataFrame and sort probabilities from lowest to highest
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('Probability', ascending=True)

            # Color, icon, and description configuration for each diagnosis type
            color_config = {
                'Non Demented': {
                    'primary': '#22c55e',
                    'secondary': '#16a34a', 
                    'light': '#dcfce7',
                    'icon': '💚',
                    'emoji': '😊',
                    'risk_level': 'Healthy',
                    'interpretation': 'Normal cognitive function',
                    'description': 'No signs of cognitive decline'
                },
                'Very Mild Demented': {
                    'primary': '#fbbf24',
                    'secondary': '#f59e0b',
                    'light': '#fef3c7',
                    'icon': '💛',
                    'emoji': '😐',
                    'risk_level': 'Very Mild',
                    'interpretation': 'Minimal cognitive changes',
                    'description': 'Slight memory concerns'
                },
                'Mild Demented': {
                    'primary': '#fb923c',
                    'secondary': '#ea580c',
                    'light': '#fed7aa',
                    'icon': '🧡',
                    'emoji': '😕',
                    'risk_level': 'Mild',
                    'interpretation': 'Noticeable cognitive decline',
                    'description': 'Memory and thinking affected'
                },
                'Moderate Demented': {
                    'primary': '#ef4444',
                    'secondary': '#dc2626',
                    'light': '#fecaca',
                    'icon': '❤️',
                    'emoji': '😰',
                    'risk_level': 'Moderate',
                    'interpretation': 'Significant cognitive impairment',
                    'description': 'Daily activities may be affected'
                }
            }

            # Identify the class with the highest probability (predicted class)
            max_prob_class = prob_df.loc[prob_df['Probability'].idxmax(), 'Class']
            max_prob_value = prob_df['Probability'].max()
            
            fig_prob = go.Figure()

            # Create horizontal probability bars
            for idx, row in prob_df.iterrows():
                class_name = row['Class']
                probability = row['Probability']
                config = color_config.get(class_name, {})
                
                # Highlight predicted class with bolder styling
                is_predicted = class_name == max_prob_class
                bar_opacity = 1.0 if is_predicted else 0.75
                line_width = 4 if is_predicted else 2
                
                # Add bar to chart with gradient effect and detailed hover info
                fig_prob.add_trace(go.Bar(
                    y=[class_name],
                    x=[probability],
                    orientation='h',
                    name=class_name,
                    marker=dict(
                        color=config.get('primary', '#6b7280'),
                        opacity=bar_opacity,
                        line=dict(
                            color='white',
                            width=line_width
                        ),
                        pattern=dict(  # Apply pattern only to predicted class
                            shape="",
                            bgcolor=config.get('primary', '#6b7280'),
                            fgcolor=config.get('secondary', '#6b7280'),
                            size=8,
                            solidity=0.3
                        ) if is_predicted else None
                    ),
                    text=f'<b>{config.get("emoji", "")} {probability:.1%}</b>',  # Percentage with emoji
                    textposition='inside',
                    textfont=dict(
                        size=20 if is_predicted else 17, 
                        color='white', 
                        family='SF Pro Display, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif'
                    ),
                    hovertemplate=f'<b>{config.get("emoji", "")} %{{y}}</b><br>' +
                                f'<b>Probability:</b> %{{x:.2%}}<br>' +
                                f'<b>Risk Level:</b> {config.get("risk_level", "Unknown")}<br>' +
                                f'<b>Clinical Meaning:</b> {config.get("description", "")}<br>' +
                                f'<b>Interpretation:</b> {config.get("interpretation", "")}<br>' +
                                '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor=config.get('primary', '#6b7280'),
                        bordercolor="white",
                        font=dict(color="white", size=15, family='SF Pro Display, sans-serif')
                    )
                ))
            
            # Stunning modern layout for the probability distribution chart
            fig_prob.update_layout(
                xaxis=dict(
                    title=dict(
                        # X-axis title with custom font and styling
                        text='<b style="font-size:18px; color:#374151;">Confidence Score</b>',
                        font=dict(size=18, family='SF Pro Display, sans-serif')
                    ),
                    tickformat='.0%',                # Format X-axis ticks as percentages
                    range=[0, 1.1],                   # Show scale from 0% to slightly over 100% for padding
                    gridcolor='rgba(148, 163, 184, 0.15)',  # Light grey grid lines
                    gridwidth=1.5,
                    showgrid=True,
                    zeroline=False,                   # Remove the zero line
                    tickfont=dict(size=16, family='SF Pro Display, sans-serif', color='#64748b'),
                    linecolor='#e2e8f0', linewidth=3, # Border color and width for axis
                    showspikes=True,                  # Show vertical guideline when hovering
                    spikecolor='rgba(59, 130, 246, 0.3)',
                    spikethickness=2,
                    spikedash='dot'
                ),
                
                yaxis=dict(
                    title='',                          # No title for Y-axis
                    showgrid=False,                    # Remove horizontal grid lines
                    autorange='reversed',              # Reverse order so highest probability is at top
                    tickfont=dict(size=18, family='SF Pro Display, sans-serif', color='#1a202c'),
                    showline=False,
                    categoryorder='array',             # Keep order as in `prob_df`
                    categoryarray=prob_df['Class'].tolist()
                ),
                
                # Chart background colors for clean UI
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                paper_bgcolor='#f8fafc',
                
                height=500,                            # Chart height
                margin=dict(l=30, r=120, t=40, b=120), # Space around chart
                font=dict(family="SF Pro Display, -apple-system, sans-serif", size=16),
                
                showlegend=False,                      # No legend (redundant info already in bars)
                
                # Decorative outer border with transparency
                shapes=[
                    dict(
                        type="rect",
                        xref="paper", yref="paper",
                        x0=-0.02, y0=-0.02, x1=1.02, y1=1.02,
                        line=dict(color="rgba(255, 255, 255, 0.2)", width=0),
                        fillcolor="rgba(255, 255, 255, 0.05)",
                        layer="below"
                    )
                ]
            )

            # confidence zones for visual interpretation
            confidence_zones = [
                {'range': [0, 0.3],   'color': 'rgba(239, 68, 68, 0.08)',  'label': '🔴 Low Confidence',       'pos': 0.15},
                {'range': [0.3, 0.6], 'color': 'rgba(251, 191, 36, 0.08)', 'label': '🟡 Moderate Confidence', 'pos': 0.45},
                {'range': [0.6, 0.8], 'color': 'rgba(34, 197, 94, 0.08)',  'label': '🟢 High Confidence',     'pos': 0.7},
                {'range': [0.8, 1],   'color': 'rgba(59, 130, 246, 0.08)', 'label': '🔵 Very High Confidence','pos': 0.9}
            ]

            for zone in confidence_zones:
                # Highlight background range for each confidence category
                fig_prob.add_vrect(
                    x0=zone['range'][0], x1=zone['range'][1],
                    fillcolor=zone['color'],
                    layer="below",
                    line_width=0
                )
                
                # Add label above the chart to indicate zone meaning
                fig_prob.add_annotation(
                    x=zone['pos'],
                    y=1.02,                              # Position just above chart
                    xref="x", yref="paper",
                    text=f"<b style='font-size:13px;'>{zone['label']}</b>",
                    showarrow=False,
                    font=dict(size=13, color="#4a5568", family="SF Pro Display, sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="rgba(203, 213, 225, 0.6)",
                    borderwidth=1.5,
                    borderpad=6
                )

            # Highlight the predicted class with an arrow and label
            predicted_config = color_config.get(max_prob_class, {})
            fig_prob.add_annotation(
                x=min(max_prob_value + 0.12, 1.05),  # Position arrow near the end of the bar
                y=max_prob_class,
                text=f'<b style="font-size:14px;">🎯 PREDICTED</b><br>'
                    f'<b style="font-size:13px; color:{predicted_config.get("primary", "#6b7280")};">'
                    f'{max_prob_value:.1%}</b>',
                showarrow=True,
                arrowhead=3,
                arrowsize=1.8,
                arrowwidth=4,
                arrowcolor=predicted_config.get('primary', '#6b7280'),
                font=dict(
                    size=14, 
                    color=predicted_config.get('primary', '#6b7280'),
                    family="SF Pro Display, sans-serif"
                ),
                bgcolor="rgba(255, 255, 255, 0.98)",
                bordercolor=predicted_config.get('primary', '#6b7280'),
                borderwidth=2.5,
                borderpad=12
            )

            # Determine qualitative confidence level text
            confidence_level = (
                "Excellent" if max_prob_value >= 0.9 else
                "High" if max_prob_value >= 0.75 else
                "Good" if max_prob_value >= 0.6 else
                "Moderate" if max_prob_value >= 0.45 else
                "Low"
            )

            # Match confidence level with corresponding color code
            confidence_color = (
                "#10b981" if max_prob_value >= 0.75 else
                "#fbbf24" if max_prob_value >= 0.6 else
                "#fb923c" if max_prob_value >= 0.45 else
                "#ef4444"
            )

            # Calculate entropy to evaluate how clear the probability distribution is
            entropy = -sum([p * np.log2(p) if p > 0 else 0 for p in prob_df['Probability']])
            distribution_clarity = (
                "Very Clear" if entropy < 0.5 else
                "Clear" if entropy < 1.0 else
                "Moderate" if entropy < 1.5 else
                "Uncertain"
            )

            # -----------------------------
            # SUMMARY CARD
            # -----------------------------
            fig_prob.add_annotation(
                x=0.5, y=-0.35,                     # Centered below the chart
                xref="paper", yref="paper",
                text=f'<b style="font-size:16px; color:#1a202c;">Model Confidence: '
                    f'<span style="color:{confidence_color};">{confidence_level}</span></b><br>'
                    f'<span style="font-size:14px; color:#64748b;">Most likely: '
                    f'<b>{max_prob_class}</b> at {max_prob_value:.1%} confidence</span><br>',
                showarrow=False,
                font=dict(family="SF Pro Display, sans-serif"),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="rgba(203, 213, 225, 0.5)",
                borderwidth=2,
                borderpad=5,
                xanchor="center"
            )

            # -----------------------------
            # INTERACTIVITY & ANIMATION
            # -----------------------------
            fig_prob.update_traces(
                marker_line_color='white',   # White borders for bars
                opacity=0.95                 # Slight transparency for soft look
            )

            fig_prob.update_layout(
                hovermode='y unified',       # Hover highlights all bars at that Y-level
                transition_duration=500      # Smooth animated transitions
            )

            # Render the probability distribution chart
            st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})

            # -----------------------------
            # IMAGE MATCHING LOGIC
            # -----------------------------
            scan_filename = selected_scan.get('Filename', selected_scan.get('filename', ''))

            st.markdown("---")
            st.markdown('<h4 class="subsection-title">🖼️ MRI Scan Visualizations</h4>', unsafe_allow_html=True)

            patient_id = extract_patient_id(scan_filename)

            # Initialize dataframe for matched images
            scan_images = pd.DataFrame()

            if not df_images.empty:
                # Strategy 1: Direct match on patient ID
                if patient_id:
                    scan_images = df_images[df_images['patient_id'].astype(str) == str(patient_id)]
                
                # Strategy 2: Match using base filename
                if scan_images.empty:
                    scan_base = scan_filename.split('.')[0] if '.' in scan_filename else scan_filename
                    scan_images = df_images[df_images['filename'].str.contains(scan_base, case=False, na=False)]
                
                # Strategy 3: Match any part of patient ID in image patient_id
                if scan_images.empty and patient_id:
                    scan_images = df_images[df_images['patient_id'].astype(str).str.contains(str(patient_id), case=False, na=False)]
                
                # Strategy 4: Manual fallback
                if scan_images.empty:
                    st.warning("⚠️ Could not automatically match images to this scan.")

                # -----------------------------
                # ORGANIZE IMAGES BY TYPE
                # -----------------------------
                if not scan_images.empty:
                    image_dict = {}
                    for _, img_row in scan_images.iterrows():
                        image_type = img_row['image_type']
                        file_path = img_row['file_path']
                        
                        # Map known types to friendly display names
                        if 'original' in image_type.lower():
                            image_dict['Original'] = file_path
                        elif 'heatmap' in image_type.lower():
                            image_dict['Heatmap'] = file_path
                        elif 'scorecam' in image_type.lower() or 'score_cam' in image_type.lower():
                            image_dict['Scorecam Overlay'] = file_path
                        elif 'lime' in image_type.lower():
                            image_dict['LIME'] = file_path
                        elif 'brain_mask' in image_type.lower():
                            image_dict['Brain Mask'] = file_path
                        elif 'region' in image_type.lower():
                            # Extract region name for display
                            region_name = image_type.replace('_region_', ' - ').replace('region_', '').replace('_', ' ').title()
                            image_dict[f'Region: {region_name}'] = file_path

                    # -----------------------------
                    # DISPLAY IMAGES USING TABS
                    # -----------------------------
                    if image_dict:
                        tab_names = []
                        tab_keys = []

                        # Add tabs based on available images
                        if 'Original' in image_dict:
                            tab_names.append("🧠 Original")
                            tab_keys.append('Original')
                        if 'Heatmap' in image_dict:
                            tab_names.append("🔥 Heatmap")
                            tab_keys.append('Heatmap')
                        
                        # Group region-based images
                        region_tabs = [(name, key) for key, name in image_dict.items() if 'Region:' in key]
                        if region_tabs:
                            tab_names.append("🎯 Brain Regions")
                            tab_keys.append('regions')

                        # Create Streamlit tabs
                        if tab_names:
                            tabs = st.tabs(tab_names)
                            tab_idx = 0

                            # ---- Original MRI Tab ----
                            if 'Original' in image_dict:
                                with tabs[tab_idx]:
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        display_image_from_db(image_dict['Original'], "Original MRI Scan")
                                tab_idx += 1

                            # ---- Heatmap & Scorecam Tab ----
                            if 'Heatmap' in image_dict:
                                with tabs[tab_idx]:
                                    if 'Scorecam Overlay' in image_dict:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            # Heatmap visualization
                                            display_image_from_db(image_dict['Heatmap'], "Attention Heatmap")
                                            # Heatmap legend
                                            st.markdown("""
                                            <div style="
                                                background: #f9fafb;
                                                border-radius: 8px;
                                                padding: 16px;
                                                margin-top: 16px;
                                                border: 1px solid #e5e7eb;
                                            ">
                                                <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 8px;">
                                                    🎨 Heatmap Color Scale
                                                </div>
                                                <div style="
                                                    background: linear-gradient(to right, #3b82f6, #10b981, #eab308, #f97316, #ef4444);
                                                    height: 20px;
                                                    border-radius: 4px;
                                                    margin: 8px 0;
                                                "></div>
                                                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6b7280;">
                                                    <span>Low Attention</span>
                                                    <span>High Attention</span>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        with col2:
                                            # Scorecam overlay visualization
                                            display_image_from_db(image_dict['Scorecam Overlay'], "Scorecam Overlay")
                                            # Scorecam explanation
                                            st.markdown("""
                                            <div style="
                                                background: #fef3c7;
                                                border-radius: 8px;
                                                padding: 16px;
                                                margin-top: 16px;
                                                border: 1px solid #fbbf24;
                                                border-left: 3px solid #f59e0b;
                                            ">
                                                <div style="font-size: 12px; font-weight: 600; color: #92400e; margin-bottom: 4px;">
                                                    📍 About Scorecam Overlay
                                                </div>
                                                <div style="font-size: 11px; color: #78350f;">
                                                    Scorecam overlays highlight the most important regions for the model's 
                                                    prediction by masking different areas and measuring impact on confidence.
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                tab_idx += 1
                            # Brain Regions Tab
                            if region_tabs:
                                with tabs[tab_idx]:
                                    # Section header for brain region analysis
                                    st.markdown("""
                                    <div style="margin-bottom: 16px;">
                                        <h5 style="color: #111827; font-weight: 600;">🎯 Brain Region Analysis</h5>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Prepare list of (region_name, file_path) pairs for all detected brain regions
                                    region_images = [(name.replace('Region: ', ''), path) for name, path in image_dict.items() if 'Region:' in name]
                                    
                                    if len(region_images) == 1:
                                        # If there is only one brain region image, center it on the page
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            region_name, region_path = region_images[0]
                                            display_image_from_db(region_path, f"{region_name} Region")
                                    else:
                                        # If multiple brain region images exist, show them in a 2-column grid
                                        cols = st.columns(2)
                                        for idx, (region_name, region_path) in enumerate(region_images):
                                            with cols[idx % 2]:
                                                display_image_from_db(region_path, f"{region_name} Region")

                            # Brain region analysis with CLINICAL ENHANCEMENT 3
                            if not df_regions.empty:
                                # Select correct filename column name (different datasets might use different casing)
                                filename_col_regions = 'Filename' if 'Filename' in df_regions.columns else 'filename'
                                
                                # Filter brain region data for the selected scan
                                scan_regions = df_regions[df_regions[filename_col_regions] == scan_id]
                                
                                if not scan_regions.empty:
                                    st.markdown("---")  # Separator line
                                    st.markdown('<h4 class="subsection-title">🧠 Brain Region Analysis + Clinical Insights</h4>', unsafe_allow_html=True)
                                    
                                    # Keep only top 10 regions based on ScoreCAM importance
                                    top_regions = scan_regions.nlargest(10, 'ScoreCAM_Importance_Score')
                                    
                                    # Calculate total and percentage importance for each region
                                    total_importance = top_regions['ScoreCAM_Importance_Score'].sum()
                                    top_regions = top_regions.copy()
                                    top_regions['ScoreCAM_Importance_Percentage'] = (top_regions['ScoreCAM_Importance_Score'] / total_importance) * 100
                                    
                                    # Define color, icon, and description for each anatomical region category
                                    region_categories = {
                                        'Frontal': {'icon': '🧩', 'color': '#3b82f6', 'description': 'Executive function & decision making'},
                                        'Parietal': {'icon': '🎯', 'color': '#8b5cf6', 'description': 'Spatial processing & attention'},
                                        'Temporal': {'icon': '💭', 'color': '#10b981', 'description': 'Memory & language processing'},
                                        'Occipital': {'icon': '👁️', 'color': '#f59e0b', 'description': 'Visual processing center'},
                                        'Hippocampus': {'icon': '🧠', 'color': '#ef4444', 'description': 'Memory formation & learning'},
                                        'Ventricular': {'icon': '💧', 'color': '#06b6d4', 'description': 'Cerebrospinal fluid spaces'},
                                        'Cerebellum': {'icon': '⚖️', 'color': '#84cc16', 'description': 'Balance & motor coordination'},
                                        'Brainstem': {'icon': '🌿', 'color': '#f97316', 'description': 'Vital functions control'}
                                    }
                                    
                                    def get_region_info(region_name):
                                        """Return the category info dict for the given brain region name."""
                                        for category, info in region_categories.items():
                                            if category.lower() in region_name.lower():
                                                return info
                                        # Default styling if region is not found in categories
                                        return {'icon': '🔵', 'color': '#6b7280', 'description': 'Brain region analysis'}
                                    
                                    def hex_to_rgba(hex_color, alpha=1.0):
                                        """Convert HEX color to RGBA format with adjustable transparency."""
                                        hex_color = hex_color.lstrip('#')
                                        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
                                    
                                    # Create the interactive horizontal bar chart
                                    fig_regions = go.Figure()
                                    
                                    max_percentage = top_regions['ScoreCAM_Importance_Percentage'].max()
                                    
                                    for idx, row in top_regions.iterrows():
                                        region_name = row['Brain_Region']
                                        importance = row['ScoreCAM_Importance_Score']
                                        percentage = row['ScoreCAM_Importance_Percentage']
                                        region_info = get_region_info(region_name)
                                        
                                        # Highlight the most important region visually
                                        is_highest = percentage == max_percentage
                                        bar_opacity = 0.95 if is_highest else 0.8
                                        line_width = 3 if is_highest else 2
                                        
                                        # Adjust intensity based on percentage importance
                                        intensity = percentage / max_percentage if max_percentage > 0 else 0
                                        base_color = region_info['color']
                                        bar_color = hex_to_rgba(base_color, 0.3 + 0.7 * intensity)
                                        border_color = hex_to_rgba(base_color, 0.9)
                                        
                                        # Add each region as a horizontal bar
                                        fig_regions.add_trace(go.Bar(
                                            x=[percentage],
                                            y=[region_name],
                                            orientation='h',
                                            name=region_name,
                                            marker=dict(
                                                color=bar_color,
                                                line=dict(color=border_color, width=line_width),
                                                opacity=bar_opacity
                                            ),
                                            text=f'<b>{region_info["icon"]} {percentage:.1f}%</b>',
                                            textposition='outside',
                                            textfont=dict(
                                                size=14 if is_highest else 12,
                                                color='#1f2937',
                                                family='Inter, sans-serif'
                                            ),
                                            hovertemplate=f'<b>{region_info["icon"]} %{{y}}</b><br>' +
                                                        f'<b>Importance:</b> %{{x:.1f}}%<br>' +
                                                        f'<b>Raw Score:</b> {importance:.4f}<br>' +
                                                        f'<b>Function:</b> {region_info["description"]}<br>' +
                                                        f'<b>Impact Level:</b> {"Critical" if percentage > 20 else "High" if percentage > 15 else "Moderate" if percentage > 10 else "Low"}<br>' +
                                                        '<extra></extra>',
                                            hoverlabel=dict(
                                                bgcolor=base_color,
                                                bordercolor="white",
                                                font=dict(color="white", size=13)
                                            ),
                                            showlegend=False
                                        ))
                                    
                                    # Configure layout and styling of the chart
                                    fig_regions.update_layout(
                                        title=dict(
                                            text='<b>🎯 Brain Region Importance Distribution</b><br>' +
                                                '<span style="font-size:13px; color:#64748b;">AI attention focus across anatomical regions (%)</span>',
                                            font=dict(size=20, family='Inter, sans-serif', color='#1e293b'),
                                            x=0.02,
                                            xanchor='left',
                                            pad=dict(b=20)
                                        ),
                                        xaxis=dict(
                                            title=dict(
                                                text='<b>Importance Percentage (%)</b>',
                                                font=dict(size=14, family='Inter, sans-serif', color='#374151')
                                            ),
                                            range=[0, max(top_regions['ScoreCAM_Importance_Percentage']) * 1.2],
                                            tickformat='.1f',
                                            ticksuffix='%',
                                            gridcolor='rgba(148, 163, 184, 0.2)',
                                            gridwidth=1,
                                            showgrid=True,
                                            zeroline=False,
                                            tickfont=dict(size=12, family='Inter, sans-serif', color='#64748b'),
                                            linecolor='#e2e8f0',
                                            linewidth=2
                                        ),
                                        yaxis=dict(
                                            title='',
                                            showgrid=False,
                                            autorange='reversed',
                                            tickfont=dict(size=13, family='Inter, sans-serif', color='#374151'),
                                            showline=False
                                        ),
                                        plot_bgcolor='rgba(248, 250, 252, 0.3)',
                                        paper_bgcolor='white',
                                        height=500,
                                        margin=dict(l=20, r=120, t=80, b=40),
                                        font=dict(family="Inter, sans-serif", size=12),
                                        shapes=[
                                            dict(
                                                type="rect",
                                                xref="paper", yref="paper",
                                                x0=0, y0=0, x1=1, y1=1,
                                                line=dict(color="rgba(226, 232, 240, 0.6)", width=1),
                                                fillcolor="rgba(0,0,0,0)"
                                            )
                                        ]
                                    )
                                    
                                    # Render the plotly chart in Streamlit
                                    st.plotly_chart(fig_regions, use_container_width=True, key="region_chart", config={'displayModeBar': False})

                                    # 4: Interactive Region Explanations
                                    if CLINICAL_FEATURES_AVAILABLE:  # Check if clinical features data is available for analysis
                                        st.markdown("---")  # Add a horizontal separator line in the UI
                                        st.markdown('<h4 class="subsection-title">💡 Clinical Analysis for Region</h4>', unsafe_allow_html=True)  # Section title

                                        # Dropdown or selection widget for choosing a brain region from available scan data
                                        selected_region = create_interactive_region_selector(scan_regions)

                                        if selected_region:  # Only proceed if a region has been selected
                                            col1, col2 = st.columns([1, 1])  # Split layout into two equal columns

                                            with col1:
                                                # Retrieve the importance score for the selected brain region from the DataFrame
                                                region_score = scan_regions[scan_regions['Brain_Region'] == selected_region]['ScoreCAM_Importance_Score'].iloc[0]

                                                # Generate a textual clinical explanation for the selected brain region
                                                explanation = generate_region_explanation(
                                                    region_name=selected_region,
                                                    importance_score=region_score,
                                                    predicted_class=selected_scan['Predicted_Class'],  # Predicted Alzheimer’s risk category
                                                    confidence=selected_scan['Confidence']  # Model prediction confidence score
                                                )

                                                # Render the explanation in a styled box with gradient background
                                               # Step 1: Clean the explanation outside the f-string
                                                clean_explanation = (
                                                    explanation.replace('<br>', '\n')
                                                            .replace('<p>', '')
                                                            .replace('</p>', '')
                                                            .replace('\\n\\n', '\n')  # handle literal \n\n
                                                            .strip()
                                                )

                                                # Step 2: Use the cleaned string in the f-string safely
                                                st.markdown(f"""
                                                <div style="
                                                    background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
                                                    border-radius: 15px;
                                                    padding: 1.5rem;
                                                    border-left: 4px solid #3b82f6;
                                                ">
                                                    <h5 style="color: #1e40af; margin-bottom: 1rem;">
                                                        📍 {selected_region} Clinical Analysis
                                                    </h5>
                                                    <div style="color: #475569; line-height: 1.6; white-space: pre-wrap;">
                                                        {clean_explanation}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)


                                            with col2:
                                                # Retrieve additional clinical insights related to the selected brain region
                                                insights = get_clinical_insights(selected_region, scan_regions)

                                                # Loop through each insight and display it in a styled colored box
                                                for insight in insights:
                                                    st.markdown(f"""
                                                    <div style="background: {insight['color']}10; border-radius: 10px; 
                                                                padding: 1rem; margin-bottom: 0.5rem; border-left: 3px solid {insight['color']};">
                                                        <strong style="color: {insight['color']};">{insight['title']}</strong><br>
                                                        <span style="color: #64748b; font-size: 0.9rem;">{insight['text']}</span>
                                                    </div>
                                                    """, unsafe_allow_html=True)

            else:
                st.info("🖼️ No visualization images available for this scan.")  # Message when no region visualizations exist
        else:
            st.info("📊 No scan data available for individual analysis.")  # Message when no scan data exists

elif st.session_state.selected_tab == "⚖️ Comparative Patient Insights":
    # Create a container to hold all Tab 3 (Comparative Insights) content
    tab3_container = st.container()

    with tab3_container:
        # Show an info message if there are fewer than 2 patients in the dataset
        if len(df_predictions) < 2:
            st.info("📊 Need at least 2 patients for comparative analysis.")
        else:
            # Section header for patient selection
            st.markdown(" ")
            st.markdown("""
                <div style="background: #4e54c8; 
                            padding: 2px; border-radius: 5px; margin-bottom: 5px; 
                            color: white; box-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
                    <h4 style="margin: 0; color: white;"> 🎯 Choose Two Patients for Comparison</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Initialize session state variables for primary patient, comparison patient, and tab view if not already set
            if 'tab3_primary_patient' not in st.session_state:
                st.session_state.tab3_primary_patient = 0
            if 'tab3_comparison_patient' not in st.session_state:
                st.session_state.tab3_comparison_patient = 1 if len(df_predictions) > 1 else 0
            if 'tab3_brain_image_tab' not in st.session_state:
                st.session_state.tab3_brain_image_tab = "🧠 Original Scans"
                
            # Create two columns for side-by-side patient selection
            col1, col2 = st.columns(2)

            # Determine correct filename column name (case-insensitive match)
            filename_col = 'Filename' if 'Filename' in df_predictions.columns else 'filename'

            # Prepare patient display options
            patient_options = []
            for idx, row in df_predictions.iterrows():
                filename = row[filename_col]
                # Remove file extension if present
                clean_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
                # Try to extract patient number from filename (e.g., "patient_12")
                match = re.search(r'patient[_\s]*(\d+)', clean_filename, re.IGNORECASE)
                if match:
                    patient_display = f"Patient {match.group(1)}"
                else:
                    # Fallback: prettify filename
                    patient_display = clean_filename.replace('_', ' ').title()

                patient_label = f"{patient_display}"
                patient_options.append((idx, patient_label))
            
            # Primary patient selection (reference patient)
            with col1:
                primary_patient_idx = st.selectbox(
                    "🎯 Primary Patient (Reference)",
                    options=[opt[0] for opt in patient_options],  # List of patient indices
                    format_func=lambda x: next(opt[1] for opt in patient_options if opt[0] == x),  # Display label instead of index
                    key="tab3_primary_fixed",  # Fixed key for Streamlit state handling
                    index=st.session_state.tab3_primary_patient,  # Default to saved session state value
                    on_change=lambda: setattr(st.session_state, 'tab3_primary_patient', st.session_state.tab3_primary_fixed)
                )
                
            # Comparison patient selection (cannot be the same as primary)
            with col2:
                # Exclude the primary patient from the comparison list
                comparison_options = [opt for opt in patient_options if opt[0] != primary_patient_idx]
                
                comparison_patient_idx = st.selectbox(
                    "🔍 Patient to Compare",
                    options=[opt[0] for opt in comparison_options],
                    format_func=lambda x: next(opt[1] for opt in comparison_options if opt[0] == x),
                    key="tab3_comparison_fixed",  # Fixed key for Streamlit state handling
                    index=0,  # Always default to first available option
                    on_change=lambda: setattr(st.session_state, 'tab3_comparison_patient', st.session_state.tab3_comparison_fixed)
                )
            
            # Update the session state with the current selection
            st.session_state.tab3_primary_patient = primary_patient_idx
            st.session_state.tab3_comparison_patient = comparison_patient_idx

            # Cache comparison data
            @st.cache_data(ttl=60)  # Caches the result of the function for 60 seconds to reduce recomputation
            def prepare_comparison_data_cached(primary_idx, comparison_idx):
                # Retrieve primary patient data from prediction DataFrame based on index
                primary_patient = df_predictions.iloc[primary_idx]
                # Retrieve comparison patient data from prediction DataFrame based on index
                comparison_patient = df_predictions.iloc[comparison_idx]
                
                # Filter MRI region data for the primary patient
                primary_regions = df_regions[df_regions[filename_col] == primary_patient[filename_col]]
                # Filter MRI region data for the comparison patient
                comparison_regions = df_regions[df_regions[filename_col] == comparison_patient[filename_col]]
                
                # Return both patient data and their corresponding MRI regions
                return primary_patient, comparison_patient, primary_regions, comparison_regions

            # Prepare patient and region data for the given indices using cached results
            primary_patient, comparison_patient, primary_regions, comparison_regions = prepare_comparison_data_cached(
                primary_patient_idx, comparison_patient_idx
            )

            def create_patient_card(patient_data, patient_label, is_primary=False):
                """Create a styled patient information card with risk level"""
                
                # ----------------------------
                # Extract prediction class and confidence
                # ----------------------------
                pred_class = patient_data['Predicted_Class']
                confidence = patient_data['Confidence']
                filename_col = 'Filename' if 'Filename' in patient_data.index else 'filename'
                
                # ----------------------------
                # Determine card styling based on predicted dementia class
                # ----------------------------
                if 'Non Demented' in pred_class:
                    bg_color = '#10b981'           # green background for healthy
                    status_icon = '✅'
                    status_text = 'Healthy Brain'
                    risk_level = 'LOW'
                    risk_color = '#10b981'
                    risk_description = 'No signs of dementia detected'
                elif 'Very Mild Demented' in pred_class:
                    bg_color = '#eab308'           # yellow/orange
                    status_icon = '⚠️'
                    status_text = 'Early Signs'
                    risk_level = 'LOW-MEDIUM'
                    risk_color = '#eab308'
                    risk_description = 'Very early signs detected'
                elif 'Mild Demented' in pred_class:
                    bg_color = '#f97316'           # orange
                    status_icon = '⚠️'
                    status_text = 'Mild Dementia'
                    risk_level = 'MEDIUM'
                    risk_color = '#f97316'
                    risk_description = 'Mild cognitive impairment present'
                else:
                    bg_color = '#ef4444'           # red for severe
                    status_icon = '🚨'
                    status_text = 'Moderate Dementia'
                    risk_level = 'HIGH'
                    risk_color = '#ef4444'
                    risk_description = 'Significant cognitive impairment'
                
                # ----------------------------
                # Add "PRIMARY" badge if this is the primary patient
                # ----------------------------
                primary_badge = f'<div style="position: absolute; top: 8px; right: 8px; background: #3b82f6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600;">PRIMARY</div>' if is_primary else ''
                
                # ----------------------------
                # Compose main card HTML
                # ----------------------------
                card_html = f"""
                <div style="
                    background: linear-gradient(135deg, {bg_color}10 0%, {bg_color}20 100%);
                    border-radius: 16px;
                    padding: 20px;
                    border: 2px solid {bg_color}30;
                    position: relative;
                    height: 100%;
                    min-height: 320px;
                ">
                    {primary_badge}
                    
                    <!-- Large faint status icon in background -->
                    <div style="
                        position: absolute;
                        top: -10px;
                        right: -10px;
                        font-size: 60px;
                        opacity: 0.1;
                        transform: rotate(15deg);
                    ">{status_icon}</div>
                    
                    <!-- Patient label -->
                    <h4 style="margin: 0 0 16px 0; color: #111827; font-weight: 600;">
                        {patient_label}
                    </h4>
                    
                    <!-- Prediction status with icon -->
                    <div style="display: flex; align-items: center; margin-bottom: 16px;">
                        <div style="font-size: 28px; margin-right: 12px;">{status_icon}</div>
                        <div>
                            <div style="font-size: 16px; font-weight: 700; color: {bg_color};">
                                {pred_class.replace('_', ' ')}
                            </div>
                            <div style="font-size: 12px; color: #6b7280; font-weight: 500;">
                                {status_text}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Confidence Section -->
                    <div style="
                        background: white;
                        border-radius: 8px;
                        padding: 12px;
                        margin: 12px 0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 10px; color: #6b7280; margin-bottom: 2px;">Confidence</div>
                                <div style="font-size: 20px; font-weight: 700; color: {bg_color};">
                                    {confidence:.1%}
                                </div>
                            </div>
                            <div style="
                                background: {bg_color}15;
                                border-radius: 6px;
                                padding: 4px 8px;
                                border: 1px solid {bg_color}30;
                            ">
                            <div style="font-size: 10px; color: {bg_color}; font-weight: 600;">
                                    {'HIGH' if confidence >= 0.8 else 'MEDIUM' if confidence >= 0.5 else 'LOW'}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Risk Level Section -->
                    <div style="
                        background: white;
                        border-radius: 8px;
                        padding: 12px;
                        margin: 12px 0;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <div style="font-size: 10px; color: #6b7280; margin-bottom: 2px;">Risk Level</div>
                                <div style="font-size: 16px; font-weight: 700; color: {risk_color}; margin-bottom: 4px;">
                                    {risk_level}
                                </div>
                                <div style="font-size: 11px; color: #6b7280; line-height: 1.3;">
                                    {risk_description}
                                </div>
                            </div>
                            <div style="
                                background: {risk_color}15;
                                border-radius: 50%;
                                width: 40px;
                                height: 40px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                border: 2px solid {risk_color}30;
                            ">
                                <div style="font-size: 18px;">{status_icon}</div>
                            </div>
                        </div>
                    </div>
                """
                
                return card_html

            
            def get_region_images(patient_filename, df_images, region_name):
                """Get region-specific images for a patient"""
                # Create a normalized pattern for matching region names
                region_pattern = f"region_{region_name.lower().replace(' ', '_').replace('-', '_')}"
                
                # Filter dataframe for:
                # 1. Rows containing the patient filename (ignoring extension, case-insensitive)
                # 2. Rows where the 'image_type' matches the region pattern
                region_images = df_images[
                    (df_images['filename'].str.contains(patient_filename.split('.')[0], case=False)) & 
                    (df_images['image_type'].str.contains(region_pattern, case=False))
                ]
                return region_images


            def display_region_image_with_score(region_name, importance_score, patient_filename, df_images, patient_label):
                """Display region image with its importance score"""
                # Retrieve matching region images
                region_images = get_region_images(patient_filename, df_images, region_name)
                
                # Split layout into two columns: col1 for image, col2 for score
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if not region_images.empty:
                        # Show first matching image from DB with label
                        display_image_from_db(region_images.iloc[0]['file_path'], f"{patient_label} - {region_name}")
                    else:
                        # Inform user if no region image is found
                        st.info(f"No {region_name} region image available")
                
                with col2:
                    # Determine score color: green (>0.5), orange (>0.3), else red
                    score_color = "green" if importance_score > 0.5 else "orange" if importance_score > 0.3 else "red"
                    
                    # Render styled HTML box for the score
                    st.markdown(f"""
                    <div style="padding: 20px; border: 2px solid {score_color}; border-radius: 10px; text-align: center;">
                        <h3 style="color: {score_color}; margin: 0;">{importance_score:.3f}</h3>
                        <p style="margin: 5px 0; color: gray;">Importance Score</p>
                    </div>
                    """, unsafe_allow_html=True)


            # ----------------- Main Comparison Section -----------------
            st.markdown("---")
            st.markdown('<h4 class="subsection-title">📊 Comparison Overview</h4>', unsafe_allow_html=True)
            
            # Patient Cards Row
            col1, col2 = st.columns(2)
            
            with col1:
                st.html(create_patient_card(primary_patient, "👑 Primary Patient", is_primary=True))
            
            with col2:
                st.html(create_patient_card(comparison_patient, "🔍 Comparison Patient"))
        
            # Enhanced Detailed Comparison Metrics Section
            st.markdown("---")
            st.markdown('<h4 class="subsection-title">🎯 Classification Probability Comparison</h4>', unsafe_allow_html=True)
            st.markdown('<h8>Compare the classification probabilities for each patient across all dementia categories</h8>', unsafe_allow_html=True)

            # Probability Comparison with enhanced styling
            prob_columns = ['Mild_Demented_Probability', 'Moderate_Demented_Probability', 
                        'Non_Demented_Probability', 'Very_Mild_Demented_Probability']

            # Collect probability data for both patients
            primary_probs = {}
            comparison_probs = {}

            for col in prob_columns:
                if col in primary_patient and pd.notna(primary_patient[col]):
                    class_name = col.replace('_Probability', '').replace('_', ' ')
                    primary_probs[class_name] = float(primary_patient[col])
                
                if col in comparison_patient and pd.notna(comparison_patient[col]):
                    class_name = col.replace('_Probability', '').replace('_', ' ')
                    comparison_probs[class_name] = float(comparison_patient[col])

            if primary_probs and comparison_probs:
                # Modern color palette with gradients
                category_config = {
                    'Non Demented': {
                        'primary': '#22c55e', 'secondary': '#16a34a', 
                        'bg': 'rgba(34, 197, 94, 0.1)', 'description': 'Healthy cognition'
                    },
                    'Very Mild Demented': {
                        'primary': '#fbbf24', 'secondary': '#f59e0b', 
                        'bg': 'rgba(251, 191, 36, 0.1)', 'description': 'Minimal concerns'
                    },
                    'Mild Demented': {
                        'primary': '#fb923c', 'secondary': '#ea580c', 
                        'bg': 'rgba(251, 146, 60, 0.1)', 'description': 'Noticeable decline'
                    },
                    'Moderate Demented': {
                        'primary': '#ef4444', 'secondary': '#dc2626', 
                        'bg': 'rgba(239, 68, 68, 0.1)', 'description': 'Significant impairment'
                    }
                }
                
                # Enhanced probability comparison chart
                fig_comparison = go.Figure()
                
                classes = list(primary_probs.keys())
                primary_values = [primary_probs[cls] for cls in classes]
                comparison_values = [comparison_probs[cls] for cls in classes]
                
                # Calculate differences for insights
                differences = [abs(p - c) for p, c in zip(primary_values, comparison_values)]
                max_diff_idx = differences.index(max(differences))
                max_diff_class = classes[max_diff_idx]
                
                # Enhanced primary patient bars with modern gradient
                fig_comparison.add_trace(go.Bar(
                    name='👤 Primary Patient',
                    x=classes,
                    y=primary_values,
                    marker=dict(
                        color=[f'rgba(99, 102, 241, {0.7 + 0.3 * (p / max(primary_values)) if max(primary_values) > 0 else 0.7})' for p in primary_values],
                        line=dict(color='rgba(99, 102, 241, 1)', width=3),
                        pattern=dict(
                            shape="",
                            bgcolor='rgba(99, 102, 241, 0.8)',
                            fgcolor='rgba(139, 92, 246, 0.3)',
                            size=8,
                            solidity=0.2
                        )
                    ),
                    text=[f'<b>{p:.1%}</b>' for p in primary_values],
                    textposition='auto',
                    textfont=dict(size=16, color='white', family='SF Pro Display, sans-serif', weight='bold'),
                    hovertemplate='<b>👤 Primary Patient</b><br>' +
                                '%{x}: <b>%{y:.2%}</b><br>' +
                                '<i>%{customdata}</i><br>' +
                                '<extra></extra>',
                    customdata=[category_config.get(cls, {}).get('description', '') for cls in classes],
                    opacity=0.95,
                    width=0.35
                ))

                # Enhanced comparison patient bars with modern styling
                fig_comparison.add_trace(go.Bar(
                    name='🔍 Comparison Patient',
                    x=classes,
                    y=comparison_values,
                    marker=dict(
                        color=[f'rgba(236, 72, 153, {0.7 + 0.3 * (c / max(comparison_values)) if max(comparison_values) > 0 else 0.7})' for c in comparison_values],
                        line=dict(color='rgba(236, 72, 153, 1)', width=3),
                        pattern=dict(
                            shape="",
                            bgcolor='rgba(236, 72, 153, 0.8)',
                            fgcolor='rgba(251, 113, 133, 0.3)',
                            size=8,
                            solidity=0.2
                        )
                    ),
                    text=[f'<b>{c:.1%}</b>' for c in comparison_values],
                    textposition='auto',
                    textfont=dict(size=16, color='white', family='SF Pro Display, sans-serif', weight='bold'),
                    hovertemplate='<b>🔍 Comparison Patient</b><br>' +
                                '%{x}: <b>%{y:.2%}</b><br>' +
                                '<i>%{customdata}</i><br>' +
                                '<extra></extra>',
                    customdata=[category_config.get(cls, {}).get('description', '') for cls in classes],
                    opacity=0.95,
                    width=0.35
                ))

                # Add confidence bands for each category
                for i, cls in enumerate(classes):
                    config = category_config.get(cls, {})
                    fig_comparison.add_shape(
                        type="rect",
                        x0=i-0.4, x1=i+0.4,
                        y0=0, y1=max(max(primary_values), max(comparison_values)) * 1.25,
                        fillcolor=config.get('bg', 'rgba(156, 163, 175, 0.05)'),
                        layer="below",
                        line_width=0
                    )

                # Enhanced prediction indicators
                primary_max_idx = primary_values.index(max(primary_values))
                comparison_max_idx = comparison_values.index(max(comparison_values))
                
                # Primary patient highest indicator
                fig_comparison.add_annotation(
                    x=primary_max_idx,
                    y=max(primary_values) * 1.08,
                    text="<b>🎯 PREDICTED</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=3,
                    arrowcolor="#6366f1",
                    font=dict(size=12, color="#6366f1", family="SF Pro Display, sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="#6366f1",
                    borderpad=8
                )

                # Comparison patient highest indicator
                fig_comparison.add_annotation(
                    x=comparison_max_idx,
                    y=max(comparison_values) * 1.08,
                    text="<b>🎯 PREDICTED</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=3,
                    arrowcolor="#ec4899",
                    font=dict(size=12, color="#ec4899", family="SF Pro Display, sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="#ec4899",
                    borderpad=8
                )

                # Update layout with modern styling
                fig_comparison.update_layout(
                    yaxis=dict(
                        title=dict(
                            text='<b>Prediction Confidence</b>',
                            font=dict(size=18, family='SF Pro Display, sans-serif', color='#1f2937')
                        ),
                        tickformat='.0%',
                        range=[0, max(max(primary_values), max(comparison_values)) * 1.25],
                        gridcolor='rgba(229, 231, 235, 0.5)',
                        showgrid=True,
                        tickfont=dict(size=16, color='#374151', family='SF Pro Display, sans-serif'),
                        linecolor='#e5e7eb',
                        linewidth=2,
                        showline=True,
                        zeroline=True,
                        zerolinecolor='rgba(156, 163, 175, 0.3)',
                        zerolinewidth=2
                    ),
                    barmode='group',
                    bargap=0.4,
                    bargroupgap=0.15,
                    plot_bgcolor='rgba(255, 255, 255, 0.95)',
                    paper_bgcolor='#fafafa',
                    height=550,
                    margin=dict(l=80, r=80, t=20, b=80),
                    
                    # Modern legend design
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(255, 255, 255, 0.95)',
                        bordercolor='#d1d5db',
                        font=dict(size=16, color='#1f2937', family='SF Pro Display, sans-serif'),
                        itemsizing='constant',
                        itemwidth=40
                    ),
                    
                    font=dict(family="SF Pro Display, sans-serif"),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=15,
                        font_family="SF Pro Display, sans-serif",
                        bordercolor="#e5e7eb"
                    ),
                    
                    # Add subtle border
                    shapes=[
                        dict(
                            type="rect",
                            xref="paper", yref="paper",
                            x0=0, y0=0, x1=1, y1=1,
                            line=dict(color="rgba(229, 231, 235, 0.8)", width=2),
                            fillcolor="rgba(0,0,0,0)"
                        )
                    ]
                )

                st.plotly_chart(fig_comparison, use_container_width=True, config={'displayModeBar': False})
        
            st.markdown('<div style="margin: 3rem 0 2rem 0;"><h3 style="text-align: center; color: #1f2937; font-size: 2rem; font-weight: 700; margin-bottom: 1rem;">📊 Key Performance Indicators</h3></div>', unsafe_allow_html=True)

            # Calculate differences
            confidence_diff = comparison_patient['Confidence'] - primary_patient['Confidence']
            same_class = primary_patient['Predicted_Class'] == comparison_patient['Predicted_Class']

            # Enhanced risk level function
            def get_risk_level(pred_class):
                if 'Non Demented' in pred_class:
                    return 'Low Risk', '#10b981', '🟢'
                elif 'Very Mild Demented' in pred_class:
                    return 'Medium Risk', '#eab308', '🟡'
                elif 'Mild Demented' in pred_class:
                    return 'High Risk', '#f97316', '🟠'
                else:
                    return 'Critical Risk', '#ef4444', '🔴'

            primary_risk, primary_color, primary_icon = get_risk_level(primary_patient['Predicted_Class'])
            comparison_risk, comparison_color, comparison_icon = get_risk_level(comparison_patient['Predicted_Class'])

            col1, col2, col3 = st.columns(3, gap="large")

            with col1:
                # Enhanced Confidence Difference Card
                trend_icon = "📈" if confidence_diff > 0 else "📉" if confidence_diff < 0 else "➡️"
                trend_color = "#16a34a" if confidence_diff > 0 else "#dc2626" if confidence_diff < 0 else "#6b7280"
                trend_bg = "#dcfce7" if confidence_diff > 0 else "#fef2f2" if confidence_diff < 0 else "#f3f4f6"
                trend_text = "Higher" if confidence_diff > 0 else "Lower" if confidence_diff < 0 else "Same"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {trend_bg} 0%, white 100%);
                    border-radius: 20px;
                    padding: 2rem;
                    border: 3px solid {trend_color}40;
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.15)'"
                onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.1)'">
                    <div style="
                        position: absolute;
                        top: -20px;
                        right: -20px;
                        font-size: 100px;
                        opacity: 0.1;
                        color: {trend_color};
                        transform: rotate(15deg);
                    ">{trend_icon}</div>
                    <div style="
                        font-size: 3rem;
                        margin-bottom: 0.5rem;
                        position: relative;
                        z-index: 2;
                    ">{trend_icon}</div>
                    <div style="
                        font-size: 0.9rem;
                        color: #6b7280;
                        margin-bottom: 0.5rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">Confidence Difference</div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 800;
                        color: {trend_color};
                        margin-bottom: 0.5rem;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                    ">{confidence_diff:+.1%}</div>
                    <div style="
                        font-size: 0.85rem;
                        color: #6b7280;
                        font-weight: 500;
                        background: rgba(255,255,255,0.7);
                        padding: 0.3rem 0.8rem;
                        border-radius: 15px;
                        display: inline-block;
                    ">{trend_text} than Primary</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Classification Agreement Card
                # Determine if predicted classes match and set corresponding visuals
                agreement_icon     = "✅" if same_class else "⚠️"         # Icon for match/mismatch
                agreement_color    = "#16a34a" if same_class else "#d97706"  # Green=match, Orange=mismatch
                agreement_bg       = "#dcfce7" if same_class else "#fef3c7"  # Background color
                agreement_text     = "PERFECT MATCH" if same_class else "DIFFERENT"  # Main label
                agreement_subtext  = "Same diagnosis" if same_class else "Different diagnosis"  # Sub-label
                
                # Render Classification Agreement card with hover animation and icon overlay
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {agreement_bg} 0%, white 100%);
                    border-radius: 20px;
                    padding: 2rem;
                    border: 3px solid {agreement_color}40;
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.15)'"
                onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.1)'">
                    <!-- Large faded icon in background -->
                    <div style="
                        position: absolute;
                        top: -20px;
                        right: -20px;
                        font-size: 100px;
                        opacity: 0.1;
                        color: {agreement_color};
                        transform: rotate(15deg);
                    ">{agreement_icon}</div>
                    <!-- Main icon in foreground -->
                    <div style="
                        font-size: 3rem;
                        margin-bottom: 0.5rem;
                        position: relative;
                        z-index: 2;
                    ">{agreement_icon}</div>
                    <!-- Title -->
                    <div style="
                        font-size: 0.9rem;
                        color: #6b7280;
                        margin-bottom: 0.5rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">Classification Agreement</div>
                    <!-- Main status text -->
                    <div style="
                        font-size: 1.3rem;
                        font-weight: 800;
                        color: {agreement_color};
                        margin-bottom: 0.5rem;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                    ">{agreement_text}</div>
                    <!-- Sub status text -->
                    <div style="
                        font-size: 0.85rem;
                        color: #6b7280;
                        font-weight: 500;
                        background: rgba(255,255,255,0.7);
                        padding: 0.3rem 0.8rem;
                        border-radius: 15px;
                        display: inline-block;
                    ">{agreement_subtext}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                # Risk Level Comparison Card
                # This shows both patients' risk levels side by side with colors and icons
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {primary_color}15 0%, {comparison_color}15 50%, white 100%);
                    border-radius: 20px;
                    padding: 2rem;
                    border: 3px solid #e5e7eb;
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.15)'"
                onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.1)'">
                    <!-- Faded target icon in background -->
                    <div style="
                        position: absolute;
                        top: -20px;
                        right: -20px;
                        font-size: 100px;
                        opacity: 0.1;
                        color: #6b7280;
                        transform: rotate(15deg);
                    ">🎯</div>
                    <!-- Foreground icon -->
                    <div style="
                        font-size: 3rem;
                        margin-bottom: 0.5rem;
                        position: relative;
                        z-index: 2;
                    ">🎯</div>
                    <!-- Title -->
                    <div style="
                        font-size: 0.9rem;
                        color: #6b7280;
                        margin-bottom: 0.8rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    ">Risk Level Comparison</div>
                    <!-- Side-by-side comparison container -->
                    <div style="
                        display: flex;
                        justify-content: space-around;
                        align-items: center;
                        background: rgba(255,255,255,0.7);
                        padding: 0.8rem;
                        border-radius: 15px;
                    ">
                        <!-- Primary patient section -->
                        <div style="text-align: center;">
                            <div style="font-size: 0.7rem; color: #6b7280; margin-bottom: 0.2rem; font-weight: 600;">PRIMARY</div>
                            <div style="font-size: 1rem; font-weight: 700; color: {primary_color}; display: flex; align-items: center; gap: 0.3rem;">
                                {primary_icon} <span style="font-size: 0.8rem;">{primary_risk.split()[0]}</span>
                            </div>
                        </div>
                        <!-- VS label -->
                        <div style="
                            color: #6b7280;
                            font-size: 1.2rem;
                            font-weight: 700;
                            transform: scale(1.5);
                            opacity: 0.5;
                        ">VS</div>
                        <!-- Comparison patient section -->
                        <div style="text-align: center;">
                            <div style="font-size: 0.7rem; color: #6b7280; margin-bottom: 0.2rem; font-weight: 600;">COMPARISON</div>
                            <div style="font-size: 1rem; font-weight: 700; color: {comparison_color}; display: flex; align-items: center; gap: 0.3rem;">
                                {comparison_icon} <span style="font-size: 0.8rem;">{comparison_risk.split()[0]}</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ----------------------------
            # Visual Comparison Section
            # ----------------------------
            st.markdown("---")
            st.markdown('<h4 class="subsection-title">🖼️ Visual Comparison</h4>', unsafe_allow_html=True)

            # Retrieve images for both patients from the image dataframe
            primary_images = get_patient_images(primary_patient.get(filename_col, ''), df_images)
            comparison_images = get_patient_images(comparison_patient.get(filename_col, ''), df_images)

            # Check if both patients have images available
            if not primary_images.empty and not comparison_images.empty:
                st.markdown("### 🖼️ Select Image Type:")
                
                # Create button navigation for image tabs (Original, Heatmaps, Brain Region)
                img_col1, img_col2, img_col3 = st.columns(3)
                
                with img_col1:
                    # Button for Original MRI scans
                    if st.button(
                        "🧠 Original Scans",
                        key="brain_img_original_btn",
                        use_container_width=True,
                        type="primary" if st.session_state.tab3_brain_image_tab == "🧠 Original Scans" else "secondary"
                    ):
                        st.session_state.tab3_brain_image_tab = "🧠 Original Scans"
                        st.rerun()
                
                with img_col2:
                    # Button for Heatmaps
                    if st.button(
                        "🔥 Heatmaps",
                        key="brain_img_heatmap_btn",
                        use_container_width=True,
                        type="primary" if st.session_state.tab3_brain_image_tab == "🔥 Heatmaps" else "secondary"
                    ):
                        st.session_state.tab3_brain_image_tab = "🔥 Heatmaps"
                        st.rerun()
                
                with img_col3:
                    # Button for Brain Region Importance
                    if st.button(
                        "🧬 Brain Region",
                        key="brain_img_region_btn",
                        use_container_width=True,
                        type="primary" if st.session_state.tab3_brain_image_tab == "🧬 Brain Region" else "secondary"
                    ):
                        st.session_state.tab3_brain_image_tab = "🧬 Brain Region"
                        st.rerun()

                st.markdown("---")

                # ----------------------------
                # Original MRI Tab
                # ----------------------------
                if st.session_state.tab3_brain_image_tab == "🧠 Original Scans":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**👑 Primary Patient - Original MRI**")
                        # Filter primary patient original images
                        primary_original = primary_images[primary_images['image_type'].str.contains('original', case=False)]
                        if not primary_original.empty:
                            display_image_from_db(primary_original.iloc[0]['file_path'], "Primary Patient - Original MRI")
                        else:
                            st.info("No original MRI available")
                    
                    with col2:
                        st.markdown("**🔍 Comparison Patient - Original MRI**")
                        # Filter comparison patient original images
                        comparison_original = comparison_images[comparison_images['image_type'].str.contains('original', case=False)]
                        if not comparison_original.empty:
                            display_image_from_db(comparison_original.iloc[0]['file_path'], "Comparison Patient - Original MRI")
                        else:
                            st.info("No original MRI available")

                # ----------------------------
                # Heatmap Tab
                # ----------------------------
                elif st.session_state.tab3_brain_image_tab == "🔥 Heatmaps":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**👑 Primary Patient - Attention Heatmap**")
                        primary_heatmap = primary_images[primary_images['image_type'].str.contains('heatmap', case=False)]
                        if not primary_heatmap.empty:
                            display_image_from_db(primary_heatmap.iloc[0]['file_path'], "Primary Patient - Heatmap")
                        else:
                            st.info("No heatmap available")
                    
                    with col2:
                        st.markdown("**🔍 Comparison Patient - Attention Heatmap**")
                        comparison_heatmap = comparison_images[comparison_images['image_type'].str.contains('heatmap', case=False)]
                        if not comparison_heatmap.empty:
                            display_image_from_db(comparison_heatmap.iloc[0]['file_path'], "Comparison Patient - Heatmap")
                        else:
                            st.info("No heatmap available")

                # ----------------------------
                # Brain Region Analysis Tab
                # ----------------------------
                elif st.session_state.tab3_brain_image_tab == "🧬 Brain Region":
                    st.markdown('<h4 class="subsection-title">🧠 Brain Region Importance Analysis</h4>', unsafe_allow_html=True)
                    
                    # Use a container to avoid unnecessary reruns
                    brain_region_container = st.container()
                    
                    with brain_region_container:
                        if not primary_regions.empty and not comparison_regions.empty:
                            # Initialize session state for stable tab behavior
                            if 'tab3_brain_analysis_option' not in st.session_state:
                                st.session_state.tab3_brain_analysis_option = "📊 Comparative Chart"
                            if 'tab3_brain_chart_type' not in st.session_state:
                                st.session_state.tab3_brain_chart_type = "📊 Side-by-Side Bars"
                            if 'tab3_selected_region' not in st.session_state:
                                st.session_state.tab3_selected_region = None
                            
                            # ----------------------------
                            # Analysis Option Buttons
                            # ----------------------------
                            st.markdown("**Select Analysis Type:**")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Button for Comparative Chart
                                if st.button(
                                    "📊 Comparative Chart",
                                    key="tab3_comp_chart_btn_fixed",  # stable key
                                    use_container_width=True,
                                    type="primary" if st.session_state.tab3_brain_analysis_option == "📊 Comparative Chart" else "secondary"
                                ):
                                    st.session_state.tab3_brain_analysis_option = "📊 Comparative Chart"
                                    st.session_state.last_action = 'analysis_option_change'
                                    st.rerun()
                            
                            with col2:
                                # Button for Individual Region Analysis
                                if st.button(
                                    "🔍 Individual Region Analysis",
                                    key="tab3_ind_region_btn_fixed",  # stable key
                                    use_container_width=True,
                                    type="primary" if st.session_state.tab3_brain_analysis_option == "🔍 Individual Region Analysis" else "secondary"
                                ):
                                    st.session_state.tab3_brain_analysis_option = "🔍 Individual Region Analysis"
                                    st.session_state.last_action = 'analysis_option_change'
                                    st.rerun()
                            
                            # Current selected analysis option
                            analysis_option = st.session_state.tab3_brain_analysis_option
                            
                            # ----------------------------
                            # Comparative Chart Processing
                            # ----------------------------
                            if analysis_option == "📊 Comparative Chart":
                                try:
                                    # Merge primary and comparison brain region importance scores
                                    merged_regions = pd.merge(
                                        primary_regions[['Brain_Region', 'ScoreCAM_Importance_Score']],
                                        comparison_regions[['Brain_Region', 'ScoreCAM_Importance_Score']],
                                        on='Brain_Region',
                                        how='outer',
                                        suffixes=('_Primary', '_Comparison')
                                    ).fillna(0)
                                    
                                    # Calculate differences between comparison and primary scores
                                    merged_regions['Importance_Diff'] = merged_regions['ScoreCAM_Importance_Score_Comparison'] - merged_regions['ScoreCAM_Importance_Score_Primary']
                                    merged_regions['Abs_Diff'] = abs(merged_regions['Importance_Diff'])
                                    merged_regions['Relative_Change'] = np.where(
                                        merged_regions['ScoreCAM_Importance_Score_Primary'] > 0,
                                        (merged_regions['Importance_Diff'] / merged_regions['ScoreCAM_Importance_Score_Primary']) * 100,
                                        0
                                    )
                                    
                                    # Sort by absolute difference to show top regions
                                    top_diff_regions = merged_regions.nlargest(12, 'Abs_Diff')

                                    # ----------------------------
                                    # Check if top brain regions exist for comparison
                                    # ----------------------------
                                    if len(top_diff_regions) == 0:
                                        # Warn user if no data is available
                                        st.warning("⚠️ No brain region data available for comparison")
                                    else:
                                        # ----------------------------
                                        # Chart Type Selection Buttons
                                        # ----------------------------
                                        st.markdown("**Select Chart Type:**")
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Button for Side-by-Side Bar chart
                                            if st.button(
                                                "📊 Side-by-Side Bars",
                                                key="tab3_side_by_side_btn_fixed",  # Stable key for session state
                                                use_container_width=True,
                                                type="primary" if st.session_state.tab3_brain_chart_type == "📊 Side-by-Side Bars" else "secondary"
                                            ):
                                                st.session_state.tab3_brain_chart_type = "📊 Side-by-Side Bars"
                                                st.session_state.last_action = 'chart_type_change'
                                                st.rerun()
                                        
                                        with col2:
                                            # Button for Difference Analysis chart
                                            if st.button(
                                                "📈 Difference Analysis",
                                                key="tab3_diff_analysis_btn_fixed",  # Stable key for session state
                                                use_container_width=True,
                                                type="primary" if st.session_state.tab3_brain_chart_type == "📈 Difference Analysis" else "secondary"
                                            ):
                                                st.session_state.tab3_brain_chart_type = "📈 Difference Analysis"
                                                st.session_state.last_action = 'chart_type_change'
                                                st.rerun()
                                        
                                        # Retrieve currently selected chart type
                                        chart_type = st.session_state.tab3_brain_chart_type

                                        # ----------------------------
                                        # Side-by-Side Bar Chart
                                        # ----------------------------
                                        if chart_type == "📊 Side-by-Side Bars":
                                            # Compute total importance scores for both patients
                                            total_primary = top_diff_regions['ScoreCAM_Importance_Score_Primary'].sum()
                                            total_comparison = top_diff_regions['ScoreCAM_Importance_Score_Comparison'].sum()
                                            
                                            # Convert raw scores to percentages for better visualization
                                            primary_percentages = (top_diff_regions['ScoreCAM_Importance_Score_Primary'] / total_primary * 100).tolist()
                                            comparison_percentages = (top_diff_regions['ScoreCAM_Importance_Score_Comparison'] / total_comparison * 100).tolist()
                                            
                                            # Initialize Plotly figure
                                            fig_region_comparison = go.Figure()
                                            
                                            # ----------------------------
                                            # Primary Patient Bars
                                            # ----------------------------
                                            fig_region_comparison.add_trace(go.Bar(
                                                name='👤 Primary Patient',
                                                x=top_diff_regions['Brain_Region'],
                                                y=primary_percentages,
                                                marker=dict(
                                                    color='rgba(99, 102, 241, 0.8)',
                                                    line=dict(color='rgba(99, 102, 241, 1)', width=3),
                                                    pattern=dict(shape="", solidity=0.8)
                                                ),
                                                # Display percentage on bars
                                                text=[f'<b>{p:.1f}%</b>' for p in primary_percentages],
                                                textposition='auto',
                                                textfont=dict(size=14, color='white', family='SF Pro Display, sans-serif', weight='bold'),
                                                hovertemplate='<b>👤 Primary Patient</b><br>' +
                                                            'Region: <b>%{x}</b><br>' +
                                                            'Importance: <b>%{y:.1f}%</b><br>' +
                                                            'Raw Score: <b>%{customdata:.4f}</b><br>' +
                                                            '<extra></extra>',
                                                customdata=top_diff_regions['ScoreCAM_Importance_Score_Primary'],
                                                width=0.35
                                            ))
                                            
                                            # ----------------------------
                                            # Comparison Patient Bars
                                            # ----------------------------
                                            fig_region_comparison.add_trace(go.Bar(
                                                name='🔍 Comparison Patient',
                                                x=top_diff_regions['Brain_Region'],
                                                y=comparison_percentages,
                                                marker=dict(
                                                    color='rgba(236, 72, 153, 0.8)',
                                                    line=dict(color='rgba(236, 72, 153, 1)', width=3),
                                                    pattern=dict(shape="", solidity=0.8)
                                                ),
                                                text=[f'<b>{p:.1f}%</b>' for p in comparison_percentages],
                                                textposition='auto',
                                                textfont=dict(size=14, color='white', family='SF Pro Display, sans-serif', weight='bold'),
                                                hovertemplate='<b>🔍 Comparison Patient</b><br>' +
                                                            'Region: <b>%{x}</b><br>' +
                                                            'Importance: <b>%{y:.1f}%</b><br>' +
                                                            'Raw Score: <b>%{customdata:.4f}</b><br>' +
                                                            '<extra></extra>',
                                                customdata=top_diff_regions['ScoreCAM_Importance_Score_Comparison'],
                                                width=0.35
                                            ))
                                            
                                            # ----------------------------
                                            # Layout Configuration
                                            # ----------------------------
                                            fig_region_comparison.update_layout(
                                                xaxis=dict(
                                                    title=dict(
                                                        text='<b>Brain Regions</b>',
                                                        font=dict(size=18, family='SF Pro Display, sans-serif', color='#1f2937')
                                                    ),
                                                    tickangle=45,
                                                    tickfont=dict(size=14, color='#374151', family='SF Pro Display, sans-serif'),
                                                    showgrid=False,
                                                    linecolor='#e5e7eb',
                                                    linewidth=2
                                                ),
                                                yaxis=dict(
                                                    title=dict(
                                                        text='<b>Importance Percentage (%)</b>',
                                                        font=dict(size=18, family='SF Pro Display, sans-serif', color='#1f2937')
                                                    ),
                                                    tickformat='.1f',
                                                    ticksuffix='%',
                                                    gridcolor='rgba(229, 231, 235, 0.5)',
                                                    showgrid=True,
                                                    tickfont=dict(size=14, color='#374151', family='SF Pro Display, sans-serif'),
                                                    linecolor='#e5e7eb',
                                                    linewidth=2,
                                                    range=[0, max(max(primary_percentages), max(comparison_percentages)) * 1.1]
                                                ),
                                                barmode='group',
                                                bargap=0.3,
                                                bargroupgap=0.15,
                                                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                                                paper_bgcolor='#fafafa',
                                                height=600,
                                                margin=dict(l=80, r=80, t=60, b=120),
                                                
                                                # ----------------------------
                                                # Legend Configuration
                                                # ----------------------------
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.1,
                                                    xanchor="center",
                                                    x=0.5,
                                                    bgcolor='rgba(255, 255, 255, 0.95)',
                                                    bordercolor='#d1d5db',
                                                    font=dict(size=16, color='#1f2937', family='SF Pro Display, sans-serif'),
                                                    itemsizing='constant',
                                                    itemwidth=40
                                                ),
                                                
                                                font=dict(family="SF Pro Display, sans-serif"),
                                                hoverlabel=dict(
                                                    bgcolor="white",
                                                    font_size=15,
                                                    font_family="SF Pro Display, sans-serif",
                                                    bordercolor="#e5e7eb"
                                                ),
                                                
                                                # Add subtle border around plot
                                                shapes=[
                                                    dict(
                                                        type="rect",
                                                        xref="paper", yref="paper",
                                                        x0=0, y0=0, x1=1, y1=1,
                                                        line=dict(color="rgba(229, 231, 235, 0.8)", width=2),
                                                        fillcolor="rgba(0,0,0,0)"
                                                    )
                                                ]
                                            )
                                            
                                            # Render the Plotly figure in Streamlit
                                            st.plotly_chart(fig_region_comparison, use_container_width=True, config={'displayModeBar': False})

                                        # ----------------------------
                                        # Difference Analysis Chart
                                        # ----------------------------
                                        elif chart_type == "📈 Difference Analysis":
                                            # ----------------------------
                                            # Compute percentage scores for each patient
                                            # ----------------------------
                                            total_primary = top_diff_regions['ScoreCAM_Importance_Score_Primary'].sum()
                                            total_comparison = top_diff_regions['ScoreCAM_Importance_Score_Comparison'].sum()
                                            
                                            primary_percentages = (top_diff_regions['ScoreCAM_Importance_Score_Primary'] / total_primary * 100)
                                            comparison_percentages = (top_diff_regions['ScoreCAM_Importance_Score_Comparison'] / total_comparison * 100)
                                            
                                            # Compute difference in percentage points between primary and comparison patient
                                            percentage_differences = (primary_percentages - comparison_percentages).tolist()
                                            
                                            # Initialize Plotly figure
                                            fig_diff = go.Figure()
                                            
                                            # ----------------------------
                                            # Assign colors based on magnitude and direction of percentage difference
                                            # ----------------------------
                                            colors = []
                                            for diff in percentage_differences:
                                                if diff > 5:
                                                    colors.append('#dc2626')  # Strong increase - red
                                                elif diff > 2:
                                                    colors.append('#f97316')  # Moderate increase - orange
                                                elif diff > 0:
                                                    colors.append('#eab308')  # Slight increase - yellow
                                                elif diff > -2:
                                                    colors.append('#06b6d4')  # Slight decrease - cyan
                                                elif diff > -5:
                                                    colors.append('#0891b2')  # Moderate decrease - blue
                                                else:
                                                    colors.append('#059669')  # Strong decrease - green
                                            
                                            # ----------------------------
                                            # Add bar trace for percentage differences
                                            # ----------------------------
                                            fig_diff.add_trace(go.Bar(
                                                x=top_diff_regions['Brain_Region'],        # Brain regions on x-axis
                                                y=percentage_differences,                  # Difference in % points
                                                marker=dict(
                                                    color=colors,                          # Color-coded based on difference
                                                    line=dict(color='white', width=2),
                                                    opacity=0.9
                                                ),
                                                # Show text labels above bars
                                                text=[f'<b>{d:+.1f}%</b>' for d in percentage_differences],
                                                textposition='outside',
                                                textfont=dict(size=14, color='#1f2937', family='SF Pro Display, sans-serif', weight='bold'),
                                                # Hover info for detailed values
                                                hovertemplate='<b>%{x}</b><br>' +
                                                            'Percentage Difference: <b>%{y:+.2f}%</b><br>' +
                                                            'Primary: <b>%{customdata[0]:.1f}%</b><br>' +
                                                            'Comparison: <b>%{customdata[1]:.1f}%</b><br>' +
                                                            '<extra></extra>',
                                                customdata=list(zip(primary_percentages, comparison_percentages))  # Attach original percentages
                                            ))
                                            
                                            # ----------------------------
                                            # Layout and axis configuration
                                            # ----------------------------
                                            fig_diff.update_layout(
                                                xaxis=dict(
                                                    title=dict(
                                                        text='<b>Brain Regions</b>',
                                                        font=dict(size=18, family='SF Pro Display, sans-serif', color='#1f2937')
                                                    ),
                                                    tickangle=45,
                                                    tickfont=dict(size=14, color='#374151', family='SF Pro Display, sans-serif'),
                                                    showgrid=False,
                                                    linecolor='#e5e7eb',
                                                    linewidth=2
                                                ),
                                                yaxis=dict(
                                                    title=dict(
                                                        text='<b>Percentage Point Difference</b>',
                                                        font=dict(size=18, family='SF Pro Display, sans-serif', color='#1f2937')
                                                    ),
                                                    tickformat='+.1f',
                                                    ticksuffix='%',
                                                    gridcolor='rgba(229, 231, 235, 0.5)',
                                                    showgrid=True,
                                                    tickfont=dict(size=14, color='#374151', family='SF Pro Display, sans-serif'),
                                                    linecolor='#e5e7eb',
                                                    linewidth=2,
                                                    zeroline=True,
                                                    zerolinecolor='rgba(107, 114, 128, 0.3)',  # Highlight zero line
                                                    zerolinewidth=2
                                                ),
                                                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                                                paper_bgcolor='#fafafa',
                                                height=600,
                                                margin=dict(l=80, r=80, t=60, b=180),
                                                showlegend=False,
                                                font=dict(family="SF Pro Display, sans-serif"),
                                                hoverlabel=dict(
                                                    bgcolor="white",
                                                    font_size=15,
                                                    font_family="SF Pro Display, sans-serif",
                                                    bordercolor="#e5e7eb"
                                                ),
                                                # Subtle border around plot
                                                shapes=[
                                                    dict(
                                                        type="rect",
                                                        xref="paper", yref="paper",
                                                        x0=0, y0=0, x1=1, y1=1,
                                                        line=dict(color="rgba(229, 231, 235, 0.8)", width=2),
                                                        fillcolor="rgba(0,0,0,0)"
                                                    )
                                                ]
                                            )
                                            
                                            # Render the Difference Analysis chart
                                            st.plotly_chart(fig_diff, use_container_width=True, config={'displayModeBar': False})
                                            
                                            # ----------------------------
                                            # Interpretation Section
                                            # ----------------------------
                                            # Color-coded guide for understanding differences
                                            st.markdown("""
                                            <div style="
                                                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                                                border-radius: 15px;
                                                padding: 1.5rem;
                                                margin: 1rem 0;
                                                border-left: 5px solid #10b981;
                                                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                            ">
                                                <h4 style="color: #059669; margin: 0 0 1rem 0; font-family: 'SF Pro Display', sans-serif;">📊 Percentage Difference Interpretation</h4>
                                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; font-family: 'SF Pro Display', sans-serif;">
                                                    <div style="padding: 0.75rem; background: rgba(220, 38, 38, 0.1); border-radius: 8px;">
                                                        <span style="color: #dc2626; font-weight: bold;">🔴 Major Increase:</span> > +5%
                                                    </div>
                                                    <div style="padding: 0.75rem; background: rgba(249, 115, 22, 0.1); border-radius: 8px;">
                                                        <span style="color: #f97316; font-weight: bold;">🟠 Moderate Increase:</span> +2% to +5%
                                                    </div>
                                                    <div style="padding: 0.75rem; background: rgba(234, 179, 8, 0.1); border-radius: 8px;">
                                                        <span style="color: #eab308; font-weight: bold;">🟡 Minor Increase:</span> 0% to +2%
                                                    </div>
                                                    <div style="padding: 0.75rem; background: rgba(6, 182, 212, 0.1); border-radius: 8px;">
                                                        <span style="color: #06b6d4; font-weight: bold;">🔵 Minor Decrease:</span> 0% to -2%
                                                    </div>
                                                    <div style="padding: 0.75rem; background: rgba(8, 145, 178, 0.1); border-radius: 8px;">
                                                        <span style="color: #0891b2; font-weight: bold;">🔷 Moderate Decrease:</span> -2% to -5%
                                                    </div>
                                                    <div style="padding: 0.75rem; background: rgba(5, 150, 105, 0.1); border-radius: 8px;">
                                                        <span style="color: #059669; font-weight: bold;">🟢 Major Decrease:</span> < -5%
                                                    </div>
                                                </div>
                                                <div style="margin-top: 1rem; padding: 1rem; background: rgba(255, 255, 255, 0.7); border-radius: 8px; font-size: 14px; color: #6b7280;">
                                                    <strong>Note:</strong> Percentage differences show how much more (positive) or less (negative) attention each brain region receives in the Primary Patient compared to the Comparison Patient.
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)

                                # ----------------------------
                                # Error Handling
                                # ----------------------------
                                except Exception as e:
                                    st.error(f"❌ Error in brain region analysis: {str(e)}")
                                    st.write("Debug info:", {
                                        'Primary regions shape': primary_regions.shape if 'primary_regions' in locals() else 'Not available',
                                        'Comparison regions shape': comparison_regions.shape if 'comparison_regions' in locals() else 'Not available'
                                    })

                            elif analysis_option == "🔍 Individual Region Analysis":
                                # ----------------------------
                                # Combine and sort all available brain regions from both patients
                                # ----------------------------
                                available_regions = sorted(list(set(primary_regions['Brain_Region'].tolist() + comparison_regions['Brain_Region'].tolist())))
                                
                                # ----------------------------
                                # Initialize session state for selected region (persistent selection)
                                # ----------------------------
                                if st.session_state.tab3_selected_region is None or st.session_state.tab3_selected_region not in available_regions:
                                    st.session_state.tab3_selected_region = available_regions[0] if available_regions else None
                                
                                st.markdown("**Select Brain Region for Detailed Analysis:**")
                                
                                # ----------------------------
                                # Radio selector for choosing the region (horizontal layout, stable key)
                                # ----------------------------
                                selected_region = st.radio(
                                    "Choose region:",
                                    available_regions,
                                    key="tab3_region_radio_fixed",  # stable key to prevent rerun issues
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    index=available_regions.index(st.session_state.tab3_selected_region) if st.session_state.tab3_selected_region in available_regions else 0
                                )
                                
                                # Update session state with current selection
                                st.session_state.tab3_selected_region = selected_region
                                
                                # ----------------------------
                                # Display detailed analysis if a region is selected
                                # ----------------------------
                                if selected_region:
                                    st.markdown(f"### 🧠 Detailed Analysis: {selected_region}")
                                    
                                    # Retrieve importance scores for the selected region
                                    primary_score = primary_regions[primary_regions['Brain_Region'] == selected_region]['ScoreCAM_Importance_Score'].iloc[0] \
                                                    if not primary_regions[primary_regions['Brain_Region'] == selected_region].empty else 0
                                    comparison_score = comparison_regions[comparison_regions['Brain_Region'] == selected_region]['ScoreCAM_Importance_Score'].iloc[0] \
                                                    if not comparison_regions[comparison_regions['Brain_Region'] == selected_region].empty else 0
                                    
                                    # ----------------------------
                                    # Layout columns for primary, comparison, and difference visualizations
                                    # ----------------------------
                                    col1, col2 = st.columns(2)
                                    
                                    # ----------------------------
                                    # Display Primary Patient image and score
                                    # ----------------------------
                                    with col1:
                                        st.markdown("**👑 Primary Patient**")
                                        display_region_image_with_score(
                                            selected_region, 
                                            primary_score, 
                                            primary_patient.get(filename_col, ''), 
                                            df_images, 
                                            "Primary Patient"
                                        )
                                    
                                    # ----------------------------
                                    # Display Comparison Patient image and score
                                    # ----------------------------
                                    with col2:
                                        st.markdown("**🔍 Comparison Patient**")
                                        display_region_image_with_score(
                                            selected_region, 
                                            comparison_score, 
                                            comparison_patient.get(filename_col, ''), 
                                            df_images, 
                                            "Comparison Patient"
                                        )
                                    
                                    # ----------------------------
                                    # Compute difference and absolute difference
                                    # ----------------------------
                                    score_diff = comparison_score - primary_score
                                    abs_diff = abs(score_diff)

                                    # ----------------------------
                                    # Determine significance level based on difference thresholds
                                    # ----------------------------
                                    if abs_diff > 0.15:
                                        significance = "Major"
                                        significance_color = "#dc2626"
                                        significance_bg = "#fef2f2"
                                        significance_icon = "🚨"
                                    elif abs_diff > 0.08:
                                        significance = "Moderate"
                                        significance_color = "#f59e0b"
                                        significance_bg = "#fef3c7"
                                        significance_icon = "⚠️"
                                    elif abs_diff > 0.03:
                                        significance = "Minor"
                                        significance_color = "#0891b2"
                                        significance_bg = "#cffafe"
                                        significance_icon = "ℹ️"
                                    else:
                                        significance = "Minimal"
                                        significance_color = "#059669"
                                        significance_bg = "#dcfce7"
                                        significance_icon = "✅"

                                    # ----------------------------
                                    # Columns for visual cards: Primary score, Comparison score, Difference
                                    # ----------------------------
                                    col1, col2, col3 = st.columns(3)

                                    # Primary patient card
                                    with col1:
                                        primary_percentage = min(primary_score * 100, 100)
                                        st.markdown(f"""...""", unsafe_allow_html=True)  # Card HTML (blue gradient, score bar, intensity)

                                    # Comparison patient card
                                    with col2:
                                        comparison_percentage = min(comparison_score * 100, 100)
                                        st.markdown(f"""...""", unsafe_allow_html=True)  # Card HTML (red gradient, score bar, intensity)

                                    # Difference card
                                    with col3:
                                        diff_percentage = min(abs(score_diff) * 100, 100)
                                        st.markdown(f"""...""", unsafe_allow_html=True)  # Card HTML (color based on significance)

                                else:
                                    # ----------------------------
                                    # Fallback warning if no region data available
                                    # ----------------------------
                                    st.warning("⚠️ Region importance data not available for one or both patients")

                                        















