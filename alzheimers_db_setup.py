"""
Alzheimer's Disease Prediction Database Storage System

This module provides a comprehensive database solution for storing and managing
Alzheimer's disease prediction results from machine learning models. It handles
both clinical data analysis and brain MRI image processing with explainability.
"""

import sqlite3
import json
from pathlib import Path
import shutil
from PIL import Image # type: ignore For image metadata extraction


class AlzheimerPredictionStorage:
    """
    Streamlined Database Storage for Alzheimer's Disease Analysis
    
    This simplified version only includes the functions actually used by your upload code:
    - store_global_importance() - for CSV clinical data feature importance
    - store_individual_prediction() - for CSV clinical predictions
    - store_batch_prediction() - for image batch predictions  
    - store_batch_region() - for brain region analysis
    - store_image_with_metadata() - for storing analysis images
    - close() - for cleanup
    """

    def __init__(self, base_dir=None):
        """
        Initialize the storage system with only essential tables.
        Works both locally and on Render.
        """
        # Use relative path if no base_dir provided
        if base_dir is None:
            self.base_dir = Path.cwd() / "Alzheimer_Project" / "Alzheimer_Database"
        else:
            self.base_dir = Path(base_dir)
        
        self.images_dir = self.base_dir / "stored_images"
        self.db_path = self.base_dir / "alzheimer_predictions.db"
        
        # Create directories safely
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Initialize SQLite database
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """
        Create comprehensive database schema for Alzheimer's analysis storage.
        
        This method sets up six main tables with appropriate relationships:
        
        1. global_importance: Model feature importance rankings
        2. individual_predictions: Detailed patient predictions
        3. batch_predictions: Image-based batch analysis
        4. batch_region: Brain region importance scores
        5. stored_images: Medical image file metadata
        6. analysis_sessions: Complete analysis session tracking
        
        Each table includes appropriate indexes for optimal query performance
        in medical research scenarios.
        """
        
        # ====================================================================
        # TABLE 1: GLOBAL FEATURE IMPORTANCE
        # ====================================================================
        # Stores model-wide feature importance statistics across all patients
        # Used for understanding which clinical features are most predictive
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS global_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Feature TEXT NOT NULL,                    -- Clinical feature name (e.g., 'MMSE', 'Age')
                Mean_Absolute_SHAP REAL NOT NULL,        -- Mean absolute SHAP value across patients
                Mean_SHAP REAL NOT NULL,                 -- Mean SHAP value (can be negative)
                Std_SHAP REAL NOT NULL,                  -- Standard deviation of SHAP values
                Max_SHAP REAL NOT NULL,                  -- Maximum SHAP value observed
                Min_SHAP REAL NOT NULL,                  -- Minimum SHAP value observed
                Importance_Rank INTEGER NOT NULL,         -- Rank by importance (1 = most important)
                model_name TEXT,                         -- Model identifier
                model_version TEXT,                      -- Model version for tracking
                UNIQUE(Feature, model_name, model_version)  -- Prevent duplicates
            )
        ''')
        
        # ====================================================================
        # TABLE 2: INDIVIDUAL PATIENT PREDICTIONS
        # ====================================================================
        # Comprehensive storage for individual patient analysis including
        # all clinical features and model predictions
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS individual_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Patient_ID TEXT NOT NULL,                -- Unique patient identifier
                Predicted_Diagnosis INTEGER NOT NULL,    -- Model prediction (0-3 for 4-class)
                Prediction_Probability REAL NOT NULL,    -- Probability of predicted class
                Prediction_Confidence REAL NOT NULL,     -- Overall model confidence
                model_name TEXT,                         -- Model identifier
                model_version TEXT,                      -- Model version
                
                -- RAW CLINICAL FEATURES (as collected from medical records)
                -- Demographic Information
                RAW_Age REAL,                           -- Patient age in years
                RAW_Gender INTEGER,                     -- Gender (0/1 encoded)
                RAW_Ethnicity INTEGER,                  -- Ethnicity code
                RAW_EducationLevel INTEGER,             -- Education level code
                
                -- Lifestyle Factors
                RAW_BMI REAL,                          -- Body Mass Index
                RAW_Smoking INTEGER,                   -- Smoking status (0/1)
                RAW_AlcoholConsumption REAL,           -- Alcohol consumption level
                RAW_PhysicalActivity REAL,             -- Physical activity score
                RAW_DietQuality REAL,                  -- Diet quality assessment
                RAW_SleepQuality REAL,                 -- Sleep quality score
                
                -- Medical History
                RAW_FamilyHistoryAlzheimers INTEGER,   -- Family history flag
                RAW_CardiovascularDisease INTEGER,     -- Cardiovascular disease flag
                RAW_Diabetes INTEGER,                  -- Diabetes flag
                RAW_Depression INTEGER,                -- Depression flag
                RAW_HeadInjury INTEGER,               -- Head injury history
                RAW_Hypertension INTEGER,             -- Hypertension flag
                
                -- Vital Signs and Lab Results
                RAW_SystolicBP REAL,                  -- Systolic blood pressure
                RAW_DiastolicBP REAL,                 -- Diastolic blood pressure
                RAW_CholesterolTotal REAL,            -- Total cholesterol
                RAW_CholesterolLDL REAL,              -- LDL cholesterol
                RAW_CholesterolHDL REAL,              -- HDL cholesterol
                RAW_CholesterolTriglycerides REAL,    -- Triglycerides
                
                -- Cognitive Assessments
                RAW_MMSE REAL,                        -- Mini-Mental State Examination score
                RAW_FunctionalAssessment REAL,       -- Functional assessment score
                RAW_MemoryComplaints INTEGER,         -- Memory complaints flag
                RAW_BehavioralProblems INTEGER,       -- Behavioral problems flag
                RAW_ADL REAL,                         -- Activities of Daily Living score
                
                -- Symptom Assessment
                RAW_Confusion INTEGER,                -- Confusion episodes flag
                RAW_Disorientation INTEGER,           -- Disorientation flag
                RAW_PersonalityChanges INTEGER,       -- Personality changes flag
                RAW_DifficultyCompletingTasks INTEGER, -- Task completion difficulty
                RAW_Forgetfulness INTEGER,            -- Forgetfulness flag
                
                -- Medical Records
                RAW_Diagnosis INTEGER,                -- Original diagnosis code
                RAW_DoctorInCharge TEXT,             -- Attending physician
                
                -- PROCESSED FEATURES (after preprocessing for model input)
                -- These are the cleaned, normalized features used by the ML model
                FunctionalAssessment REAL,           -- Processed functional score
                ADL REAL,                           -- Processed ADL score
                MMSE REAL,                          -- Processed MMSE score
                Age REAL,                           -- Processed age
                DifficultyCompletingTasks REAL,     -- Processed task difficulty
                PersonalityChanges REAL,            -- Processed personality changes
                MemoryComplaints REAL,              -- Processed memory complaints
                Forgetfulness REAL,                 -- Processed forgetfulness
                BehavioralProblems REAL,            -- Processed behavioral problems
                Confusion REAL,                      -- Processed confusion score
                Disorientation REAL,            -- Processed disorientation score
                SleepQuality REAL,         -- Processed sleep quality
                Diabetes REAL               -- Processed diabetes flag
            )
        ''')
        # ====================================================================
        # TABLE 3: BATCH PREDICTIONS (IMAGE-BASED ANALYSIS)
        # ====================================================================
        # Stores results from batch processing of brain MRI images
        # Includes class probabilities for all diagnostic categories
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS batch_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Patient_ID TEXT,                        -- Patient identifier (extracted from filename)
                Filename TEXT NOT NULL,                 -- Original image filename
                Predicted_Class TEXT NOT NULL,          -- Predicted diagnosis class
                Confidence REAL NOT NULL,               -- Prediction confidence
                
                -- Individual class probabilities for 4-class Alzheimer's classification
                Mild_Demented_Probability REAL,        -- Probability of mild dementia
                Moderate_Demented_Probability REAL,     -- Probability of moderate dementia
                Non_Demented_Probability REAL,          -- Probability of no dementia
                Very_Mild_Demented_Probability REAL,    -- Probability of very mild dementia
                
                model_name TEXT,                        -- Model identifier
                model_version TEXT                      -- Model version
            )
        ''')

        # ====================================================================
        # TABLE 4: BRAIN REGION ANALYSIS
        # ====================================================================
        # Stores detailed brain region importance scores from Score-CAM and LIME analysis
        # This is crucial for understanding which brain areas drive predictions
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS batch_region (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Patient_ID TEXT,                        -- Patient identifier
                Filename TEXT NOT NULL,                 -- Source image filename
                Brain_Region TEXT NOT NULL,             -- Anatomical brain region name
                
                -- IMPORTANCE SCORES FROM DIFFERENT ANALYSIS METHODS
                ScoreCAM_Importance_Score REAL,         -- Score-CAM attention score (0-1)
                ScoreCAM_Importance_Percentage REAL,    -- Score-CAM score as percentage
                
                -- Note: LIME scores would be added here in future versions:
                -- LIME_Importance_Score REAL,
                -- LIME_Importance_Percentage REAL,
                -- Combined_Importance_Score REAL,
                
                -- ANATOMICAL STATISTICS
                Region_Area_Pixels INTEGER,             -- Region size in pixels
                Region_Area_Percentage REAL,            -- Region as % of total brain
                
                -- ANALYSIS METADATA
                analysis_method TEXT,                   -- Method used ('score_cam', 'lime', 'combined')
                model_name TEXT,                        -- Model identifier
                model_version TEXT,                     -- Model version
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP  -- Analysis timestamp
            )
        ''')
        
        # ====================================================================
        # TABLE 5: STORED MEDICAL IMAGES
        # ====================================================================
        # Enhanced metadata storage for medical imaging files
        # Tracks various visualization types and their importance scores
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS stored_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,               -- Patient identifier
                filename TEXT NOT NULL,                 -- Image filename
                image_type TEXT NOT NULL,               -- Type of image (see _extract_image_type)
                file_path TEXT NOT NULL,                -- Full file path
                file_size INTEGER,                      -- File size in bytes
                image_dimensions TEXT,                  -- Image dimensions (e.g., "331x331")
                model_name TEXT,                        -- Model used for analysis
                model_version TEXT,                     -- Model version
                
                -- ENHANCED METADATA FOR MEDICAL IMAGING
                analysis_type TEXT,                     -- Analysis method ('score_cam', 'lime', etc.)
                region_name TEXT,                       -- Specific brain region (if applicable)
                importance_score REAL,                  -- Importance score for this visualization
                
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, filename, image_type)  -- Prevent duplicate entries
            )
        ''')
        
        # ====================================================================
        # TABLE 6: COMPLETE ANALYSIS SESSIONS
        # ====================================================================
        # Tracks complete analysis sessions for comprehensive result management
        # Allows reconstruction of entire analysis workflows
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,        -- Unique session identifier (UUID)
                patient_id TEXT NOT NULL,               -- Patient being analyzed
                analysis_type TEXT NOT NULL,            -- Type of analysis performed
                predicted_class TEXT,                   -- Overall prediction
                confidence REAL,                        -- Overall confidence
                analysis_time REAL,                     -- Time taken for analysis (seconds)
                model_name TEXT,                        -- Model identifier
                model_version TEXT,                     -- Model version
                metadata_json TEXT,                     -- Additional data as JSON string
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ====================================================================
        # DATABASE PERFORMANCE OPTIMIZATION
        # ====================================================================
        # Create comprehensive indexes for fast query performance
        # Essential for medical research applications with large datasets
        
        
        # Create essential indexes only
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_global_feature ON global_importance(Feature)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_individual_patient ON individual_predictions(Patient_ID)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_batch_patient ON batch_predictions(Patient_ID)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_region_patient ON batch_region(Patient_ID)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_images_patient ON stored_images(patient_id)')
        
        self.conn.commit()
    
    def _extract_patient_id_from_filename(self, filename):
        """Extract patient ID from filename"""
        import re
        base_name = Path(filename).stem
        pattern = r'^(patient_\d+)'
        match = re.match(pattern, base_name, re.IGNORECASE)
        return match.group(1).lower() if match else base_name
    
    def store_global_importance(self, importance_data, model_name='CatBoost', model_version=None):
        """Store global feature importance data (used by CSV upload)"""
        if hasattr(importance_data, 'to_dict'):
            importance_data = importance_data.to_dict('records')
        
        for row in importance_data:
            existing = self.conn.execute(
                "SELECT id FROM global_importance WHERE Feature = ? AND model_name = ? AND model_version = ?",
                (row['Feature'], model_name, model_version)
            ).fetchone()
            
            if existing:
                self.conn.execute('''
                    UPDATE global_importance 
                    SET Mean_Absolute_SHAP = ?, Mean_SHAP = ?, Std_SHAP = ?, 
                        Max_SHAP = ?, Min_SHAP = ?, Importance_Rank = ?
                    WHERE Feature = ? AND model_name = ? AND model_version = ?
                ''', (
                    row['Mean_Absolute_SHAP'], row['Mean_SHAP'], row['Std_SHAP'],
                    row['Max_SHAP'], row['Min_SHAP'], row['Importance_Rank'],
                    row['Feature'], model_name, model_version
                ))
            else:
                self.conn.execute('''
                    INSERT INTO global_importance 
                    (Feature, Mean_Absolute_SHAP, Mean_SHAP, Std_SHAP, Max_SHAP, Min_SHAP, 
                     Importance_Rank, model_name, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['Feature'], row['Mean_Absolute_SHAP'], row['Mean_SHAP'], row['Std_SHAP'],
                    row['Max_SHAP'], row['Min_SHAP'], row['Importance_Rank'],
                    model_name, model_version
                ))
        
        self.conn.commit()
    
    def store_individual_prediction(self, prediction_data, model_name='CatBoost', model_version=None):
        """Store individual prediction data (used by CSV upload)"""
        existing = self.conn.execute(
            "SELECT id FROM individual_predictions WHERE Patient_ID = ? AND model_name = ? AND model_version = ?",
            (prediction_data['Patient_ID'], model_name, model_version)
        ).fetchone()
        
        data = dict(prediction_data)
        data['model_name'] = model_name
        data['model_version'] = model_version
        
        if existing:
            # Update existing record
            update_columns = []
            update_values = []
            
            for key, value in data.items():
                if key not in ['Patient_ID', 'model_name', 'model_version']:
                    update_columns.append(f"{key} = ?")
                    update_values.append(value)
            
            update_values.extend([prediction_data['Patient_ID'], model_name, model_version])
            
            self.conn.execute(f'''
                UPDATE individual_predictions 
                SET {', '.join(update_columns)}
                WHERE Patient_ID = ? AND model_name = ? AND model_version = ?
            ''', update_values)
        else:
            # Insert new record
            columns = list(data.keys())
            placeholders = ', '.join(['?' for _ in columns])
            
            self.conn.execute(f'''
                INSERT INTO individual_predictions ({', '.join(columns)})
                VALUES ({placeholders})
            ''', list(data.values()))
        
        self.conn.commit()
    
    def store_batch_prediction(self, prediction_data, model_name='ImageModel', model_version=None):
        """Store batch prediction data (used by image upload)"""
        if isinstance(prediction_data, dict):
            prediction_data = [prediction_data]
        
        if hasattr(prediction_data, 'to_dict'):
            prediction_data = prediction_data.to_dict('records')
        
        for row in prediction_data:
            patient_id = self._extract_patient_id_from_filename(row['Filename'])
            
            existing = self.conn.execute(
                "SELECT id FROM batch_predictions WHERE Filename = ? AND model_name = ? AND model_version = ?",
                (row['Filename'], model_name, model_version)
            ).fetchone()
            
            if existing:
                self.conn.execute('''
                    UPDATE batch_predictions 
                    SET Patient_ID = ?, Predicted_Class = ?, Confidence = ?, 
                        Mild_Demented_Probability = ?, Moderate_Demented_Probability = ?, 
                        Non_Demented_Probability = ?, Very_Mild_Demented_Probability = ?
                    WHERE Filename = ? AND model_name = ? AND model_version = ?
                ''', (
                    patient_id, row['Predicted_Class'], row['Confidence'], 
                    row.get('Mild_Demented_Probability'), row.get('Moderate_Demented_Probability'),
                    row.get('Non_Demented_Probability'), row.get('Very_Mild_Demented_Probability'),
                    row['Filename'], model_name, model_version
                ))
            else:
                self.conn.execute('''
                    INSERT INTO batch_predictions 
                    (Patient_ID, Filename, Predicted_Class, Confidence, 
                     Mild_Demented_Probability, Moderate_Demented_Probability, 
                     Non_Demented_Probability, Very_Mild_Demented_Probability,
                     model_name, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient_id, row['Filename'], row['Predicted_Class'], 
                    row['Confidence'], row.get('Mild_Demented_Probability'), 
                    row.get('Moderate_Demented_Probability'), row.get('Non_Demented_Probability'),
                    row.get('Very_Mild_Demented_Probability'), model_name, model_version
                ))
        
        self.conn.commit()
    
    def store_batch_region(self, region_data, model_name='ImageModel', model_version=None):
        """Store brain region analysis data (used by image upload)"""
        if isinstance(region_data, dict):
            region_data = [region_data]
        
        if hasattr(region_data, 'to_dict'):
            region_data = region_data.to_dict('records')
        
        for row in region_data:
            patient_id = self._extract_patient_id_from_filename(row['Filename'])
            
            scorecam_score = row.get('ScoreCAM_Importance_Score', 0)
            scorecam_percentage = row.get('ScoreCAM_Importance_Percentage', 0)
            region_area_pixels = row.get('Region_Area_Pixels')
            region_area_percentage = row.get('Region_Area_Percentage')
            analysis_method = row.get('analysis_method', 'score_cam')
            
            existing = self.conn.execute(
                "SELECT id FROM batch_region WHERE Filename = ? AND Brain_Region = ? AND model_name = ? AND model_version = ?",
                (row['Filename'], row['Brain_Region'], model_name, model_version)
            ).fetchone()
            
            if existing:
                self.conn.execute('''
                    UPDATE batch_region 
                    SET Patient_ID = ?, ScoreCAM_Importance_Score = ?, ScoreCAM_Importance_Percentage = ?,
                        Region_Area_Pixels = ?, Region_Area_Percentage = ?, analysis_method = ?
                    WHERE Filename = ? AND Brain_Region = ? AND model_name = ? AND model_version = ?
                ''', (
                    patient_id, scorecam_score, scorecam_percentage,
                    region_area_pixels, region_area_percentage, analysis_method,
                    row['Filename'], row['Brain_Region'], model_name, model_version
                ))
            else:
                self.conn.execute('''
                    INSERT INTO batch_region 
                    (Patient_ID, Filename, Brain_Region, ScoreCAM_Importance_Score, ScoreCAM_Importance_Percentage,
                     Region_Area_Pixels, Region_Area_Percentage, analysis_method, model_name, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient_id, row['Filename'], row['Brain_Region'],
                    scorecam_score, scorecam_percentage,
                    region_area_pixels, region_area_percentage,
                    analysis_method, model_name, model_version
                ))
        
        self.conn.commit()
    
    def store_image_with_metadata(self, image_path, patient_id, image_type, 
                                 model_name=None, model_version=None,
                                 analysis_type=None, region_name=None, 
                                 importance_score=None):
        """Store image file with metadata (used by image upload)"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Generate filename
        if region_name and 'region' in image_type:
            filename = f"{patient_id}_region_{region_name.lower()}.png"
        elif image_type == 'original':
            filename = f"{patient_id}_original.png"
        elif image_type == 'brain_mask':
            filename = f"{patient_id}_brain_mask.png"
        elif image_type == 'scorecam_overlay':
            filename = f"{patient_id}_scorecam_overlay.png"
        elif image_type == 'scorecam_heatmap':
            filename = f"{patient_id}_scorecam_heatmap.png"
        else:
            filename = f"{patient_id}_{image_type}.png"
        
        # Create patient directory
        patient_dir = self.images_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        # Copy image
        dest_path = patient_dir / filename
        shutil.copy2(image_path, dest_path)
        
        # Get metadata
        file_size = dest_path.stat().st_size
        try:
            with Image.open(dest_path) as img:
                dimensions = f"{img.width}x{img.height}"
        except Exception:
            dimensions = "unknown"
        
        # Store in database
        existing = self.conn.execute(
            "SELECT id FROM stored_images WHERE patient_id = ? AND filename = ? AND image_type = ?",
            (patient_id, filename, image_type)
        ).fetchone()
        
        if existing:
            self.conn.execute('''
                UPDATE stored_images 
                SET file_path = ?, file_size = ?, image_dimensions = ?, 
                    model_name = ?, model_version = ?, analysis_type = ?,
                    region_name = ?, importance_score = ?
                WHERE patient_id = ? AND filename = ? AND image_type = ?
            ''', (
                str(dest_path), file_size, dimensions, 
                model_name, model_version, analysis_type,
                region_name, importance_score,
                patient_id, filename, image_type
            ))
        else:
            self.conn.execute('''
                INSERT INTO stored_images 
                (patient_id, filename, image_type, file_path, file_size, 
                image_dimensions, model_name, model_version, analysis_type,
                region_name, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id, filename, image_type, str(dest_path), file_size,
                dimensions, model_name, model_version, analysis_type,
                region_name, importance_score
            ))
        
        self.conn.commit()
        return str(dest_path)
    
    def get_individual_predictions(self, patient_id=None, model_name=None, model_version=None, limit=None):
        """Get individual prediction data"""
        query = "SELECT * FROM individual_predictions"
        params = []
        
        conditions = []
        if patient_id:
            conditions.append("Patient_ID = ?")
            params.append(patient_id)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY id DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_global_importance(self, model_name=None, model_version=None, top_n=None):
        """Get global importance data"""
        query = "SELECT * FROM global_importance"
        params = []
        
        conditions = []
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY Importance_Rank"
        
        if top_n:
            query += f" LIMIT {top_n}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    

    def get_batch_predictions(self, filename=None, patient_id=None, predicted_class=None, 
                            model_name=None, model_version=None, limit=None):
        """Get batch prediction data"""
        query = "SELECT * FROM batch_predictions"
        params = []
        
        conditions = []
        if filename:
            conditions.append("Filename = ?")
            params.append(filename)
        if patient_id:
            conditions.append("Patient_ID = ?")
            params.append(patient_id)
        if predicted_class:
            conditions.append("Predicted_Class = ?")
            params.append(predicted_class)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY id DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_batch_region(self, filename=None, patient_id=None, brain_region=None,
                        model_name=None, model_version=None, analysis_method=None, 
                        order_by='Combined_Importance_Score', limit=None):
        """
        Get batch region data with enhanced querying options
        
        Args:
            order_by: Field to order by ('Combined_Importance_Score', 'ScoreCAM_Importance_Score', 
                     'LIME_Importance_Score', 'Region_Area_Percentage')
        """
        query = "SELECT * FROM batch_region"
        params = []
        
        conditions = []
        if filename:
            conditions.append("Filename = ?")
            params.append(filename)
        if patient_id:
            conditions.append("Patient_ID = ?")
            params.append(patient_id)
        if brain_region:
            conditions.append("Brain_Region = ?")
            params.append(brain_region)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        if analysis_method:
            conditions.append("analysis_method = ?")
            params.append(analysis_method)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Validate order_by field
        valid_order_fields = [
             'ScoreCAM_Importance_Score', 
             'Region_Area_Percentage', 'created_timestamp'
        ]
        if order_by not in valid_order_fields:
            order_by = 'ScoreCAM_Importance_Score'
        
        query += f" ORDER BY {order_by} DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def get_stored_images(self, patient_id=None, image_type=None, model_name=None, 
                         model_version=None, limit=None):
        """Get stored image metadata"""
        query = "SELECT * FROM stored_images"
        params = []
        
        conditions = []
        if patient_id:
            conditions.append("patient_id = ?")
            params.append(patient_id)
        if image_type:
            conditions.append("image_type = ?")
            params.append(image_type)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY id DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Create cursor and execute
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Get column names from cursor
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to list of dictionaries
        return [dict(zip(columns, row)) for row in results]
    
    
    def get_patient_images(self, patient_id):
        """Get all images for a specific patient"""
        return self.get_stored_images(patient_id=patient_id)
    
    
    def close(self):
        """Close database connection (used by upload code)"""
        self.conn.close()
