# shap_utils.py
"""
Improved SHAP analysis utilities for Alzheimer's disease prediction model.
Provides functions to generate detailed SHAP-based analysis, including global
feature importance, individual predictions with SHAP values, high-risk patient
analysis, feature interactions, and demographic insights.
"""

import pandas as pd
import numpy as np


def create_shap_analysis_results(shap_values, predictions, probabilities, feature_names, 
                               actual_labels=None, data=None, model_performance=None):
    """
    Generate comprehensive SHAP analysis results for further inspection or Excel export.

    Parameters:
        shap_values (np.ndarray): Array of SHAP values for each sample and feature.
        predictions (np.ndarray): Model-predicted labels (0/1).
        probabilities (np.ndarray): Model-predicted probabilities for positive class.
        feature_names (list): Names of features corresponding to SHAP values.
        actual_labels (np.ndarray, optional): True labels (0/1), if available.
        data (pd.DataFrame, optional): Original input data with patient info.
        model_performance (dict, optional): Dictionary with performance metrics.
    
    Returns:
        dict: Dictionary containing multiple DataFrames with SHAP and prediction analyses.
    """
    
    # ------------------------------
    # Validate input dimensions
    # ------------------------------
    if len(shap_values) != len(predictions) or len(predictions) != len(probabilities):
        raise ValueError("Mismatch in length of shap_values, predictions, and probabilities")
    
    if len(feature_names) != shap_values.shape[1]:
        raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match SHAP values features ({shap_values.shape[1]})")
    
    print(f"Processing {len(predictions)} predictions with {len(feature_names)} features")
    
    # ------------------------------
    # 1. Global feature importance
    # ------------------------------
    # Compute mean, std, max, min, and absolute SHAP for each feature
    global_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': np.mean(np.abs(shap_values), axis=0),
        'Mean_SHAP': np.mean(shap_values, axis=0),
        'Std_SHAP': np.std(shap_values, axis=0),
        'Max_SHAP': np.max(shap_values, axis=0),
        'Min_SHAP': np.min(shap_values, axis=0)
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    
    # Add ranking
    global_importance['Importance_Rank'] = range(1, len(global_importance) + 1)

    # ------------------------------
    # 2. Individual patient predictions with SHAP
    # ------------------------------
    # Determine Patient IDs, handling multiple possible column names
    patient_ids = None
    if data is not None:
        id_columns = ['PatientID', 'Patient_ID', 'patient_id', 'ID', 'id']
        for col in id_columns:
            if col in data.columns:
                patient_ids = data[col].values
                print(f"Using {col} column for Patient IDs")
                break
    
    # Fallback: generate sequential IDs
    if patient_ids is None:
        patient_ids = [f"P{i+4751}" for i in range(len(predictions))]
        print("Warning: No Patient ID column found. Generated sequential IDs starting from P4751")
    
    # Ensure patient_ids length matches predictions
    if len(patient_ids) != len(predictions):
        print(f"Warning: Patient ID count ({len(patient_ids)}) != predictions count ({len(predictions)})")
        patient_ids = patient_ids[:len(predictions)]
        if len(patient_ids) < len(predictions):
            missing_count = len(predictions) - len(patient_ids)
            last_id = int(patient_ids[-1][1:]) if patient_ids else 4750
            patient_ids.extend([f"P{last_id + i + 1}" for i in range(missing_count)])
    
    # Create DataFrame with predictions
    individual_predictions = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Predicted_Diagnosis': predictions,
        'Prediction_Probability': probabilities,
        'Prediction_Confidence': np.abs(probabilities - 0.5) * 2  # Confidence from 0-1
    })
    
    # Add demographic info if available (prefix RAW_ to avoid conflicts)
    if data is not None:
        exclude_cols = ['PatientID', 'Patient_ID', 'patient_id', 'ID', 'id']
        for col in data.columns:
            if col not in exclude_cols:
                raw_col_name = f'RAW_{col}'
                individual_predictions[raw_col_name] = data[col].values[:len(predictions)]
                print(f"Added {raw_col_name} from original data")
    
    # Add actual labels if available
    if actual_labels is not None:
        individual_predictions['Actual_Diagnosis'] = actual_labels
        individual_predictions['Correct_Prediction'] = (actual_labels == predictions).astype(int)
        individual_predictions['Prediction_Error'] = np.abs(actual_labels - predictions)
    
    # Append SHAP values for each feature
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    individual_predictions = pd.concat([individual_predictions, shap_df], axis=1)

    # ------------------------------
    # 3. Feature contributions by predicted diagnosis
    # ------------------------------
    positive_indices = predictions == 1
    negative_indices = predictions == 0

    positive_cases = shap_values[positive_indices]
    negative_cases = shap_values[negative_indices]

    diagnosis_comparison = pd.DataFrame({
        'Feature': feature_names,
        'Positive_Count': np.sum(positive_indices),
        'Negative_Count': np.sum(negative_indices),
        'Positive_Mean_SHAP': np.mean(positive_cases, axis=0) if len(positive_cases) > 0 else np.zeros(len(feature_names)),
        'Negative_Mean_SHAP': np.mean(negative_cases, axis=0) if len(negative_cases) > 0 else np.zeros(len(feature_names)),
        'Positive_Std_SHAP': np.std(positive_cases, axis=0) if len(positive_cases) > 0 else np.zeros(len(feature_names)),
        'Negative_Std_SHAP': np.std(negative_cases, axis=0) if len(negative_cases) > 0 else np.zeros(len(feature_names))
    })
    diagnosis_comparison['Difference'] = diagnosis_comparison['Positive_Mean_SHAP'] - diagnosis_comparison['Negative_Mean_SHAP']
    diagnosis_comparison['Abs_Difference'] = np.abs(diagnosis_comparison['Difference'])
    diagnosis_comparison = diagnosis_comparison.sort_values('Abs_Difference', ascending=False)

    # ------------------------------
    # 4. Model performance summary
    # ------------------------------
    if model_performance is not None and actual_labels is not None:
        # Training scenario: compute detailed metrics
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
            
            test_accuracy = accuracy_score(actual_labels, predictions)
            test_roc_auc = roc_auc_score(actual_labels, probabilities)
            test_avg_precision = average_precision_score(actual_labels, probabilities)
            
            val_accuracy = model_performance.get('CatBoost', {}).get('accuracy', 0)
            val_roc_auc = model_performance.get('CatBoost', {}).get('roc_auc', 0)
            val_avg_precision = model_performance.get('CatBoost', {}).get('avg_precision', 0)
            
            model_summary = pd.DataFrame({
                'Metric': ['Test_Accuracy', 'Test_ROC_AUC', 'Test_Avg_Precision',
                          'Val_Accuracy', 'Val_ROC_AUC', 'Val_Avg_Precision'],
                'Value': [test_accuracy, test_roc_auc, test_avg_precision,
                         val_accuracy, val_roc_auc, val_avg_precision]
            })
        except ImportError:
            print("Warning: sklearn not available for detailed metrics")
            model_summary = _create_basic_summary(predictions, probabilities, data)
    else:
        # Inference scenario: basic summary only
        model_summary = _create_basic_summary(predictions, probabilities, data)

    # ------------------------------
    # 5. Individual model performance table
    # ------------------------------
    if model_performance is not None:
        individual_performance = pd.DataFrame(model_performance).T
        individual_performance.reset_index(inplace=True)
        individual_performance.rename(columns={'index': 'Model'}, inplace=True)
    else:
        individual_performance = pd.DataFrame({
            'Model': ['CatBoost'],
            'Algorithm': ['Gradient Boosting'],
            'Features_Used': [len(feature_names)],
            'Note': ['Inference mode - see training results for detailed performance metrics']
        })

    # ------------------------------
    # 6. High-risk patient SHAP analysis
    # ------------------------------
    high_risk_threshold = 0.7
    high_risk_indices = probabilities >= high_risk_threshold

    if np.sum(high_risk_indices) > 0:
        high_risk_shap = shap_values[high_risk_indices]
        
        high_risk_features_data = {
            'Feature': feature_names,
            'Mean_SHAP_High_Risk': np.mean(high_risk_shap, axis=0),
            'Std_SHAP_High_Risk': np.std(high_risk_shap, axis=0),
            'Max_SHAP_High_Risk': np.max(high_risk_shap, axis=0),
            'Min_SHAP_High_Risk': np.min(high_risk_shap, axis=0),
            'Count_High_Risk_Patients': np.sum(high_risk_indices)
        }
        
        # Add demographics info if available
        if data is not None:
            high_risk_data = data.iloc[high_risk_indices] if len(data) == len(predictions) else None
            if high_risk_data is not None:
                if 'Age' in data.columns:
                    high_risk_features_data['Mean_Age_High_Risk'] = np.mean(high_risk_data['Age'])
                    high_risk_features_data['Age_Range_High_Risk'] = f"{high_risk_data['Age'].min():.0f}-{high_risk_data['Age'].max():.0f}"
                if 'Gender' in data.columns:
                    try:
                        # Support both numeric and string gender
                        if high_risk_data['Gender'].dtype == 'object':
                            male_hr = np.sum(high_risk_data['Gender'].str.lower().isin(['male', 'm']))
                            female_hr = np.sum(high_risk_data['Gender'].str.lower().isin(['female', 'f']))
                        else:
                            male_hr = np.sum(high_risk_data['Gender'] == 1)
                            female_hr = np.sum(high_risk_data['Gender'] == 0)
                        
                        high_risk_features_data['Male_High_Risk_Count'] = male_hr
                        high_risk_features_data['Female_High_Risk_Count'] = female_hr
                    except:
                        print("Warning: Could not process gender data for high-risk analysis")
        
        high_risk_features = pd.DataFrame(high_risk_features_data)
        high_risk_features = high_risk_features.sort_values('Mean_SHAP_High_Risk', key=abs, ascending=False)
    else:
        # No high-risk patients detected
        high_risk_features = pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP_High_Risk': [0] * len(feature_names),
            'Note': ['No high-risk patients found (probability >= 0.7)'] * len(feature_names)
        })

    # ------------------------------
    # 7. Feature interaction analysis (top 10 features)
    # ------------------------------
    top_10_features = global_importance.head(10)['Feature'].tolist()
    
    try:
        top_10_indices = [feature_names.index(feat) for feat in top_10_features]
        
        interaction_matrix = np.zeros((len(top_10_features), len(top_10_features)))
        for i, idx1 in enumerate(top_10_indices):
            for j, idx2 in enumerate(top_10_indices):
                if i != j:
                    # Correlation of SHAP values as proxy for interaction
                    corr = np.corrcoef(shap_values[:, idx1], shap_values[:, idx2])[0, 1]
                    interaction_matrix[i, j] = corr if not np.isnan(corr) else 0

        feature_interactions = pd.DataFrame(
            interaction_matrix,
            index=top_10_features,
            columns=top_10_features
        )
    except Exception as e:
        print(f"Warning: Could not create interaction matrix: {e}")
        feature_interactions = pd.DataFrame()

    # ------------------------------
    # 8. Demographic analysis
    # ------------------------------
    demographic_analysis = None
    if data is not None and len(data) == len(predictions):
        try:
            demographic_analysis = _create_demographic_analysis(data, predictions, probabilities, patient_ids)
        except Exception as e:
            print(f"Warning: Could not create demographic analysis: {e}")

    # Return dictionary with all analysis results
    return {
        'global_importance': global_importance,
        'individual_predictions': individual_predictions,
        'diagnosis_comparison': diagnosis_comparison,
        'model_summary': model_summary,
        'individual_performance': individual_performance,
        'high_risk_features': high_risk_features,
        'feature_interactions': feature_interactions,
        'demographic_analysis': demographic_analysis
    }


# ------------------------------
# Helper functions
# ------------------------------

def _create_basic_summary(predictions, probabilities, data):
    """Create basic summary statistics for inference mode."""
    
    summary_data = {
        'Metric': ['Total_Patients', 'Positive_Predictions', 'Negative_Predictions', 
                  'Positive_Rate', 'Average_Risk_Score', 'High_Risk_Count', 'Medium_Risk_Count', 'Low_Risk_Count'],
        'Value': [
            len(predictions), 
            np.sum(predictions == 1), 
            np.sum(predictions == 0),
            np.mean(predictions),
            np.mean(probabilities),
            np.sum(probabilities >= 0.7),
            np.sum((probabilities >= 0.3) & (probabilities < 0.7)),
            np.sum(probabilities < 0.3)
        ]
    }
    
    # Add demographic info if available
    if data is not None:
        if 'Gender' in data.columns:
            try:
                if data['Gender'].dtype == 'object':
                    male_count = np.sum(data['Gender'].str.lower().isin(['male', 'm']))
                    female_count = np.sum(data['Gender'].str.lower().isin(['female', 'f']))
                else:
                    male_count = np.sum(data['Gender'] == 1)
                    female_count = np.sum(data['Gender'] == 0)
                
                summary_data['Metric'].extend(['Male_Patients', 'Female_Patients'])
                summary_data['Value'].extend([male_count, female_count])
            except:
                print("Warning: Could not process gender data")
        
        if 'Age' in data.columns:
            try:
                summary_data['Metric'].extend(['Mean_Age', 'Min_Age', 'Max_Age', 'Age_Std'])
                summary_data['Value'].extend([
                    np.mean(data['Age']), 
                    np.min(data['Age']), 
                    np.max(data['Age']),
                    np.std(data['Age'])
                ])
            except:
                print("Warning: Could not process age data")
    
    return pd.DataFrame(summary_data)


def _create_demographic_analysis(data, predictions, probabilities, patient_ids):
    """Create detailed demographic analysis grouped by age and gender."""
    
    demographic_df = pd.DataFrame({'Patient_ID': patient_ids})
    
    # Add age and create age brackets
    if 'Age' in data.columns:
        demographic_df['Age'] = data['Age'].values[:len(predictions)]
        age_brackets = pd.cut(
            demographic_df['Age'], 
            bins=[0, 50, 60, 70, 80, 100], 
            labels=['<50', '50-60', '60-70', '70-80', '80+'],
            include_lowest=True
        )
        demographic_df['Age_Bracket'] = age_brackets
    
    # Add gender if available
    if 'Gender' in data.columns:
        demographic_df['Gender'] = data['Gender'].values[:len(predictions)]
    
    # Add predictions and risk scores
    demographic_df['Predicted_Diagnosis'] = predictions
    demographic_df['Risk_Score'] = probabilities
    
    # Aggregate by demographics
    group_cols = []
    if 'Gender' in demographic_df.columns:
        group_cols.append('Gender')
    if 'Age_Bracket' in demographic_df.columns:
        group_cols.append('Age_Bracket')
    
    if group_cols:
        demographic_analysis = demographic_df.groupby(group_cols).agg({
            'Patient_ID': 'count',
            'Predicted_Diagnosis': ['sum', 'mean'],
            'Risk_Score': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Flatten column names
        demographic_analysis.columns = [
            'Patient_Count', 'Positive_Cases', 'Positive_Rate', 
            'Avg_Risk_Score', 'Risk_Score_Std', 'Min_Risk_Score', 'Max_Risk_Score'
        ]
        demographic_analysis.reset_index(inplace=True)
        
        return demographic_analysis
    
    return None
