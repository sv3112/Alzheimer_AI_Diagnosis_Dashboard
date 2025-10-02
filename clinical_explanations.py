# clinical_explanations.py
"""
Clinical explanation and patient narrative generation system for Alzheimer's prediction analysis
"""

import streamlit as st
import numpy as np
import pandas as pd

def generate_patient_recommendations(patient_data):
        """Generate personalized recommendations based on patient profile"""
        
        recommendations = {
            'immediate_actions': [],
            'lifestyle_modifications': [],
            'medical_followup': [],
            'cognitive_interventions': [],
            'monitoring_suggestions': []
        }
        
        # Get raw values for analysis
        raw_cols = [col for col in patient_data.index if col.startswith('RAW_')]
        
        # Risk level assessment
        prob = patient_data['Prediction_Probability']
        risk_level = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.3 else 'Low'
        
        # IMMEDIATE ACTIONS based on risk level
        if risk_level == 'High':
            recommendations['immediate_actions'].extend([
                "🚨 **Urgent**: Schedule comprehensive neurological evaluation within 2 weeks",
                "📋 **Priority**: Initiate cognitive assessment battery (MoCA, Clock Drawing Test)",
                "👥 **Family**: Discuss care planning and support systems with family members"
            ])
        elif risk_level == 'Medium':
            recommendations['immediate_actions'].extend([
                "📅 **Schedule**: Neurological consultation within 1 month",
                "🧠 **Assessment**: Complete detailed cognitive evaluation",
                "📊 **Monitor**: Begin systematic tracking of cognitive changes"
            ])
        else:
            recommendations['immediate_actions'].extend([
                "✅ **Maintain**: Continue current preventive care approach",
                "📅 **Schedule**: Annual cognitive screening",
                "🎯 **Focus**: Optimize protective factors"
            ])
        
        # COGNITIVE INTERVENTIONS based on MMSE and cognitive symptoms
        if 'RAW_MMSE' in raw_cols:
            mmse_score = patient_data['RAW_MMSE']
            if pd.notna(mmse_score):
                if mmse_score < 24:
                    recommendations['cognitive_interventions'].extend([
                        "🧩 **Cognitive Training**: Enroll in structured cognitive rehabilitation program",
                        "📚 **Memory Strategies**: Learn compensatory memory techniques",
                        "👨‍⚕️ **Specialist**: Consider referral to neuropsychologist",
                        "🎵 **Music Therapy**: Explore music-based cognitive interventions"
                    ])
                elif mmse_score < 27:
                    recommendations['cognitive_interventions'].extend([
                        "🧠 **Brain Training**: Engage in cognitively stimulating activities (puzzles, reading)",
                        "🎓 **Learning**: Pursue new skills or hobbies to maintain cognitive reserve",
                        "💭 **Mindfulness**: Practice meditation and mindfulness exercises"
                    ])
                else:
                    recommendations['cognitive_interventions'].extend([
                        "📖 **Intellectual Engagement**: Maintain challenging mental activities",
                        "🎯 **Cognitive Reserve**: Continue lifelong learning activities"
                    ])
        
        # Check for memory complaints and cognitive symptoms
        cognitive_symptoms = ['RAW_MemoryComplaints', 'RAW_Confusion', 'RAW_Disorientation', 
                            'RAW_DifficultyCompletingTasks', 'RAW_Forgetfulness']
        
        symptom_count = 0
        for symptom in cognitive_symptoms:
            if symptom in raw_cols and pd.notna(patient_data[symptom]) and patient_data[symptom] == 1:
                symptom_count += 1
        
        if symptom_count > 2:
            recommendations['cognitive_interventions'].extend([
                "📝 **Memory Aids**: Implement external memory supports (calendars, reminder apps)",
                "🏠 **Environment**: Optimize home environment for navigation and safety",
                "📱 **Technology**: Consider assistive technology for daily tasks"
            ])
        
        # LIFESTYLE MODIFICATIONS
        
        # Physical Activity
        if 'RAW_PhysicalActivity' in raw_cols:
            activity_level = patient_data['RAW_PhysicalActivity']
            if pd.notna(activity_level):
                if activity_level < 2.5:
                    recommendations['lifestyle_modifications'].extend([
                        "🏃 **Exercise**: Aim for 150 minutes of moderate aerobic activity weekly",
                        "🚶 **Walking**: Start with 30-minute daily walks",
                        "💪 **Strength**: Include resistance training 2-3 times per week",
                        "🏊 **Low-Impact**: Consider swimming or water aerobics"
                    ])
                elif activity_level < 5:
                    recommendations['lifestyle_modifications'].extend([
                        "⬆️ **Increase**: Gradually increase exercise intensity and duration",
                        "🎯 **Variety**: Include both aerobic and strength training"
                    ])
                else:
                    recommendations['lifestyle_modifications'].append(
                        "✅ **Maintain**: Continue excellent physical activity routine"
                    )
        
        # Diet Quality
        if 'RAW_DietQuality' in raw_cols:
            diet_score = patient_data['RAW_DietQuality']
            if pd.notna(diet_score):
                if diet_score < 6:
                    recommendations['lifestyle_modifications'].extend([
                        "🥗 **Mediterranean Diet**: Adopt Mediterranean-style eating pattern",
                        "🐟 **Omega-3**: Include fatty fish 2-3 times per week",
                        "🥜 **Nuts**: Daily handful of nuts and seeds",
                        "🫐 **Berries**: Include blueberries and other antioxidant-rich foods",
                        "🍷 **Moderate**: Limit alcohol consumption",
                        "🧂 **Reduce**: Minimize processed foods and excess sodium"
                    ])
                elif diet_score < 8:
                    recommendations['lifestyle_modifications'].extend([
                        "🎯 **Optimize**: Fine-tune current diet with more anti-inflammatory foods",
                        "🥦 **Vegetables**: Increase variety of colorful vegetables"
                    ])
                else:
                    recommendations['lifestyle_modifications'].append(
                        "✅ **Maintain**: Continue excellent dietary habits"
                    )
        
        # Sleep Quality
        if 'RAW_SleepQuality' in raw_cols:
            sleep_score = patient_data['RAW_SleepQuality']
            if pd.notna(sleep_score):
                if sleep_score < 6:
                    recommendations['lifestyle_modifications'].extend([
                        "😴 **Sleep Hygiene**: Establish consistent sleep schedule (7-9 hours)",
                        "🌙 **Environment**: Create dark, cool, quiet sleeping environment",
                        "📱 **Digital Detox**: Avoid screens 1 hour before bedtime",
                        "☕ **Caffeine**: Limit caffeine after 2 PM",
                        "🧘 **Relaxation**: Practice relaxation techniques before bed"
                    ])
                elif sleep_score < 8:
                    recommendations['lifestyle_modifications'].extend([
                        "🎯 **Optimize**: Fine-tune sleep routine for better quality",
                        "📊 **Track**: Monitor sleep patterns with sleep diary"
                    ])
        
        # Smoking
        if 'RAW_Smoking' in raw_cols and pd.notna(patient_data['RAW_Smoking']) and patient_data['RAW_Smoking'] == 1:
            recommendations['lifestyle_modifications'].extend([
                "🚭 **URGENT**: Quit smoking immediately - major risk factor",
                "📞 **Support**: Contact smoking cessation helpline",
                "💊 **Aids**: Discuss nicotine replacement therapy with doctor",
                "👥 **Program**: Join smoking cessation support group"
            ])
        
        # MEDICAL FOLLOW-UP
        
        # Cardiovascular Disease
        if 'RAW_CardiovascularDisease' in raw_cols and pd.notna(patient_data['RAW_CardiovascularDisease']) and patient_data['RAW_CardiovascularDisease'] == 1:
            recommendations['medical_followup'].extend([
                "❤️ **Cardiology**: Optimize cardiovascular disease management",
                "💊 **Medications**: Ensure optimal blood pressure and cholesterol control",
                "🩺 **Monitoring**: Regular cardiovascular risk assessment"
            ])
        
        # Diabetes
        if 'RAW_Diabetes' in raw_cols and pd.notna(patient_data['RAW_Diabetes']) and patient_data['RAW_Diabetes'] == 1:
            recommendations['medical_followup'].extend([
                "🍬 **Diabetes**: Optimize blood glucose control (HbA1c < 7%)",
                "👨‍⚕️ **Endocrinology**: Regular diabetes management review",
                "👁️ **Screening**: Annual diabetic eye and kidney function checks"
            ])
        
        # Depression
        if 'RAW_Depression' in raw_cols and pd.notna(patient_data['RAW_Depression']) and patient_data['RAW_Depression'] == 1:
            recommendations['medical_followup'].extend([
                "😔 **Mental Health**: Address depression with psychiatrist/psychologist",
                "💊 **Treatment**: Optimize antidepressant therapy if needed",
                "🧘 **Therapy**: Consider cognitive-behavioral therapy (CBT)"
            ])
        
        # Hypertension
        if 'RAW_Hypertension' in raw_cols and pd.notna(patient_data['RAW_Hypertension']) and patient_data['RAW_Hypertension'] == 1:
            recommendations['medical_followup'].extend([
                "🩸 **Blood Pressure**: Maintain target BP < 130/80 mmHg",
                "💊 **Medications**: Regular medication review and adjustment",
                "🏠 **Home Monitoring**: Consider home blood pressure monitoring"
            ])
        
        # Blood Pressure Values
        if 'RAW_SystolicBP' in raw_cols and 'RAW_DiastolicBP' in raw_cols:
            systolic = patient_data['RAW_SystolicBP']
            diastolic = patient_data['RAW_DiastolicBP']
            if pd.notna(systolic) and pd.notna(diastolic):
                if systolic >= 140 or diastolic >= 90:
                    recommendations['medical_followup'].extend([
                        "🚨 **Hypertension**: Address elevated blood pressure urgently",
                        "💊 **Antihypertensive**: Consider medication adjustment"
                    ])
        
        # Cholesterol
        cholesterol_high = False
        if 'RAW_CholesterolTotal' in raw_cols and pd.notna(patient_data['RAW_CholesterolTotal']) and patient_data['RAW_CholesterolTotal'] > 200:
            cholesterol_high = True
        if 'RAW_CholesterolLDL' in raw_cols and pd.notna(patient_data['RAW_CholesterolLDL']) and patient_data['RAW_CholesterolLDL'] > 100:
            cholesterol_high = True
        
        if cholesterol_high:
            recommendations['medical_followup'].extend([
                "🧪 **Cholesterol**: Address elevated cholesterol levels",
                "💊 **Statins**: Consider statin therapy if appropriate",
                "🥗 **Diet**: Implement cholesterol-lowering diet"
            ])
        
        # Functional Assessment
        if 'RAW_FunctionalAssessment' in raw_cols:
            func_score = patient_data['RAW_FunctionalAssessment']
            if pd.notna(func_score) and func_score < 6:
                recommendations['medical_followup'].extend([
                    "🏥 **Occupational Therapy**: Referral for functional assessment",
                    "🏠 **Home Safety**: Evaluate home safety modifications",
                    "👥 **Support Services**: Explore available support services"
                ])
        
        # ADL Assessment
        if 'RAW_ADL' in raw_cols:
            adl_score = patient_data['RAW_ADL']
            if pd.notna(adl_score) and adl_score < 6:
                recommendations['medical_followup'].extend([
                    "🛁 **ADL Support**: Assess need for daily living assistance",
                    "🏠 **Home Modifications**: Consider adaptive equipment",
                    "👥 **Caregiver**: Explore caregiver support options"
                ])
        
        # MONITORING SUGGESTIONS
        
        # Based on risk level
        if risk_level == 'High':
            recommendations['monitoring_suggestions'].extend([
                "📊 **Frequent Monitoring**: Cognitive assessment every 3 months",
                "📝 **Symptom Tracking**: Daily symptom and mood tracking",
                "🏥 **Regular Visits**: Monthly healthcare provider visits",
                "📱 **Technology**: Consider wearable devices for activity/sleep monitoring"
            ])
        elif risk_level == 'Medium':
            recommendations['monitoring_suggestions'].extend([
                "📊 **Regular Monitoring**: Cognitive assessment every 6 months",
                "📝 **Symptom Tracking**: Weekly symptom review",
                "🏥 **Regular Visits**: Quarterly healthcare provider visits"
            ])
        else:
            recommendations['monitoring_suggestions'].extend([
                "📊 **Annual Monitoring**: Yearly comprehensive cognitive assessment",
                "📝 **Self-Monitoring**: Monthly self-assessment of cognitive function",
                "🏥 **Regular Visits**: Bi-annual healthcare provider visits"
            ])
        
        # Age-specific recommendations
        if 'RAW_Age' in raw_cols:
            age = patient_data['RAW_Age']
            if pd.notna(age):
                if age >= 75:
                    recommendations['monitoring_suggestions'].extend([
                        "👴 **Age-Related**: Increased monitoring due to advanced age",
                        "🏠 **Safety**: Regular home safety assessments",
                        "👥 **Social**: Maintain social connections and activities"
                    ])
        
        # Family History
        if 'RAW_FamilyHistoryAlzheimers' in raw_cols and pd.notna(patient_data['RAW_FamilyHistoryAlzheimers']) and patient_data['RAW_FamilyHistoryAlzheimers'] == 1:
            recommendations['monitoring_suggestions'].extend([
                "👨‍👩‍👧‍👦 **Genetic**: Consider genetic counseling consultation",
                "📊 **Enhanced Monitoring**: More frequent cognitive assessments due to family history",
                "👥 **Family Education**: Educate family members about early warning signs"
            ])
        
        return recommendations

def calculate_intervention_recommendations(modifications, shap_values):
        """Generate personalized intervention recommendations based on modifications"""
        recommendations = []
        

        # Physical Activity Recommendations
        if 'PhysicalActivity' in modifications:
            activity_level = modifications['PhysicalActivity']
            if activity_level >= 7:
                recommendations.append({
                    'category': 'Physical Activity',
                    'priority': 'High',
                    'recommendation': 'Maintain your current high activity level with 150+ minutes of moderate exercise per week.',
                    'impact': 'Has a strong positive effect in reducing Alzheimer\'s risk.'
                })
            else:
                recommendations.append({
                    'category': 'Physical Activity',
                    'priority': 'Critical',
                    'recommendation': 'Gradually increase physical activity to at least 30 minutes daily.',
                    'impact': 'Improving activity could significantly reduce your risk.'
                })

        # Cognitive Health Recommendations
        if 'MMSE' in modifications:
            mmse_score = modifications['MMSE']
            if mmse_score < 24:
                recommendations.append({
                    'category': 'Cognitive Health',
                    'priority': 'Critical',
                    'recommendation': 'Immediate cognitive assessment and intervention recommended.',
                    'impact': 'Early intervention is crucial at this stage.'
                })

        
        # Age-based Recommendations
        if 'Age' in modifications:
            age = modifications['Age']
            if age >= 65:
                recommendations.append({
                    'category': 'Age',
                    'priority': 'Moderate',
                    'recommendation': 'Increase monitoring frequency and maintain brain-healthy lifestyle habits.',
                    'impact': 'Older age increases risk, proactive management is beneficial.'
                })

        # Functional Assessment Recommendations
        if 'FunctionalAssessment' in modifications:
            func_score = modifications['FunctionalAssessment']
            if func_score < 6:  # assuming lower score means more impairment
                recommendations.append({
                    'category': 'Functional Assessment',
                    'priority': 'High',
                    'recommendation': 'Initiate supportive therapies to maintain daily functioning.',
                    'impact': 'Helps preserve independence and quality of life.'
                })

        # Activities of Daily Living (ADL) Recommendations
        if 'ADL' in modifications:
            adl_score = modifications['ADL']
            if adl_score < 6:  # again assuming lower is worse
                recommendations.append({
                    'category': 'Activities of Daily Living',
                    'priority': 'Critical',
                    'recommendation': 'Consider caregiver support and occupational therapy interventions.',
                    'impact': 'Essential for safety and maintaining functional independence.'
                })
        symptom_threshold = 1  # assuming binary or severity scale where 1 means presence/severity

        symptom_map = {
            'MemoryComplaints': "Address memory concerns with targeted cognitive exercises and clinical follow-up.",
            'Forgetfulness': "Implement memory aids and schedule regular cognitive evaluations.",
            'Confusion': "Immediate clinical assessment recommended to identify underlying causes.",
            'Disorientation': "Monitor closely and consider neurological evaluation.",
            'PersonalityChanges': "Consult a specialist for behavioral assessment and support planning.",
            'DifficultyCompletingTasks': "Occupational therapy evaluation advised to maintain independence.",
            'BehavioralProblems': "Behavioral interventions and caregiver support recommended.",
        }

        for symptom, advice in symptom_map.items():
            if modifications.get(symptom, 0) >= symptom_threshold:
                recommendations.append({
                    'category': 'Symptoms',
                    'priority': 'High',
                    'recommendation': advice,
                    'impact': 'Early attention can improve patient outcomes and quality of life.'
                })

            return recommendations

# Clinical knowledge base
BRAIN_REGION_KNOWLEDGE = {
    'Hippocampus': {
        'function': 'Memory formation and consolidation',
        'ad_significance': 'Early target in Alzheimer\'s disease',
        'symptoms': 'Short-term memory loss, difficulty forming new memories',
        'normal_range': (0.0, 0.3),
        'concerning_range': (0.3, 0.7),
        'high_risk_range': (0.7, 1.0),
        'clinical_terms': ['hippocampal atrophy', 'mesial temporal sclerosis', 'memory circuit dysfunction']
    },
    'Frontal': {
        'function': 'Executive function, planning, decision-making',
        'ad_significance': 'Affects judgment and personality in mid-stage AD',
        'symptoms': 'Poor judgment, personality changes, difficulty planning',
        'normal_range': (0.0, 0.25),
        'concerning_range': (0.25, 0.6),
        'high_risk_range': (0.6, 1.0),
        'clinical_terms': ['frontal lobe atrophy', 'executive dysfunction', 'behavioral variant']
    },
    'Temporal': {
        'function': 'Language processing, auditory processing, memory',
        'ad_significance': 'Language difficulties and semantic memory loss',
        'symptoms': 'Word-finding difficulties, language comprehension issues',
        'normal_range': (0.0, 0.3),
        'concerning_range': (0.3, 0.65),
        'high_risk_range': (0.65, 1.0),
        'clinical_terms': ['temporal lobe atrophy', 'semantic memory loss', 'language circuit dysfunction']
    },
    'Parietal': {
        'function': 'Spatial processing, attention, integration',
        'ad_significance': 'Spatial disorientation and attention deficits',
        'symptoms': 'Getting lost, difficulty with spatial tasks, attention problems',
        'normal_range': (0.0, 0.28),
        'concerning_range': (0.28, 0.62),
        'high_risk_range': (0.62, 1.0),
        'clinical_terms': ['parietal atrophy', 'spatial disorientation', 'attention network dysfunction']
    },
    'Occipital': {
        'function': 'Visual processing and perception',
        'ad_significance': 'Visual processing difficulties in advanced AD',
        'symptoms': 'Visual perception problems, difficulty recognizing objects',
        'normal_range': (0.0, 0.25),
        'concerning_range': (0.25, 0.55),
        'high_risk_range': (0.55, 1.0),
        'clinical_terms': ['occipital atrophy', 'visual processing deficit', 'posterior cortical atrophy']
    },
    'Ventricular': {
        'function': 'CSF circulation and brain structure support',
        'ad_significance': 'Enlarged ventricles indicate brain volume loss',
        'symptoms': 'Often asymptomatic but indicates brain atrophy',
        'normal_range': (0.0, 0.2),
        'concerning_range': (0.2, 0.5),
        'high_risk_range': (0.5, 1.0),
        'clinical_terms': ['ventricular enlargement', 'brain atrophy', 'central volume loss']
    }
}

def generate_region_explanation(region_name, importance_score, predicted_class, confidence):
    """Generate detailed explanation for a specific brain region"""
    
    if region_name not in BRAIN_REGION_KNOWLEDGE:
        return f"Analysis available for {region_name} (Score: {importance_score:.3f})"
    
    region_info = BRAIN_REGION_KNOWLEDGE[region_name]
    score = float(importance_score) if hasattr(importance_score, 'iloc') else importance_score
    
    # Determine risk level
    if score <= region_info['normal_range'][1]:
        risk_level = "normal"
        risk_color = "🟢"
        risk_text = "within normal limits"
    elif score <= region_info['concerning_range'][1]:
        risk_level = "concerning"
        risk_color = "🟡"
        risk_text = "showing concerning patterns"
    else:
        risk_level = "high_risk"
        risk_color = "🔴"
        risk_text = "indicating significant abnormalities"
    
    # Extract confidence percentage
    conf_str = str(confidence).replace('%', '')
    try:
        conf_value = float(conf_str)
    except:
        conf_value = 85.0
    
    # Generate explanation
    explanation = f"""
    {risk_color}<strong>Region Analysis:</strong> The {region_name.lower()} region shows an importance score of <strong>{score:.3f}</strong>, 
    which is <strong>{risk_text}</strong>.
    <p style="margin-top: 8px;">
    <strong>Clinical Context:</strong> This region is responsible for <em>{region_info['function'].lower()}</em>. 
    In Alzheimer's disease, {region_info['ad_significance'].lower()}.
    </p>
    <p style="margin-top: 8px;">
    <strong>Model Interpretation:</strong> The AI model assigned {conf_value:.0f}% confidence to the prediction of 
    <em>"{predicted_class}"</em>, with this region contributing significantly to the decision.
    </p>
    <p style="margin-top: 8px;">
   <strong>Clinical Significance:</strong> {get_clinical_interpretation(region_name, score, predicted_class)}
    </p>
    """
    
    return explanation

def get_clinical_interpretation(region_name, score, predicted_class):
    """Get clinical interpretation based on region and score"""
    
    region_info = BRAIN_REGION_KNOWLEDGE.get(region_name, {})
    
    if score >= 0.6:
        if region_name == 'Hippocampus':
            return "High hippocampal involvement suggests early memory circuit dysfunction, consistent with typical AD progression."
        elif region_name == 'Frontal':
            return "Significant frontal involvement may indicate executive dysfunction and behavioral changes."
        elif region_name == 'Temporal':
            return "Temporal lobe changes suggest language and semantic memory difficulties."
        elif region_name == 'Parietal':
            return "Parietal involvement indicates spatial processing deficits and attention difficulties."
        elif region_name == 'Ventricular':
            return "Ventricular prominence suggests significant brain volume loss."
    elif score >= 0.3:
        return f"Moderate involvement of the {region_name.lower()} region warrants monitoring and follow-up assessment."
    else:
        return f"The {region_name.lower()} region shows minimal involvement in this case."
    
    return "Clinical correlation recommended for comprehensive assessment."

def generate_patient_narrative(patient_data, region_data, analysis_type="comprehensive"):
    """Generate comprehensive patient narrative"""
    
    pred_class = patient_data['Predicted_Class']
    confidence_str = str(patient_data['Confidence']).replace('%', '')
    try:
        confidence = float(confidence_str)*100
    except:
        confidence = 85.0
    
    # Get top regions
    if not region_data.empty:
        top_regions = region_data.nlargest(3, 'ScoreCAM_Importance_Score')
        primary_region = top_regions.iloc[0]
        primary_score = float(primary_region['ScoreCAM_Importance_Score'])
        primary_name = primary_region['Brain_Region']
    else:
        primary_region = None
        primary_score = 0.0
        primary_name = "Unknown"
    
    # Generate narrative based on prediction
    if "Non Demented" in pred_class:
        narrative = f"""
        <strong>🟢 Assessment Summary:</strong> The AI model predicts <strong>"{pred_class}"</strong> with {confidence:.0f}% confidence.
        <p style="margin-top: 8px;">
        <strong>Key Findings:</strong> The analysis shows minimal concerning patterns across brain regions. 
        The {primary_name.lower()} region has the highest model attention (score: {primary_score:.3f}), but this remains within acceptable ranges.
        </p>
        <p style="margin-top: 8px;">
       <strong>Clinical Interpretation:</strong> The imaging features are consistent with normal aging patterns. 
        No significant indicators of neurodegenerative changes were detected by the AI model.
        </p>
        <p style="margin-top: 8px;">
        <strong>Recommendation:</strong> Continue routine monitoring. The current findings support normal cognitive aging without evidence of dementia-related changes.
        </p>
        """
    
    elif "Very Mild" in pred_class:
        narrative = f"""
        <strong>🟡 Assessment Summary:</strong> The AI model predicts <strong>"{pred_class}"</strong> with {confidence:.0f}% confidence.
        <p style="margin-top: 8px;">
        <strong>Key Findings:</strong> Early subtle changes detected, particularly in the {primary_name.lower()} region (score: {primary_score:.3f}). 
        {get_early_stage_interpretation(primary_name, primary_score)}
        </p>
        <p style="margin-top: 8px;">
        <strong>Clinical Interpretation:</strong> The findings suggest very early neurodegenerative changes that may represent 
        the earliest stages of cognitive decline. These changes are subtle but detectable by advanced AI analysis.
        </p>
        <p style="margin-top: 8px;">
        <strong>Recommendation:</strong> Close monitoring recommended. Consider comprehensive neuropsychological testing 
        and follow-up imaging in 6-12 months to track progression.
        </p>
        """
    
    elif "Mild" in pred_class:
        narrative = f"""
        <strong>🟠 Assessment Summary:</strong> The AI model predicts <strong>"{pred_class}"</strong> with {confidence:.0f}% confidence.
        <p style="margin-top: 8px;">
        <strong>Key Findings:</strong> Clear patterns of neurodegeneration detected, with prominent involvement of the 
        {primary_name.lower()} region (score: {primary_score:.3f}). {get_mild_stage_interpretation(primary_name, primary_score)}
        </p>
        <p style="margin-top: 8px;"><strong>Clinical Interpretation:</strong> The imaging features are consistent with mild cognitive impairment or early-stage dementia. 
        The pattern suggests established neurodegenerative changes affecting cognitive function.
        </p>
        <p style="margin-top: 8px;"><strong>Recommendation:</strong> Comprehensive clinical evaluation recommended. Consider biomarker testing and 
        discussion of treatment options. Regular monitoring essential.
        </p>
        """
    
    else:  # Moderate Demented
        narrative = f"""
        <strong>🔴 Assessment Summary:</strong> The AI model predicts <strong>"{pred_class}"</strong> with {confidence:.0f}% confidence.
        <p style="margin-top: 8px;">
        <strong>Key Findings:</strong> Significant neurodegenerative changes identified across multiple brain regions, 
        with severe involvement of the {primary_name.lower()} region (score: {primary_score:.3f}). 
        {get_moderate_stage_interpretation(primary_name, primary_score)}
        </p>
        <p style="margin-top: 8px;">
        <strong>Clinical Interpretation:</strong> The imaging shows advanced neurodegenerative changes consistent with 
        moderate-stage dementia. Multiple brain networks appear affected.
        </p>
        <p style="margin-top: 8px;">
        <strong>Recommendation:</strong> Immediate comprehensive clinical evaluation. Discuss care planning, safety measures, 
        and therapeutic interventions. Family counseling and support services recommended.
        </p>
        """
    
    return narrative

def get_early_stage_interpretation(region_name, score):
    """Get interpretation for early-stage findings"""
    if region_name == 'Hippocampus':
        return "Early hippocampal changes may affect short-term memory formation."
    elif region_name == 'Frontal':
        return "Subtle frontal changes might impact executive planning abilities."
    elif region_name == 'Temporal':
        return "Early temporal involvement could affect word-finding abilities."
    return "Early changes detected that warrant careful monitoring."

def get_mild_stage_interpretation(region_name, score):
    """Get interpretation for mild-stage findings"""
    if region_name == 'Hippocampus':
        return "Hippocampal involvement suggests established memory circuit dysfunction."
    elif region_name == 'Frontal':
        return "Frontal changes indicate executive function difficulties and potential behavioral changes."
    elif region_name == 'Temporal':
        return "Temporal involvement suggests language processing difficulties and semantic memory loss."
    return "Changes consistent with mild cognitive impairment."

def get_moderate_stage_interpretation(region_name, score):
    """Get interpretation for moderate-stage findings"""
    if region_name == 'Hippocampus':
        return "Severe hippocampal involvement indicates significant memory impairment."
    elif region_name == 'Frontal':
        return "Advanced frontal changes suggest severe executive dysfunction and personality changes."
    elif region_name == 'Temporal':
        return "Significant temporal involvement indicates substantial language and memory difficulties."
    return "Advanced changes requiring comprehensive care planning."

def create_interactive_region_selector(region_data):
    """Create interactive region selector"""
    if region_data.empty:
        return None
    
    regions = region_data['Brain_Region'].tolist()
    scores = region_data['ScoreCAM_Importance_Score'].tolist()
    
    # Create options with scores
    options = [f"{region} (Score: {score:.3f})" for region, score in zip(regions, scores)]
    
    selected = st.selectbox(
        "🎯 Select a brain region for detailed analysis:",
        options,
        help="Choose a brain region to see detailed clinical explanation"
    )
    
    if selected:
        return selected.split(" (Score:")[0]
    return None

def get_clinical_insights(region_name, region_data):
    """Get clinical insights for a specific region"""
    if region_name not in BRAIN_REGION_KNOWLEDGE:
        return []
    
    region_info = BRAIN_REGION_KNOWLEDGE[region_name]
    region_row = region_data[region_data['Brain_Region'] == region_name]
    
    if region_row.empty:
        return []
    
    score = float(region_row['ScoreCAM_Importance_Score'].iloc[0])
    
    insights = []
    
    # Function insight
    insights.append({
        'title': '🧠 Primary Function',
        'text': region_info['function'],
        'color': '#3b82f6'
    })
    
    # AD significance
    insights.append({
        'title': '🔬 AD Significance',
        'text': region_info['ad_significance'],
        'color': '#8b5cf6'
    })
    
    # Risk assessment
    if score >= region_info['high_risk_range'][0]:
        insights.append({
            'title': '⚠️ Risk Level',
            'text': 'High risk pattern detected - requires clinical attention',
            'color': '#ef4444'
        })
    elif score >= region_info['concerning_range'][0]:
        insights.append({
            'title': '🟡 Risk Level',
            'text': 'Concerning pattern - recommend monitoring',
            'color': '#f59e0b'
        })
    else:
        insights.append({
            'title': '✅ Risk Level',
            'text': 'Within normal limits',
            'color': '#10b981'
        })
    
    return insights

def generate_comparative_narrative(primary_patient, comparison_patient, primary_regions, comparison_regions):
    """Generate comparative analysis narrative"""
    
    primary_class = primary_patient['Predicted_Class']
    comparison_class = comparison_patient['Predicted_Class']
    
    # Get top regions for each patient
    if not primary_regions.empty and not comparison_regions.empty:
        primary_top = primary_regions.loc[primary_regions['Importance_Score'].idxmax()]
        comparison_top = comparison_regions.loc[comparison_regions['Importance_Score'].idxmax()]
        
        narrative = f"""
        <strong>🔬 Comparative Analysis:</strong> The primary patient shows a prediction of <strong>"{primary_class}"</strong> 
        while the comparison patient shows <strong>"{comparison_class}"</strong>.
        <p style="margin-top: 8px;">
        <strong>Key Differences:</strong> The primary patient's most affected region is the <strong>{primary_top['Brain_Region'].lower()}</strong> 
        (score: {primary_top['Importance_Score']:.3f}), while the comparison patient's most affected region is the 
        <strong>{comparison_top['Brain_Region'].lower()}</strong> (score: {comparison_top['Importance_Score']:.3f}).
        </p>
        <p style="margin-top: 8px;">
        <strong>Clinical Implications:</strong> {get_comparative_clinical_insight(primary_class, comparison_class, primary_top, comparison_top)}
        </p>
        <p style="margin-top: 8px;">
        <strong>Recommendation:</strong> Both cases demonstrate the importance of individualized assessment and monitoring, 
        as different brain regions may be affected in various patterns of cognitive decline.
        </p>
        """
    else:
        narrative = """
        <strong>🔬 Comparative Analysis:</strong> Insufficient region data available for detailed comparison. 
        <p style="margin-top: 8px;">
        <strong>Recommendation:</strong> Additional imaging analysis recommended for comprehensive comparison.
        </p>
        """
    
    return narrative

def get_comparative_clinical_insight(primary_class, comparison_class, primary_top, comparison_top):
    """Generate clinical insight from comparison"""
    
    if "Non Demented" in primary_class and "Demented" in comparison_class:
        return "This comparison illustrates the progression from normal aging to neurodegenerative changes, highlighting the importance of early detection."
    elif "Very Mild" in primary_class and "Moderate" in comparison_class:
        return "This comparison shows different stages of cognitive decline, demonstrating disease progression patterns."
    elif primary_top['Brain_Region'] == comparison_top['Brain_Region']:
        return f"Both patients show primary involvement of the {primary_top['Brain_Region'].lower()}, suggesting similar pathological patterns despite different severity."
    else:
        return "The different regional patterns suggest potentially different underlying pathological processes or disease variants."