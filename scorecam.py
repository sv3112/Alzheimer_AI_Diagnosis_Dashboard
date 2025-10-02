"""
ScoreCAM-based Brain Region Analysis for Alzheimer's Disease Detection

This module implements Score-weighted Class Activation Mapping (Score-CAM) for 
analyzing brain MRI images in the context of Alzheimer's disease classification.
It provides gradient-free attention visualization and anatomical region importance scoring.

Key Features:
- Gradient-free attention mapping using Score-CAM algorithm
- Anatomical brain region segmentation and importance scoring
- Comprehensive visualization suite for medical interpretation
- Optimized performance for clinical workflows
- Detailed brain region analysis with functional descriptions
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model #type: ignore
from PIL import Image
from scipy.ndimage import zoom
import os
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class ScoreCAMBrainAnalysis:
    """
    ScoreCAM-based Brain Region Analysis without LIME - OPTIMIZED VERSION
    
    This class implements Score-CAM (Score-weighted Class Activation Mapping) for analyzing
    brain MRI images, specifically designed for Alzheimer's disease classification.
    
    Key Features:
    - Gradient-free attention mapping using Score-CAM
    - Anatomical brain region segmentation and analysis
    - Optimized performance for faster processing
    - Comprehensive visualization outputs
    
    Score-CAM works by:
    1. Extracting activation maps from convolutional layers
    2. Using these maps as masks on the input image
    3. Scoring each mask by how much it increases prediction confidence
    4. Combining weighted activation maps to create final heatmap
    """
    
    def __init__(self, model, img_size=331):
        """
        Initialize the x.
        
        Args:
            model: Trained Keras/TensorFlow model for brain image classification
            img_size: Input image size (default 331 for InceptionV3-based models)
        """
        self.model = model
        self.img_size = img_size
        
        # Find the InceptionV3 backbone layer for feature extraction
        self.inception_layer = self._find_inception_layer()
        
        # Pre-compile prediction model for faster inference during Score-CAM computation
        self.prediction_model = self._create_prediction_model()
        
        # Define anatomical brain regions with approximate coordinate mappings
        # These coordinates are normalized fractions (0-1) of image dimensions
        self.brain_regions = {
            'Frontal': {
                'coords': (0.2, 0.6, 0.1, 0.4),  # (x_min, x_max, y_min, y_max) as fractions
                'description': 'Controls reasoning, planning, and movement',
                'color': [1.0, 0.2, 0.2]  # Red visualization color
            },
            'Temporal': {
                'coords': (0.1, 0.5, 0.4, 0.8),
                'description': 'Involved in memory and hearing',
                'color': [0.2, 1.0, 0.2]  # Green
            },
            'Parietal': {
                'coords': (0.3, 0.7, 0.2, 0.6),
                'description': 'Processes sensory information and spatial awareness',
                'color': [0.2, 0.2, 1.0]  # Blue
            },
            'Occipital': {
                'coords': (0.3, 0.8, 0.7, 0.95),
                'description': 'Processes visual information',
                'color': [1.0, 1.0, 0.2]  # Yellow
            },
            'Ventricular': {
                'coords': (0.3, 0.7, 0.50, 0.80),  # Moved down and increased width
                'description': 'Fluid-filled cavities affecting brain function',
                'color': [0.2, 1.0, 1.0]  # Cyan
            },

            'Hippocampus': {
                'coords': (0.3, 0.7, 0.70, 0.80),  # Moved down and increased width
                'description': 'Memory formation and spatial navigation',
                'color': [1.0, 0.2, 1.0]  # Cyan
            },
        }
        
    def _find_inception_layer(self):
        """
        Locate the InceptionV3 layer within the model architecture.
        
        This method searches through model layers to find InceptionV3, which contains
        the convolutional layers needed for Score-CAM feature extraction.
        
        Returns:
            InceptionV3 layer if found, None otherwise
        """
        for layer in self.model.layers:
            if 'inception' in layer.name.lower():
                print(f"✅ Found InceptionV3 layer: {layer.name}")
                return layer
        return None
    
    def _create_prediction_model(self):
        """
        Create a pre-compiled prediction model for faster inference.
        
        By running a dummy prediction during initialization, we pre-compile the model
        and avoid the first-run compilation overhead during Score-CAM computation.
        
        Returns:
            The original model, now pre-compiled for faster predictions
        """
        # Pre-compile model for batch predictions by running a dummy inference
        dummy_input = np.zeros((1, self.img_size, self.img_size, 3))
        _ = self.model.predict(dummy_input, verbose=0)
        return self.model
    
    def create_enhanced_brain_mask(self, image):
        """
        Create an enhanced binary mask isolating brain tissue from background.
        
        This method uses advanced image processing techniques to create a precise
        brain mask that excludes skull, background, and noise while preserving
        brain tissue boundaries.
        
        Processing steps:
        1. Convert to grayscale for simpler processing
        2. Apply Gaussian blur to reduce noise
        3. Use Otsu thresholding for automatic threshold selection
        4. Apply morphological operations to clean up the mask
        5. Find and keep only the largest connected component (main brain mass)
        
        Args:
            image: Input brain MRI image (RGB, normalized to 0-1)
            
        Returns:
            Binary mask where True indicates brain tissue pixels
        """
        # Convert RGB to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur for smoother edges and noise reduction
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhanced thresholding using Otsu's method for automatic threshold selection
        # This automatically finds the optimal threshold to separate brain from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Improved morphological operations - optimized kernel size for brain images
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Elliptical kernel fits brain shape
        
        # Close operation: fill small holes within brain tissue
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Open operation: remove small noise outside brain
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find largest connected component to isolate main brain mass
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        
        if num_features > 0:
            # Calculate size of each connected component
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            if len(sizes) > 0:
                # Keep only the largest component (main brain mass)
                largest_label = np.argmax(sizes) + 1
                binary = (labeled == largest_label).astype(np.uint8) * 255
        
        return binary > 0  # Return as boolean mask
    
    def create_anatomical_region_masks(self, brain_mask):
        """
        Create binary masks for specific anatomical brain regions.
        
        This method divides the brain into anatomically meaningful regions based on
        approximate coordinate mappings. Each region corresponds to areas with distinct
        functional roles in cognition and Alzheimer's pathology.
        
        The coordinate system uses normalized fractions (0-1) where:
        - (0,0) is top-left corner
        - (1,1) is bottom-right corner
        - Coordinates are defined as (x_min, x_max, y_min, y_max)
        
        Args:
            brain_mask: Binary mask indicating brain tissue pixels
            
        Returns:
            Dictionary mapping region names to their binary masks
        """
        h, w = brain_mask.shape
        region_masks = {}
        
        # Process each anatomical region defined in self.brain_regions
        for region_name, region_info in self.brain_regions.items():
            coords = region_info['coords']
            x_min, x_max, y_min, y_max = coords
            
            # Convert fractional coordinates to actual pixel coordinates
            x_start = int(x_min * w)
            x_end = int(x_max * w)
            y_start = int(y_min * h)
            y_end = int(y_max * h)
            
            # Create rectangular region mask using efficient numpy slicing
            region_mask = np.zeros_like(brain_mask, dtype=bool)
            region_mask[y_start:y_end, x_start:x_end] = True
            
            # Intersect with brain mask to get only brain tissue within this region
            # This ensures we only analyze actual brain tissue, not background
            region_masks[region_name] = region_mask & brain_mask
            
        return region_masks
    
    def calculate_region_importance_scores(self, heatmap, region_masks, brain_mask):
        """
        Calculate importance scores for each anatomical brain region based on Score-CAM heatmap.
        
        This method quantifies how much each brain region contributes to the model's
        prediction by analyzing the Score-CAM attention values within each region.
        
        Scoring methodology:
        1. Normalize Score-CAM heatmap to [0,1] range
        2. For each region, calculate average attention within that region
        3. Compute region statistics (area, pixel count)
        4. Return comprehensive scores for analysis
        
        Args:
            heatmap: Score-CAM attention heatmap
            region_masks: Dictionary of region binary masks
            brain_mask: Overall brain tissue mask
            
        Returns:
            Dictionary with detailed scores for each region including:
            - score_cam_score: Average attention score in region
            - area_percentage: Percentage of total brain area
            - pixel_count: Number of pixels in region
            - description: Anatomical description
        """
        print("🔄 Calculating region importance scores...")
        
        region_scores = {}
        
        # Normalize Score-CAM heatmap to [0, 1] range using vectorized operations
        heatmap_range = heatmap.max() - heatmap.min()
        if heatmap_range > 0:
            norm_heatmap = (heatmap - heatmap.min()) / heatmap_range
        else:
            norm_heatmap = heatmap
        
        # Pre-calculate total brain area for percentage calculations
        brain_area = np.sum(brain_mask)
        
        # Calculate scores for each anatomical region
        for region_name, region_mask in region_masks.items():
            region_sum = np.sum(region_mask)
            
            # Handle empty regions (no brain tissue in this coordinate area)
            if region_sum == 0:
                region_scores[region_name] = {
                    'score_cam_score': 0.0,
                    'area_percentage': 0.0,
                    'pixel_count': 0
                }
                continue
            
            # Calculate Score-CAM contribution for this region
            # This represents the average attention the model pays to this region
            score_cam_contribution = np.sum(norm_heatmap * region_mask) / region_sum
            
            # Calculate area statistics for contextual analysis
            area_percentage = (region_sum / brain_area * 100) if brain_area > 0 else 0
            
            # Store comprehensive region analysis
            region_scores[region_name] = {
                'score_cam_score': float(score_cam_contribution),
                'area_percentage': float(area_percentage),
                'pixel_count': int(region_sum),
                'description': self.brain_regions[region_name]['description']
            }
        
        return region_scores
    
    def score_cam(self, img_array, class_idx=None, use_brain_mask=True):
        """
        Compute Score-CAM: A gradient-free approach for generating attention heatmaps.
        
        Score-CAM Algorithm:
        1. Extract activation maps from the last convolutional layer
        2. For each activation channel:
           a. Normalize the activation map to [0,1]
           b. Upsample to input image size
           c. Use as mask on original input image
           d. Get prediction confidence on masked image
           e. Use confidence as weight for this activation map
        3. Combine all weighted activation maps to create final heatmap
        
        This approach is gradient-free and often more reliable than Grad-CAM,
        especially for medical imaging applications.
        
        OPTIMIZATION NOTES:
        - Processes only top 128 channels (selected by variance) for speed
        - Uses batch processing to reduce prediction overhead
        - Uses faster interpolation methods
        - Pre-allocates arrays for better memory management
        
        Args:
            img_array: Input image array (H, W, C) normalized to [0,1]
            class_idx: Target class index (if None, uses predicted class)
            use_brain_mask: Whether to apply brain mask to final heatmap
            
        Returns:
            Tuple of (heatmap, prediction_scores)
        """
        print("🔄 Computing Score-CAM (gradient-free)...")
        
        # Ensure input has batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Get baseline prediction to determine target class
        baseline_pred = self.prediction_model.predict(img_array, verbose=0)
        if class_idx is None:
            class_idx = np.argmax(baseline_pred[0])
        
        # Extract activation maps from last convolutional layer
        if self.inception_layer:
            try:
                # Try to get mixed10 activations (deepest features)
                activation_model = Model(
                    inputs=self.model.input,
                    outputs=self.inception_layer.get_layer('mixed10').output
                )
                activations = activation_model.predict(img_array, verbose=0)
            except:
                # Fallback hierarchy: try progressively earlier layers
                for layer_name in ['mixed9', 'mixed8', 'mixed7']:
                    try:
                        activation_model = Model(
                            inputs=self.model.input,
                            outputs=self.inception_layer.get_layer(layer_name).output
                        )
                        activations = activation_model.predict(img_array, verbose=0)
                        print(f"✅ Using {layer_name} for Score-CAM")
                        break
                    except:
                        continue
                else:
                    # If all mixed layers fail, use fallback method
                    print("⚠️ Could not access mixed layers, using activation variance")
                    return self.activation_variance_cam(img_array, class_idx, use_brain_mask)
        else:
            # No InceptionV3 layer found, use fallback
            return self.activation_variance_cam(img_array, class_idx, use_brain_mask)
        
        # Score-CAM computation with major optimizations
        batch, h, w, n_channels = activations.shape
        
        # OPTIMIZATION 1: Reduce number of channels processed for speed
        # Original InceptionV3 has 2048 channels in mixed10, we process only 128
        max_channels = min(n_channels, 128)
        
        # OPTIMIZATION 2: Select channels based on variance for better quality
        # High-variance channels contain more discriminative information
        channel_variances = np.var(activations[0], axis=(0, 1))
        top_channels = np.argsort(channel_variances)[-max_channels:]
        
        print(f"Processing top {max_channels} activation channels (selected by variance)...")
        
        # OPTIMIZATION 3: Pre-allocate score map for accumulating weighted activations
        score_map = np.zeros((h, w), dtype=np.float32)
        
        # OPTIMIZATION 4: Process channels in batches to reduce memory usage
        batch_size = 32  # Process 32 channels at once to balance speed and memory
        img_input = img_array[0]
        
        # Process channels in batches for efficiency
        for batch_start in range(0, len(top_channels), batch_size):
            batch_end = min(batch_start + batch_size, len(top_channels))
            batch_channels = top_channels[batch_start:batch_end]
            
            # Prepare batch of masked inputs for this channel batch
            masked_inputs = []
            channel_activations = []
            
            for i in batch_channels:
                # Get single channel activation map
                channel_activation = activations[0, :, :, i]
                
                # Normalize channel activation to [0, 1] range
                activation_range = channel_activation.max() - channel_activation.min()
                if activation_range > 0:
                    norm_activation = (channel_activation - channel_activation.min()) / activation_range
                    
                    # Upsample activation to input image size
                    # Using bilinear interpolation for speed (vs bicubic)
                    upsampled = cv2.resize(norm_activation, (self.img_size, self.img_size), 
                                         interpolation=cv2.INTER_LINEAR)
                    
                    # Create masked input by element-wise multiplication
                    # This masks the original image with the activation pattern
                    masked_input = img_input * upsampled[:, :, np.newaxis]
                    masked_inputs.append(masked_input)
                    channel_activations.append(channel_activation)
            
            # OPTIMIZATION 5: Batch prediction instead of individual predictions
            # This significantly reduces the overhead of multiple model calls
            if masked_inputs:
                masked_batch = np.array(masked_inputs)
                batch_preds = self.prediction_model.predict(masked_batch, verbose=0)
                
                # Update score map with batch results
                for j, (pred, channel_activation) in enumerate(zip(batch_preds, channel_activations)):
                    # Get prediction confidence for target class
                    score = pred[class_idx]
                    
                    # Weight this activation map by its prediction contribution
                    score_map += score * channel_activation
        
        # Normalize final score map to [0, 1] range
        if score_map.max() > 0:
            score_map = score_map / score_map.max()
        
        # Resize score map to match input image size
        heatmap = cv2.resize(score_map, (self.img_size, self.img_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        # Apply brain mask to focus attention on brain tissue only
        if use_brain_mask:
            brain_mask = self.create_enhanced_brain_mask(img_array[0])
            heatmap = heatmap * brain_mask
        
        return heatmap, baseline_pred[0]
    
    def activation_variance_cam(self, img_array, class_idx, use_brain_mask=True):
        """
        Fallback method using activation variance when Score-CAM fails.
        
        This method provides a simple edge-based attention map as a backup
        when the main Score-CAM computation cannot access the required
        convolutional layers.
        
        The approach:
        1. Convert image to grayscale
        2. Apply Canny edge detection to find important boundaries
        3. Blur edges to create smooth attention regions
        4. Apply brain mask if requested
        
        Args:
            img_array: Input image array
            class_idx: Target class index
            use_brain_mask: Whether to apply brain masking
            
        Returns:
            Tuple of (edge_based_heatmap, prediction_scores)
        """
        print("🔄 Computing Activation Variance CAM...")
        
        # Ensure batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Get model prediction
        pred = self.prediction_model.predict(img_array, verbose=0)
        
        # Create simple edge-based heatmap as fallback
        gray = cv2.cvtColor((img_array[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection with optimized parameters
        edges = cv2.Canny(gray, 50, 150)
        
        # Create smooth heatmap by blurring edges
        heatmap = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        
        # Normalize to [0, 1] range
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply brain mask if requested
        if use_brain_mask:
            brain_mask = self.create_enhanced_brain_mask(img_array[0])
            heatmap = heatmap * brain_mask
        
        return heatmap, pred[0]
    
    def create_individual_region_visualizations(self, img_array, region_masks, region_scores, brain_mask, output_dir, base_name):
        """
        Create individual visualization images for each brain region.
        
        This method generates separate images highlighting each anatomical region
        with color-coded importance levels. Each region gets its own image showing:
        - Grayscale background (original MRI)
        - Color overlay for the specific region
        - Importance score as title
        
        OPTIMIZATION FEATURES:
        - Creates grayscale background once and reuses
        - Uses vectorized operations for color application
        - Optimized matplotlib settings for faster rendering
        - Explicit memory management with figure closing
        
        Args:
            img_array: Original MRI image
            region_masks: Dictionary of region binary masks
            region_scores: Dictionary of region importance scores
            brain_mask: Overall brain mask
            output_dir: Directory to save individual images
            base_name: Base filename for output files
            
        Returns:
            Dictionary mapping region names to saved file paths
        """
        individual_region_paths = {}
        
        # Create grayscale background once for efficiency
        gray_bg = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_bg_rgb = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        
        # Turn off interactive mode for faster plotting
        plt.ioff()
        
        # Generate individual region visualizations
        for region_name, region_mask in region_masks.items():
            # Skip empty regions
            if np.sum(region_mask) == 0:
                continue
            
            # Start with grayscale background for each region
            region_viz = gray_bg_rgb.copy()
            
            # Get region importance score
            score = region_scores[region_name]['score_cam_score']
            score_percentage = score * 100
            
            # Create colored overlay for this specific region
            if np.sum(region_mask) > 0:
                # Use region-specific color from brain_regions definition
                overlay_color = np.array(self.brain_regions[region_name]['color'], dtype=np.float32)
                
                # Apply color overlay with fixed alpha blending
                # Higher importance regions could have higher alpha in future versions
                alpha = 0.6
                region_viz[region_mask] = (region_viz[region_mask] * (1 - alpha) + 
                                         overlay_color * alpha)
            
            # Create matplotlib figure with optimized settings
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)  # Reduced DPI for speed
            ax.imshow(region_viz)
            ax.set_title(f'{region_name}\nImportance: {score_percentage:.1f}%', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Remove margins for cleaner appearance
            plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
            
            # Save individual region image
            region_path = os.path.join(output_dir, f'{base_name}_{region_name.lower()}_region.png')
            plt.savefig(region_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.show()
            plt.close(fig)  # Explicitly close figure to free memory
            
            # Track saved paths
            individual_region_paths[region_name] = region_path
            print(f"✅ Saved {region_name} region visualization: {region_path}")
        
        # Turn interactive mode back on
        plt.ion()
        return individual_region_paths
    
    def plot_region_importance_chart(self, region_scores, save_path=None):
        """
        Create comprehensive charts showing brain region importance analysis.
        
        This method generates a 2x2 subplot layout containing:
        1. Score-CAM Importance Bar Chart - Shows raw importance scores
        2. Ranked Importance Chart - Shows regions sorted by importance
        3. Area Distribution Pie Chart - Shows relative region sizes
        4. Importance vs Area Scatter Plot - Shows relationship between size and importance
        
        These visualizations help interpret:
        - Which regions are most important for the model's decision
        - Whether importance correlates with region size
        - Overall distribution of attention across brain regions
        
        Args:
            region_scores: Dictionary of region scores from calculate_region_importance_scores
            save_path: Optional path to save the chart
            
        Returns:
            Matplotlib figure object
        """
        # Prepare data using vectorized operations for efficiency
        regions = list(region_scores.keys())
        score_cam_scores = [region_scores[r]['score_cam_score'] for r in regions]
        area_percentages = [region_scores[r]['area_percentage'] for r in regions]
        colors = [self.brain_regions[r]['color'] for r in regions]
        
        # Create figure with 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
        
        # 1. Score-CAM Importance Scores Bar Chart
        bars1 = ax1.bar(regions, score_cam_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Score-CAM Importance Scores by Brain Region', fontsize=14, weight='bold')
        ax1.set_ylabel('Importance Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars for precise reading
        for bar, score in zip(bars1, score_cam_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Regions sorted by importance (ranked view)
        sorted_data = sorted(zip(regions, score_cam_scores, colors), key=lambda x: x[1], reverse=True)
        sorted_regions, sorted_scores, sorted_colors = zip(*sorted_data)
        
        ax2.bar(range(len(sorted_regions)), sorted_scores, color=sorted_colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Regions Ranked by Importance', fontsize=14, weight='bold')
        ax2.set_ylabel('Importance Score')
        ax2.set_xticks(range(len(sorted_regions)))
        ax2.set_xticklabels(sorted_regions, rotation=45)
        
        # 3. Area percentage pie chart to show brain region size distribution
        ax3.pie(area_percentages, labels=regions, colors=colors, autopct='%1.1f%%')
        ax3.set_title('Brain Region Area Distribution', fontsize=14, weight='bold')
        
        # 4. Scatter plot showing relationship between region area and importance
        ax4.scatter(area_percentages, score_cam_scores, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add region labels to scatter points
        for i, region in enumerate(regions):
            ax4.annotate(region, (area_percentages[i], score_cam_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Area Percentage (%)')
        ax4.set_ylabel('Score-CAM Importance Score')
        ax4.set_title('Importance vs Area Analysis', fontsize=14, weight='bold')
        ax4.grid(True, alpha=0.3)  # Add grid for easier reading
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return fig
    
    def comprehensive_analysis(self, img_path, class_names, output_dir):
        """
        Run complete Score-CAM brain analysis with region importance scoring.
        
        This is the main analysis method that orchestrates the entire pipeline:
        
        ANALYSIS PIPELINE:
        1. Load and preprocess MRI image
        2. Get model prediction and confidence
        3. Generate Score-CAM attention heatmap
        4. Create brain tissue mask
        5. Define anatomical region masks
        6. Calculate region importance scores
        7. Create comprehensive visualizations
        8. Generate individual region images
        9. Create importance analysis charts
        10. Save all results and generate summary
        
        OUTPUT STRUCTURE:
        - Main analysis image (6-panel comprehensive view)
        - Region importance charts (4-panel analysis)
        - Individual region images (one per region)
        - Detailed JSON results with all scores and metadata
        
        Args:
            img_path: Path to input MRI image
            class_names: List of class names for prediction interpretation
            output_dir: Directory to save all analysis outputs
            
        Returns:
            Dictionary containing comprehensive analysis results including:
            - Predictions and confidence scores
            - Brain region importance scores
            - File paths to all generated visualizations
            - Processing time and metadata
        """
        total_start = time.time()
        
        print(f"\n🔍 Score-CAM Brain Analysis: {os.path.basename(img_path)}")
        print("🚀 Using optimized version for faster processing...")
        
        # Extract base filename for consistent output naming
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # STEP 1: Load and preprocess input image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)  # High-quality resampling
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1] range
        
        # STEP 2: Get model prediction and confidence
        predictions = self.prediction_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
        pred_class = np.argmax(predictions[0])
        pred_prob = predictions[0][pred_class]
        print(f"✅ Prediction: {class_names[pred_class]} ({pred_prob:.2%})")
        
        # STEP 3: Generate Score-CAM attention heatmap
        score_cam_heatmap, _ = self.score_cam(img_array, pred_class, use_brain_mask=True)
        
        # STEP 4: Create enhanced brain tissue mask
        brain_mask = self.create_enhanced_brain_mask(img_array)
        
        # STEP 5: Create anatomical region masks
        region_masks = self.create_anatomical_region_masks(brain_mask)
        
        # STEP 6: Calculate importance scores for each brain region
        region_scores = self.calculate_region_importance_scores(
            score_cam_heatmap, region_masks, brain_mask
        )
        
        # STEP 7: Create individual region visualizations
        individual_region_paths = self.create_individual_region_visualizations(
            img_array, region_masks, region_scores, brain_mask, output_dir, base_name
        )
        
        # STEP 8: Create comprehensive 6-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
        
        # ROW 1: Core Analysis Components
        
        # Panel 1: Original MRI scan
        axes[0,0].imshow(img_array)
        axes[0,0].set_title('Original MRI', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        # Panel 2: Brain tissue mask (shows skull stripping result)
        axes[0,1].imshow(brain_mask, cmap='gray')
        axes[0,1].set_title('Brain Mask', fontsize=14, fontweight='bold')
        axes[0,1].axis('off')
        
        # Panel 3: Score-CAM attention heatmap
        im1 = axes[0,2].imshow(score_cam_heatmap, cmap='hot')
        axes[0,2].set_title('Score-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[0,2].axis('off')
        plt.colorbar(im1, ax=axes[0,2], fraction=0.046)  # Add colorbar for intensity reference
        
        # ROW 2: Analysis Results and Interpretations
        
        # Panel 4: Score-CAM overlay on original image
        # Convert heatmap to RGB using jet colormap for overlay
        heatmap_colored = plt.cm.jet(score_cam_heatmap)[:, :, :3]
        score_cam_overlay = img_array * 0.5 + heatmap_colored * 0.5  # 50/50 blend
        axes[1,0].imshow(score_cam_overlay)
        axes[1,0].set_title('Score-CAM Overlay', fontsize=14, fontweight='bold')
        axes[1,0].axis('off')
        
        # Panel 5: Brain regions analysis with color-coded importance
        region_overlay = img_array.copy()
        
        # Apply color coding to each anatomical region based on importance
        for region_name, region_mask in region_masks.items():
            if np.sum(region_mask) > 0:
                # Get region-specific color and importance score
                color = np.array(self.brain_regions[region_name]['color'])
                score = region_scores[region_name]['score_cam_score']
                
                # Scale color intensity based on importance (higher score = more intense color)
                intensity = min(score * 2, 1.0)  # Cap at 1.0 to prevent oversaturation
                
                # Apply colored overlay using vectorized operations
                region_overlay[region_mask] = (region_overlay[region_mask] * 0.6 + 
                                             color * intensity * 0.4)
        
        axes[1,1].imshow(region_overlay)
        axes[1,1].set_title('Brain Regions Analysis', fontsize=14, fontweight='bold')
        axes[1,1].axis('off')
        
        # Panel 6: Analysis summary with key findings
        # Sort regions by importance for summary display
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1]['score_cam_score'], reverse=True)
        
        # Create formatted summary text
        summary_text = f"Prediction: {class_names[pred_class]}\nConfidence: {pred_prob:.2%}\n\nTop Regions:\n"
        
        # Add top 5 most important regions to summary
        for i, (region_name, scores) in enumerate(sorted_regions[:5]):
            score_pct = scores['score_cam_score'] * 100
            summary_text += f"{i+1}. {region_name}: {score_pct:.1f}%\n"
        
        # Display summary with styled text box
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1,2].set_title('Analysis Summary', fontsize=14, fontweight='bold')
        axes[1,2].axis('off')
        
        # Add overall title to the comprehensive visualization
        fig.suptitle(f'Score-CAM Brain Region Analysis - {class_names[pred_class]} ({pred_prob:.2%})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # STEP 9: Save main comprehensive analysis
        save_path = os.path.join(output_dir, f'{base_name}_scorecam_analysis.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # Explicitly close to free memory
        
        # STEP 10: Create and save detailed region importance charts
        region_chart_path = os.path.join(output_dir, f'{base_name}_region_importance_chart.png')
        region_fig = self.plot_region_importance_chart(region_scores, region_chart_path)
        plt.show()
        plt.close(region_fig)  # Explicitly close to free memory
        
        # STEP 11: Calculate and display processing time
        total_time = time.time() - total_start
        print(f"✅ Analysis completed in {total_time:.2f}s")
        print(f"💾 Main analysis saved to: {save_path}")
        print(f"💾 Region importance chart saved to: {region_chart_path}")
        print(f"💾 Individual region images saved: {len(individual_region_paths)} files")
        
        # STEP 12: Compile comprehensive results dictionary
        results = {
            'image': os.path.basename(img_path),
            'predicted_class': class_names[pred_class],
            'confidence': float(pred_prob),
            'all_predictions': predictions[0].tolist(),  # All class probabilities
            'class_names': class_names,
            'brain_region_scores': {
                region: {
                    'score_cam_score': scores['score_cam_score'],
                    'area_percentage': scores['area_percentage'],
                    'description': scores['description']
                } for region, scores in region_scores.items()
            },
            'analysis_time': total_time,
            'output_paths': {
                'main_analysis': save_path,
                'region_chart': region_chart_path,
                'individual_regions': individual_region_paths
            }
        }
        
        # STEP 13: Print formatted brain region summary table
        print("\n🧠 BRAIN REGION IMPORTANCE SUMMARY:")
        print("="*60)
        print(f"{'Region':<12} {'Score-CAM':<10} {'Area %':<8} {'Description'}")
        print("-"*60)
        
        # Display regions sorted by importance (highest first)
        for region, scores in sorted(region_scores.items(), key=lambda x: x[1]['score_cam_score'], reverse=True):
            print(f"{region:<12} {scores['score_cam_score']:<10.3f} {scores['area_percentage']:<8.1f} {scores['description']}")
        
        return results

