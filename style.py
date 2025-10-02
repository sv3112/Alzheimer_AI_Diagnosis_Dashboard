# style.py - Centralized CSS styling system for Alzheimer's AI Dashboard

def get_custom_css():
    """
    Returns the complete CSS styling system for the dashboard
    """
    return """
    <style>
        :root {
            --primary: #6B46C1;
            --secondary: #4C1D95;
            --tertiary: #1E3A8A;
            --text-primary: #333;
            --text-secondary: #666;
            --bg-light: #f8f9fa;
            --shadow-sm: 0 2px 10px rgba(0,0,0,0.05);
            --shadow-md: 0 5px 20px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 30px rgba(0,0,0,0.3);
            --radius: 15px;
        }
        
        /* Base styles */
        .main { padding: 0; }
        
        /* Hero section */
        .hero-section {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--tertiary) 100%);
            padding: 0.5rem 0.5rem;
            border-radius: var(--radius-2xl);
            margin-bottom: 2rem;
            box-shadow: var(--shadow-2xl);
            position: relative;
            overflow: hidden;
            text-align: center;
            color: white;
            backdrop-filter: blur(10px);
        }
        
        .hero-section::before {
            content: "";
            position: absolute;
            inset: -50%;
            background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.2) 0%, transparent 50%),
                        radial-gradient(circle at 70% 50%, rgba(255,255,255,0.1) 0%, transparent 50%);
            animation: float 6s ease-in-out infinite;
        }
        
        .hero-section::after {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: repeating-linear-gradient(
                45deg,
                transparent,
                transparent 10px,
                rgba(255,255,255,0.03) 10px,
                rgba(255,255,255,0.03) 20px
            );
            animation: slide 20s linear infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) scale(1); }
            50% { transform: translateY(-20px) scale(1.05); }
        }
        
        @keyframes slide {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }
        
        .hero-title {
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 900;
            margin-bottom: 1rem;
            position: relative;
            z-index: 1;
            text-shadow: 0 4px 15px rgba(0,0,0,0.2);
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        .hero-subtitle {
            font-size: 2rem;
            margin-bottom: 1rem;
            font-weight: 500;
            opacity: 0.95;
            position: relative;
            z-index: 1;
        }
        
        /* Enhanced Card Styles - Medical AI Theme */

        /* Modern Glassmorphism Card */
        .card {
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.95) 0%, 
                rgba(248, 250, 252, 0.9) 50%, 
                rgba(241, 245, 249, 0.85) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 2rem;
            border-radius: var(--radius);
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.08),
                0 4px 16px rgba(107, 70, 193, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.4);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            position: relative;
            overflow: hidden;
        }

        /* Card Hover Effects */
        .card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.15),
                0 8px 32px rgba(107, 70, 193, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.6);
            border-color: rgba(107, 70, 193, 0.4);
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.98) 0%, 
                rgba(248, 250, 252, 0.95) 50%, 
                rgba(239, 246, 255, 0.9) 100%);
        }

        /* Animated Background Gradient */
        .card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
                135deg,
                transparent 0%,
                rgba(107, 70, 193, 0.02) 25%,
                rgba(16, 185, 129, 0.01) 50%,
                rgba(236, 72, 153, 0.02) 75%,
                transparent 100%
            );
            opacity: 0;
            transition: opacity 0.4s ease;
            z-index: 1;
        }

        .card:hover::before {
            opacity: 1;
        }

        /* Top Border Accent */
        .card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(
                90deg, 
                var(--primary) 0%, 
                var(--neural) 50%, 
                var(--accent) 100%
            );
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.4s ease;
        }

        .card:hover::after {
            transform: scaleX(1);
        }

        /* Enhanced Card Icon */
        .card-icon {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--tertiary) 100%);
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            font-size: 2.2rem;
            color: white;
            position: relative;
            z-index: 2;
            box-shadow: 
                0 8px 25px rgba(107, 70, 193, 0.3),
                0 4px 12px rgba(76, 29, 149, 0.2);
            transition: all 0.3s ease;
        }

        .card:hover .card-icon {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 
                0 12px 35px rgba(107, 70, 193, 0.4),
                0 6px 18px rgba(76, 29, 149, 0.3);
        }

        /* Icon Glow Effect */
        .card-icon::before {
            content: "";
            position: absolute;
            inset: -2px;
            background: linear-gradient(135deg, var(--primary), var(--secondary), var(--tertiary));
            border-radius: 20px;
            z-index: -1;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .card:hover .card-icon::before {
            opacity: 0.3;
            animation: iconGlow 2s ease-in-out infinite;
        }

        /* Enhanced Typography */
        .card-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.8rem;
            position: relative;
            z-index: 2;
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            transition: all 0.3s ease;
        }

        .card:hover .card-title {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            transform: translateY(-2px);
        }

        .card-description {
            color: #64748b;
            font-size: 1rem;
            line-height: 1.7;
            position: relative;
            z-index: 2;
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            transition: color 0.3s ease;
        }

        .card:hover .card-description {
            color: #475569;
        }

        

        /* Featured Card Variant */
        .card-featured {
            position: relative;
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.98) 0%, 
                rgba(248, 250, 252, 0.95) 50%, 
                rgba(241, 245, 249, 0.9) 100%);
            border: 2px solid rgba(107, 70, 193, 0.3);
        }

        .card-featured::before {
            content: "FEATURED";
            position: absolute;
            top: -10px;
            right: 20px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            z-index: 10;
        }

        /* Card Loading State */
        .card-loading {
            pointer-events: none;
            opacity: 0.7;
        }

        .card-loading .card-icon {
            animation: pulse 2s infinite;
        }

        /* Card with Badge */
        .card-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(107, 70, 193, 0.1);
            color: var(--primary);
            padding: 4px 8px;
            border-radius: 8px;
            font-size: 0.7rem;
            font-weight: 600;
            backdrop-filter: blur(10px);
            z-index: 10;
        }

        /* Animations */
        @keyframes iconGlow {
            0%, 100% {
                box-shadow: 0 0 20px rgba(107, 70, 193, 0.3);
            }
            50% {
                box-shadow: 0 0 30px rgba(107, 70, 193, 0.5);
            }
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }

        /* Grid Layout Enhancements */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }

        .card-grid-compact {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .card {
                padding: 1.5rem;
            }
            
            .card-icon {
                width: 60px;
                height: 60px;
                font-size: 2rem;
            }
            
            .card-title {
                font-size: 1.2rem;
            }
            
            .card-description {
                font-size: 0.9rem;
            }
            
            .card-grid {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
        }
        
        /* Colorful Enhanced Performance Metrics Cards */

        /* ✨ Enhanced Elegant Metric Cards */

        .metric-card-button {
            width: 320px; /* ✅ Fixed width */
            height: 150px; /* ✅ Fixed height */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--tertiary) 100%);
            border: 2px solid rgba(255, 255, 255, 0.08);
            box-shadow:
                0 6px 20px rgba(0, 0, 0, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card-button::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg,
                rgba(102, 126, 234, 0.1) 0%,
                rgba(118, 75, 162, 0.06) 50%,
                rgba(240, 147, 251, 0.1) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
            border-radius: 18px;
        }

        .metric-card-button:hover {
            transform: translateY(-5px);
            box-shadow:
                0 15px 35px rgba(0, 0, 0, 0.2),
                0 0 10px rgba(102, 126, 234, 0.3);
        }

        .metric-card-button:hover::before {
            opacity: 1;
        }

        .metric-value {
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            margin-bottom: 0.4rem;
            color: #ffffff;
            margin-bottom: 0.4rem;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #f1f5f9;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }




        /* --------------------------------------------------------------------------
        METRIC CARDS
        -------------------------------------------------------------------------- */
        .metric-card_D {
            background: white;
            padding: 0.8rem;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            border-left: 3px solid #667eea;
            height: 100%;
        }

        .metric-value_D {
            font-size: 2.5rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.2rem;
        }

        .metric-label_D {
            font-size: 0.9rem;
            font-weight: 700;
            color: #666;
            margin-top: 0.2rem;
        }

        .metric-box_D {
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            margin: 12px 0;
        }
        /* --------------------------------------------------------------------------
        FEATURE CARDS & ANIMATED COMPONENTS
        -------------------------------------------------------------------------- */
        .feature-card_D {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 1rem 0.8rem;
            text-align: center;
            color: white;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .feature-card_D:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }

        .feature-card_D::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .icon-wrapper_D {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            margin: 0 auto 0.5rem;
            backdrop-filter: blur(10px);
        }

        .feature-value_D {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            margin: 0.3rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .feature-label_D {
            font-size: 0.85rem;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            opacity: 0.95;
            letter-spacing: 0.3px;
        }

        .feature-sublabel_D {
            font-size: 0.8rem;
            font-family: 'Inter', sans-serif;
            opacity: 0.8;
            margin-top: 0.3rem;
        }
        .feature-item {
            padding: 2rem;
            margin: 1rem 0;
            background: white;
            border-radius: 15px;
            border-left: 4px solid var(--primary);
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
            font-size: 1.25rem; /* 20px - good readable size */
            font-weight: 500;
            line-height: 1.6;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .feature-item:hover {
            transform: translateX(5px);
            box-shadow: var(--shadow-md);
        }
        
        /* Tech badges */
        .tech-badge {
            display: inline-block;
            background: #f0f0f0;
            color: var(--text-primary);
            padding: 0.4rem 0.8rem;
            border-radius: 15px;
            font-size: 0.85rem;
            margin: 0.2rem;
            font-weight: 500;
            border: 1px solid #e0e0e0;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            cursor: help;
        }
        
        .tooltip-text {
            visibility: hidden;
            width: 250px;
            background-color: #333;
            color: white;
            text-align: left;
            border-radius: 10px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.85rem;
            line-height: 1.4;
            box-shadow: var(--shadow-lg);
        }
        
        .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border: 5px solid transparent;
            border-top-color: #333;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
        
        /* Section styles */
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .section-subtitle {
            text-align: center;
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Process steps */
        .process-step {
            background: white;
            padding: 0.5rem;
            border-radius: 0px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            margin-left: 60px;
            position: relative;
            transition: transform 0.2s ease;
        }
        
        .process-step:hover {
            transform: translateX(10px);
        }
        
        .step-number {
            background: linear-gradient(135deg, #6B46C1 0%, #4C1D95 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            position: absolute;
            left: -60px;
            top: 50%;
            transform: translateY(-50%);
        }

        /* Upload Section Styles */
        .upload-section {
            background: linear-gradient(135deg, #f8f7ff 0%, #f0f0ff 100%);
            border: 2px solid #6B46C1;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 8px 30px rgba(107, 70, 193, 0.1);
            margin-bottom: 2rem;
        }

        .upload-section .upload-icon {
            font-size: 4rem;
            margin-right: 1.5rem;
            animation: bounce 2s infinite;
        }

        .upload-section .upload-title {
            color: #6B46C1;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
        }

        .upload-section .upload-description {
            color: #555;
            margin: 0.5rem 0 0 0;
            font-size: 1.15rem;
            line-height: 1.6;
            white-space: nowrap;
            overflow: visible;
            text-align: center;       /* Center align text */
            width: 100%; 
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        /* --------------------------------------------------------------------------
        CATEGORY CARDS & FEATURE ITEMS
        -------------------------------------------------------------------------- */
        .category-card {
            background: linear-gradient(135deg, #E0E7FF 0%, #C7D2FE 100%);
            border-radius: 15px;
            padding: 0.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .category-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.8rem;
        }

        .category-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #4C1D95;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .feature-count {
            background: #4C1D95;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.6rem;
            margin-top: 0.8rem;
        }

        .feature-item {
            background: white;
            padding: 0.5rem 0.8rem;
            border-radius: 8px;
            border-left: 3px solid #6366F1;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .feature-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background: #F5F3FF;
        }

        .feature-icon {
            color: #6366F1;
            font-size: 1.1rem;
        }

        .feature-detail-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin: 20px 0;
            position: relative;
            font-family: 'Inter', sans-serif;
        }

        .feature-header {
            display: flex;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f1f5f9;
        }

        .rank-badge {
            color: white;
            font-size: 20px;
            font-weight: bold;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
        }
        /* --------------------------------------------------------------------------
        INFO CARDS & CONTENT SECTIONS
        -------------------------------------------------------------------------- */
        .info-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }

        .content-section {
            margin-bottom: 20px;
        }

        .content-section h3 {
            color: #1e293b;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'Inter', sans-serif;
        }

        .content-section p {
            color: #475569;
            font-size: 20px;
            line-height: 1.6;
            margin: 0;
            text-align: justify;
            font-family: 'Inter', sans-serif;
        }

        .insight-box {
            border-left: 4px solid;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 10px 0;
            font-family: 'Inter', sans-serif;
        }
        /* --------------------------------------------------------------------------
        STREAMLIT COMPONENT CUSTOMIZATION
        -------------------------------------------------------------------------- */

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f8f9fa;
            border-radius: 10px 10px 0px 0px;
            font-size: 18px !important;
            font-weight: 600 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 18px !important;
        }

        /* Selectbox Styling */
        .category-selector {
            margin-bottom: 1.5rem;
        }

        .stSelectbox > label {
            font-size: 1.1rem;
            font-weight: 600;
            color: #4C1D95;
            margin-bottom: 0.5rem;
        }

        .stSelectbox > div > div {
            background: white;
            border: 2px solid #E0E7FF;
            border-radius: 10px;
        }

        /* Hide dropdown arrow/navigation symbols */
        .stSelectbox div[data-baseweb="select"] > div {
            background-image: none !important;
        }

        .stSelectbox div[data-baseweb="select"] > div::after {
            display: none !important;
        }

        .stSelectbox div[data-baseweb="select"] svg {
            display: none !important;
        }

        .stSelectbox div[data-baseweb="select"] > div > div:last-child {
            display: none !important;
        }

        .stSelectbox div[data-baseweb="select"] [data-testid="stMarkdownContainer"] + div {
            display: none !important;
        }

        /* Chart Spacing */
        .stPlotlyChart {
            margin-bottom: 2rem;
        }

        /* --------------------------------------------------------------------------
        TYPOGRAPHY & TITLES
        -------------------------------------------------------------------------- */
        .main-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #000000;
            font-family: 'Inter', sans-serif;
            margin-bottom: 1rem;
            margin-top: 1.5rem;
        }

        .section-title {
            font-size: 1.6rem;
            font-weight: 600;
            color: #000000;
            font-family: 'Inter', sans-serif;
            margin-bottom: 0.8rem;
            margin-top: 1.2rem;
        }

        .subsection-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #4C1D95;
            font-family: 'Inter', sans-serif;
            margin-bottom: 0.5rem;
            margin-top: 0.8rem;
        }

        /* Hide Streamlit's default header anchors */
        .stMarkdown h1 a, 
        .stMarkdown h2 a, 
        .stMarkdown h3 a, 
        .stMarkdown h4 a, 
        .stMarkdown h5 a, 
        .stMarkdown h6 a {
            display: none !important;
                
        /* Global font size increase */
        .stApp {
            font-size: 18px;
        }
        
        /* Slider labels and values */
        .stSlider label {
            font-size: 20px !important;
            font-weight: 600 !important;
        }
        
        .stSlider .stSlider-value {
            font-size: 18px !important;
        }
        
        /* Checkbox labels */
        .stCheckbox label {
            font-size: 18px !important;
            font-weight: 500 !important;
        }
        
        /* Tab labels */
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 18px !important;
            font-weight: 600 !important;
        }
        
        /* Expander header */
        .streamlit-expanderHeader {
            font-size: 20px !important;
            font-weight: 700 !important;
        }
        
        /* Markdown headers */
        .stMarkdown h4 {
            font-size: 22px !important;
            font-weight: 700 !important;
        }
        
        .stMarkdown h3 {
            font-size: 24px !important;
        }
        
        /* Regular text */
        .stMarkdown p {
            font-size: 16px !important;
        }
        
        /* Info boxes */
        .stInfo {
            font-size: 16px !important;
        }
        
        /* Success/Warning messages */
        .stSuccess, .stWarning {
            font-size: 16px !important;
        }
        
        /* Feature comparison cards */
        .feature-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
            transition: transform 0.2s;
        }
        
        .feature-card:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .feature-improved {
            border-left-color: #10b981;
            background: #f0fdf4;
        }
        
        .feature-worsened {
            border-left-color: #ef4444;
            background: #fef2f2;
        }
        
        .feature-name {
            font-weight: 600;
            color: #1f2937;
            font-size: 18px;
        }
        
        .feature-values {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 16px;
        }
        
        .value-change {
            font-weight: 500;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .value-improved {
            background: #10b981;
            color: white;
        }
        
        .value-worsened {
            background: #ef4444;
            color: white;
        }
        
        .value-neutral {
            background: #6b7280;
            color: white;
        }
        }
        
    </style>
    """

def apply_custom_css():
    """
    Apply the custom CSS to the Streamlit app
    """
    import streamlit as st # type: ignore
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def success_message(title="🎉 Analysis Successfully Completed!", description="Your data has been fully analyzed and results have been securely saved to the database."):
    """
    Returns a dynamic success message block with customizable title and description.
    """
    return f"""
    <div style="
        background: linear-gradient(135deg, #A7F3D0 0%, #6EE7B7 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 650px;
        margin: 2rem auto;">
        <h3 style="color: #065F46; margin-bottom: 1rem; font-size: 2.2rem; font-weight: 800;">
            {title}
        </h3>
        <p style="color: #064E3B; margin-bottom: 0; font-size: 1.2rem; line-height: 1.6;">
            {description}
        </p>
    </div>
    """
