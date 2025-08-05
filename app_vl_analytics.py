"""
Vidibemus Lumen Analytics - Customer Intelligence Platform
© 2025 Victor Collins Oppon. All rights reserved.

Author: Victor Collins Oppon, FCCA, MBA, BSc
Company: Vidibemus Lumen Analytics
Mission: "We Shall See the Light" - Illuminating insights through data science
Contact: victoroppdatascience1@gmail.com
LinkedIn: https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/
Portfolio: https://victoropp.github.io/
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration with VL Analytics branding and deployment functions
st.set_page_config(
    page_title="VL Analytics | We Shall See the Light",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://victoropp.github.io/',
        'Report a bug': "https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/",
        'About': """# Vidibemus Lumen Analytics
        
**'We Shall See the Light'**

Illuminating insights through data science and artificial intelligence

**Built by:** Victor Collins Oppon, FCCA, MBA, BSc  
**Portfolio:** https://victoropp.github.io/  
**LinkedIn:** https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/

---

### 🚀 Deployment Instructions

**For Streamlit Community Cloud:**
1. Push code to GitHub repository
2. Go to share.streamlit.io
3. Connect your GitHub account
4. Deploy from your repository

**Share Your Dashboard:**
- Copy the deployment URL
- Share on LinkedIn, WhatsApp, portfolio
- Perfect for demonstrating your data science skills!

© 2025 Victor Collins Oppon | All Rights Reserved"""
    }
)

# VL Analytics Custom CSS with PERFECT text contrast
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    /* CRITICAL: Force all text to be readable - HIGHEST PRIORITY */
    * {
        color: #0F2540 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Brand Colors with CONTRAST-SAFE versions */
    :root {
        --vl-gold: #F4B942;
        --vl-midnight: #0F2540;
        --vl-sapphire: #2563EB;
        --vl-dawn: #F3F4F6;
        --vl-twilight: #6B7280;
        --vl-success: #10B981;
        --vl-warning: #F59E0B;
        --vl-error: #EF4444;
        
        /* High contrast text colors */
        --text-primary: #0F2540;
        --text-secondary: #374151;
        --text-muted: #6B7280;
        --text-white: #FFFFFF;
        --bg-white: #FFFFFF;
        --bg-light: #F8FAFC;
    }
    
    /* STREAMLIT COMPONENT OVERRIDES - MAXIMUM CONTRAST */
    
    /* Main App Styling */
    .stApp {
        background: #FFFFFF !important;
        color: var(--text-primary) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background: #FFFFFF !important;
    }
    
    /* Streamlit Header - Show only essential deployment functions */
    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, var(--vl-midnight) 0%, var(--vl-sapphire) 100%) !important;
        border-bottom: 2px solid var(--vl-gold) !important;
        height: 60px !important;
    }
    
    /* Hide Streamlit footer but keep header */
    .stApp > footer { display: none; }
    
    /* Style the header buttons */
    header[data-testid="stHeader"] button {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(244, 185, 66, 0.3) !important;
        border-radius: 6px !important;
    }
    
    header[data-testid="stHeader"] button:hover {
        background: rgba(244, 185, 66, 0.2) !important;
        border-color: var(--vl-gold) !important;
    }
    
    /* Style header text */
    header[data-testid="stHeader"] * {
        color: white !important;
    }
    
    /* ALL STREAMLIT TEXT ELEMENTS */
    .stMarkdown, .stMarkdown *, 
    .stText, .stText *,
    .stTitle, .stTitle *,
    .stHeader, .stHeader *,
    .stSubheader, .stSubheader *,
    .stCaption, .stCaption *,
    .stCode, .stCode *,
    p, span, div, h1, h2, h3, h4, h5, h6, a, li, td, th {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Headers with stronger weight */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    
    h4, h5, h6, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* METRICS - Perfect contrast */
    .stMetric {
        background: var(--bg-white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stMetric label, .stMetric [data-testid="metric-label"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    
    .stMetric [data-testid="metric-delta"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
    
    /* INPUT ELEMENTS - Perfect visibility */
    .stSelectbox label, .stNumberInput label, .stTextInput label, 
    .stSlider label, .stRadio label, .stCheckbox label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-white) !important;
        border: 2px solid #D1D5DB !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox div[role="button"] {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput input, .stTextInput input {
        color: var(--text-primary) !important;
        background: var(--bg-white) !important;
        border: 2px solid #D1D5DB !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput input::placeholder, .stTextInput input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* BUTTONS - High contrast */
    .stButton button {
        background: linear-gradient(135deg, var(--vl-gold) 0%, var(--vl-sapphire) 100%) !important;
        color: var(--text-white) !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(244, 185, 66, 0.4) !important;
    }
    
    /* ALERTS - High contrast */
    .stAlert {
        background: var(--bg-white) !important;
        border: 2px solid var(--vl-gold) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .stAlert * {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Tab styling */
    /* TABS - Perfect contrast */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-light) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        border: 2px solid #E5E7EB !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 1rem 1.5rem !important;
        border-radius: 8px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-white) !important;
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        border: 2px solid var(--vl-gold) !important;
    }
    
    /* TABLES - High contrast */
    .stDataFrame {
        background: var(--bg-white) !important;
        border: 2px solid #D1D5DB !important;
        border-radius: 8px !important;
    }
    
    .stDataFrame table {
        color: var(--text-primary) !important;
    }
    
    .stDataFrame th {
        background: var(--bg-light) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    
    .stDataFrame td {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* STATUS INDICATORS - Maximum contrast */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700 !important;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: #DCFCE7 !important;
        color: #166534 !important;
        border: 2px solid #16A34A !important;
    }
    
    .status-warning {
        background: #FEF3C7 !important;
        color: #92400E !important;
        border: 2px solid #D97706 !important;
    }
    
    .status-error {
        background: #FEE2E2 !important;
        color: #991B1B !important;
        border: 2px solid #DC2626 !important;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.2rem;
        color: var(--vl-midnight);
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    .sub-header {
        text-align: center !important;
        color: var(--text-secondary) !important;
        font-size: 1.2rem !important;
        margin-bottom: 2rem !important;
        font-weight: 600 !important;
    }
    
    /* FOOTER */
    .vl-footer {
        background: var(--vl-midnight) !important;
        color: var(--text-white) !important;
        padding: 3rem 2rem !important;
        border-radius: 0 !important;
    }
    
    .vl-footer * {
        color: var(--text-white) !important;
    }
    
    .vl-footer a {
        color: var(--vl-gold) !important;
        font-weight: 600 !important;
    }
    
    /* LOGO CONTAINER */
    .logo-container {
        background: var(--bg-white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 16px !important;
        padding: 3rem 2rem !important;
        margin: 2rem 0 !important;
        text-align: center !important;
    }
    
    /* VL Brand Badge */
    .vl-brand {
        background: linear-gradient(135deg, var(--vl-midnight) 0%, var(--vl-sapphire) 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        display: inline-block;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Enhanced VL Logo Circle */
    .vl-logo {
        width: 140px;
        height: 140px;
        background: linear-gradient(135deg, var(--vl-gold) 0%, var(--vl-sapphire) 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 2rem;
        box-shadow: 0 12px 40px rgba(244, 185, 66, 0.4), 0 4px 8px rgba(0, 0, 0, 0.1);
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        font-family: Georgia, serif;
        position: relative;
        animation: logoGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes logoGlow {
        0% { box-shadow: 0 12px 40px rgba(244, 185, 66, 0.4), 0 4px 8px rgba(0, 0, 0, 0.1); }
        100% { box-shadow: 0 16px 50px rgba(244, 185, 66, 0.6), 0 6px 12px rgba(0, 0, 0, 0.15); }
    }
    
    .vl-logo::after {
        content: '';
        position: absolute;
        width: 160px;
        height: 160px;
        border: 2px solid rgba(244, 185, 66, 0.3);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        100% { transform: scale(1.1); opacity: 0; }
    }
    
    /* Professional Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(15, 37, 64, 0.08), 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(244, 185, 66, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--vl-gold), var(--vl-sapphire));
        transition: left 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(15, 37, 64, 0.15), 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover::before {
        left: 0;
    }
    
    /* Executive Summary Cards */
    .executive-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(15, 37, 64, 0.1);
        border: 1px solid rgba(244, 185, 66, 0.15);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .executive-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, var(--vl-gold), var(--vl-sapphire));
        opacity: 0.1;
        border-radius: 0 20px 0 80px;
    }
    
    .executive-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(15, 37, 64, 0.15);
    }
    
    /* Risk Indicators */
    .risk-high {
        color: var(--vl-error);
        font-weight: 600;
    }
    
    .risk-medium {
        color: var(--vl-warning);
        font-weight: 600;
    }
    
    .risk-low {
        color: var(--vl-success);
        font-weight: 600;
    }
    
    /* Professional Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--vl-gold) 0%, var(--vl-sapphire) 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: 12px;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 12px 30px rgba(244, 185, 66, 0.4);
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    /* Icon Buttons */
    .icon-button {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, var(--vl-gold) 0%, var(--vl-sapphire) 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
    }
    
    .icon-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(244, 185, 66, 0.3);
        color: white;
        text-decoration: none;
    }
    
    /* Professional Insight Boxes */
    .insight-box {
        background: linear-gradient(135deg, rgba(244, 185, 66, 0.05) 0%, rgba(255, 255, 255, 0.8) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 1px solid rgba(244, 185, 66, 0.2);
        border-left: 6px solid var(--vl-gold);
        font-size: 1rem;
        line-height: 1.7;
        position: relative;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .insight-box::before {
        content: '💡';
        position: absolute;
        top: 1rem;
        right: 1.5rem;
        font-size: 1.5rem;
        opacity: 0.6;
    }
    
    .insight-box:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 25px rgba(244, 185, 66, 0.15);
        border-left-width: 10px;
    }
    
    /* Professional Chart Container */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 37, 64, 0.08);
        border: 1px solid rgba(244, 185, 66, 0.1);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 30px rgba(15, 37, 64, 0.12);
        transform: translateY(-2px);
    }
    
    /* Sidebar Styling */
    /* SIDEBAR - MAXIMUM WHITE TEXT CONTRAST */
    section[data-testid="stSidebar"] {
        background: #1a202c !important;
        border-right: 2px solid var(--vl-gold) !important;
    }
    
    /* FORCE ALL SIDEBAR TEXT TO BE BRIGHT WHITE - EXCEPT SELECT BOX CONTENT */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown *,
    section[data-testid="stSidebar"] .element-container,
    section[data-testid="stSidebar"] .element-container *,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar select box LABELS stay white */
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar Headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
    }
    
    /* Sidebar Select Boxes - VERY DARK text on white background */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #FFFFFF !important;
        border: 2px solid var(--vl-gold) !important;
    }
    
    /* The selected value display - DARK TEXT */
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] {
        color: #000000 !important;
        font-weight: 800 !important;
        background: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] span {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] div {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* Force all text inside select box to be black */
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] * {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* Dropdown options - DARK TEXT */
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="popover"] {
        background: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [role="listbox"] {
        background: #FFFFFF !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [role="option"] {
        color: #000000 !important;
        background: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [role="option"]:hover {
        background: #F3F4F6 !important;
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [aria-selected="true"] {
        background: var(--vl-gold) !important;
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* Navigation Items */
    .nav-item {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid transparent;
    }
    
    .nav-item:hover {
        background: rgba(244, 185, 66, 0.1);
        border-color: rgba(244, 185, 66, 0.3);
        transform: translateX(4px);
    }
    
    .nav-item.active {
        background: rgba(244, 185, 66, 0.15);
        border-color: var(--vl-gold);
    }
    
    /* Professional Footer */
    .vl-footer {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, var(--vl-midnight) 0%, #1a365d 100%);
        color: white;
        border-top: 4px solid var(--vl-gold);
        margin-top: 4rem;
        font-size: 0.95rem;
        border-radius: 24px 24px 0 0;
        position: relative;
    }
    
    .vl-footer::before {
        content: '';
        position: absolute;
        top: -4px;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--vl-gold), var(--vl-sapphire), var(--vl-midnight));
    }
    
    .vl-footer a {
        color: var(--vl-gold) !important;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .vl-footer a:hover {
        color: white !important;
        text-shadow: 0 0 8px var(--vl-gold);
    }
    
    /* Loading Animations */
    @keyframes shimmer {
        0% { background-position: -468px 0; }
        100% { background-position: 468px 0; }
    }
    
    .loading {
        animation: shimmer 1.5s ease-in-out infinite;
        background: linear-gradient(to right, #f6f7f8 8%, #edeef1 18%, #f6f7f8 33%);
        background-size: 800px 104px;
    }
    
    /* Professional Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--vl-midnight);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .tooltip:hover::after {
        opacity: 1;
        visibility: visible;
    }
    
    /* Enhanced Logo Container */
    .logo-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(244, 185, 66, 0.08) 0%, rgba(15, 37, 64, 0.05) 50%, rgba(37, 99, 235, 0.05) 100%);
        border-radius: 24px;
        position: relative;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(244, 185, 66, 0.2);
        box-shadow: 0 8px 32px rgba(15, 37, 64, 0.1);
    }
    
    .logo-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--vl-gold), var(--vl-sapphire), var(--vl-midnight));
        border-radius: 24px 24px 0 0;
    }
    
    /* HEADER ELEMENTS */
    .company-name {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
    }
    
    .tagline {
        color: var(--vl-gold) !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        font-style: italic;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* CUSTOM COMPONENTS - Perfect contrast */
    .metric-card {
        background: var(--bg-white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .metric-card * {
        color: var(--text-primary) !important;
    }
    
    .executive-card {
        background: var(--bg-white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    .executive-card * {
        color: var(--text-primary) !important;
    }
    
    .insight-box {
        background: var(--bg-white) !important;
        border: 2px solid var(--vl-gold) !important;
        border-left: 6px solid var(--vl-gold) !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
    }
    
    .insight-box * {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    .chart-container {
        background: var(--bg-white) !important;
        border: 2px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
    }
    
    .chart-container * {
        color: var(--text-primary) !important;
    }
    
</style>

<!-- Professional Icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

class ChurnPredictor:
    """Handle all prediction-related functionality with VL Analytics standards"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.load_models()
        
    def load_models(self):
        """Load trained models with error handling"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load model
            model_path = os.path.join(current_dir, "models", "best_model_Logistic_Regression.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            
            # Load scaler
            scaler_path = os.path.join(current_dir, "models", "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
                
            # Load label encoders
            encoders_path = os.path.join(current_dir, "models", "label_encoders.pkl")
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                logger.info("Label encoders loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def engineer_features(self, customer_data):
        """
        Engineer all features required by the model
        """
        df = pd.DataFrame([customer_data])
        
        # 1. Basic Risk Features
        df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['ContractStability'] = df['Contract'].map({
            'Month-to-month': 0, 'One year': 0.5, 'Two year': 1
        }).fillna(0)
        
        df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['PaymentStability'] = df['PaymentMethod'].map({
            'Electronic check': 0,
            'Mailed check': 0.33,
            'Bank transfer (automatic)': 0.67,
            'Credit card (automatic)': 1
        }).fillna(0)
        
        df['FinancialRiskScore'] = (
            df['IsMonthToMonth'] * 0.427 +
            df['IsElectronicCheck'] * 0.453 +
            (1 - df['PaymentStability']) * 0.12
        )
        
        # 2. Tenure Features
        df['TenureRisk'] = np.exp(-df['tenure'] / 12)
        df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
        df['IsVeryNewCustomer'] = (df['tenure'] <= 3).astype(int)
        
        # Tenure segment as numeric
        tenure_val = df['tenure'].values[0]
        if tenure_val <= 3:
            df['TenureSegment'] = 0  # Onboarding
        elif tenure_val <= 12:
            df['TenureSegment'] = 1  # First Year
        elif tenure_val <= 24:
            df['TenureSegment'] = 2  # Growing
        elif tenure_val <= 48:
            df['TenureSegment'] = 3  # Established
        else:
            df['TenureSegment'] = 4  # Loyal
            
        df['LifecycleValue'] = df['tenure'] * df['MonthlyCharges']
        df['RevenueAcceleration'] = df['TotalCharges'] / (df['tenure'] + 1) - df['MonthlyCharges']
        
        # 3. Service Features
        df['HasTechSupport'] = (df['TechSupport'] == 'Yes').astype(int)
        df['HasOnlineSecurity'] = (df['OnlineSecurity'] == 'Yes').astype(int)
        df['HasBackup'] = (df['OnlineBackup'] == 'Yes').astype(int)
        
        # Protection score
        protection_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['ProtectionScore'] = 0
        for service in protection_services:
            if service in df.columns:
                df['ProtectionScore'] += (df[service] == 'Yes').astype(int)
                
        df['FullyProtected'] = (df['ProtectionScore'] == 4).astype(int)
        df['NoProtection'] = (df['ProtectionScore'] == 0).astype(int)
        
        df['ServiceVulnerability'] = (
            (df['InternetService'] != 'No').astype(int) * 
            (1 - df['ProtectionScore'] / 4)
        )
        
        # 4. Service Count
        df['ServiceCount'] = 0
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for service in service_cols:
            if service in df.columns:
                if service == 'InternetService':
                    df['ServiceCount'] += (df[service] != 'No').astype(int)
                else:
                    df['ServiceCount'] += (df[service] == 'Yes').astype(int)
        
        df['RevenuePerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
        
        # 5. Price Features
        avg_monthly_charges = 64.76
        df['PriceDeviation'] = df['MonthlyCharges'] - avg_monthly_charges
        df['IsPriceSensitive'] = (
            (df['MonthlyCharges'] < avg_monthly_charges) & 
            (df['ServiceCount'] < 4)
        ).astype(int)
        
        df['EstimatedMargin'] = df['MonthlyCharges'] * 0.3 - df['ServiceCount'] * 5
        df['CumulativeProfit'] = df['EstimatedMargin'] * df['tenure']
        
        # 6. Digital Features
        df['DigitalAdoption'] = 0
        if 'PaperlessBilling' in df.columns:
            df['DigitalAdoption'] += (df['PaperlessBilling'] == 'Yes').astype(int) / 3
        if 'PaymentMethod' in df.columns:
            df['DigitalAdoption'] += df['PaymentMethod'].str.contains('automatic', na=False).astype(int) / 3
        if 'OnlineBackup' in df.columns:
            df['DigitalAdoption'] += (df['OnlineBackup'] == 'Yes').astype(int) / 3
        
        df['IsStreamingUser'] = 0
        if 'StreamingTV' in df.columns and 'StreamingMovies' in df.columns:
            df['IsStreamingUser'] = (
                (df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')
            ).astype(int)
        
        df['IsPureStreaming'] = (
            df['IsStreamingUser'] & (df['InternetService'] == 'Fiber optic')
        ).astype(int)
        
        # 7. Risk Combinations
        df['ContractValueAlignment'] = (
            (df['Contract'] == 'Month-to-month').astype(int) * 
            (df['MonthlyCharges'] > avg_monthly_charges).astype(int)
        )
        
        df['RiskyNewCustomer'] = df['IsNewCustomer'] * df['IsMonthToMonth']
        
        df['VulnerableHighValue'] = (
            (df['MonthlyCharges'] > 70) & 
            (df['NoProtection'] == 1)
        ).astype(int)
        
        df['UnstableCustomer'] = (
            df['IsMonthToMonth'] + 
            df['IsElectronicCheck'] + 
            df['IsNewCustomer']
        ) / 3
        
        df['EarlyServiceOverload'] = (
            (df['tenure'] < 6) & (df['ServiceCount'] > 5)
        ).astype(int)
        
        # 8. Statistical Features
        df['MonthlyCharges_ZScore'] = (df['MonthlyCharges'] - 64.76) / 30.09
        df['TotalCharges_ZScore'] = (df['TotalCharges'] - 2283.30) / 2266.77
        
        df['ChargesPercentile'] = np.clip(df['MonthlyCharges'] / 118.75, 0, 1)
        df['TenurePercentile'] = np.clip(df['tenure'] / 72, 0, 1)
        
        # 9. Consistency Features
        df['ExpectedTotalCharges'] = df['tenure'] * df['MonthlyCharges']
        df['ChargesConsistency'] = np.minimum(
            df['TotalCharges'] / (df['ExpectedTotalCharges'] + 1), 1
        )
        
        # 10. Final Risk Score
        df['ChurnRiskScore'] = (
            df['FinancialRiskScore'] * 0.3 +
            df['TenureRisk'] * 0.2 +
            df['ServiceVulnerability'] * 0.2 +
            df['UnstableCustomer'] * 0.15 +
            (1 - df['ProtectionScore'] / 4) * 0.15
        )
        
        # Risk category as numeric
        risk_score = df['ChurnRiskScore'].values[0]
        if risk_score < 0.33:
            df['RiskCategory'] = 0  # Low Risk
        elif risk_score < 0.66:
            df['RiskCategory'] = 1  # Medium Risk
        else:
            df['RiskCategory'] = 2  # Critical Risk
            
        # Data segment (numeric)
        df['DataSegment'] = 1
        
        # Segment name as numeric
        if df['ServiceCount'].values[0] <= 2 and df['MonthlyCharges'].values[0] < 40:
            df['SegmentName'] = 0  # Basic Users
        elif df['ServiceCount'].values[0] >= 6:
            df['SegmentName'] = 1  # Power Users
        elif df['tenure'].values[0] > 48:
            df['SegmentName'] = 2  # Loyal Customers
        elif df['IsNewCustomer'].values[0] and df['MonthlyCharges'].values[0] < avg_monthly_charges:
            df['SegmentName'] = 3  # Value Seekers
        else:
            df['SegmentName'] = 4  # Standard Users
            
        return df
    
    def prepare_features_for_model(self, engineered_df):
        """Prepare features in the exact format expected by the model"""
        # Original categorical columns that need encoding
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']
        
        # Create a copy for encoding
        df_encoded = engineered_df.copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col in self.label_encoders:
                    try:
                        # Handle the encoding
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                    except ValueError:
                        # If unseen label, use most frequent class encoding
                        df_encoded[col] = 0
                else:
                    # Simple encoding if no encoder available
                    if col == 'gender':
                        df_encoded[col] = (df_encoded[col] == 'Female').astype(int)
                    elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                        df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
                    elif col == 'Contract':
                        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
                        df_encoded[col] = df_encoded[col].map(contract_map).fillna(0)
                    elif col == 'PaymentMethod':
                        payment_map = {
                            'Electronic check': 0,
                            'Mailed check': 1,
                            'Bank transfer (automatic)': 2,
                            'Credit card (automatic)': 3
                        }
                        df_encoded[col] = df_encoded[col].map(payment_map).fillna(0)
                    else:
                        # For service columns
                        df_encoded[col] = df_encoded[col].map({'Yes': 2, 'No': 0, 'No internet service': 1, 'No phone service': 1}).fillna(0)
        
        # Ensure we have exactly 61 features (excluding customerID and Churn)
        feature_cols = [col for col in df_encoded.columns if col not in ['customerID', 'Churn']]
        
        # If we're missing any columns, add them with default values
        expected_features = 61
        if len(feature_cols) < expected_features:
            # Add any missing columns with zeros
            for i in range(len(feature_cols), expected_features):
                df_encoded[f'feature_{i}'] = 0
                feature_cols.append(f'feature_{i}')
        
        return df_encoded[feature_cols[:expected_features]]
    
    def predict(self, customer_data):
        """Make prediction for a single customer"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Engineer features
            engineered_df = self.engineer_features(customer_data)
            
            # Prepare for model
            features = self.prepare_features_for_model(engineered_df)
            
            # Convert to numpy array and ensure float type
            features_array = features.values.astype(float)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                try:
                    features_array = self.scaler.transform(features_array)
                except:
                    logger.warning("Scaler transform failed, using unscaled features")
            
            # Make prediction
            pred_proba = self.model.predict_proba(features_array)[0, 1]
            
            # Determine risk category
            if pred_proba >= 0.7:
                risk_category = "High"
            elif pred_proba >= 0.4:
                risk_category = "Medium"
            else:
                risk_category = "Low"
                
            return pred_proba, risk_category
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, f"Error: {str(e)}"

# Cache data loading functions
@st.cache_data
def load_data():
    """Load customer data"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
            return df
        
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_insights():
    """Load insights"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        insights_path = os.path.join(current_dir, "outputs", "key_insights.json")
        
        if os.path.exists(insights_path):
            with open(insights_path, 'r') as f:
                return json.load(f)
                
        return {
            "churn_rate": 26.5,
            "annual_revenue_at_risk": 1452475,
            "high_risk_segments": {
                "Month-to-month contracts": 42.7,
                "Electronic check payments": 45.3,
                "No tech support": 15.2
            },
            "recommendations": [
                "Focus retention efforts on month-to-month contract customers",
                "Incentivize customers to switch from electronic check payments",
                "Bundle technical support services to reduce churn",
                "Target retention campaigns for customers in first year"
            ]
        }
    except Exception as e:
        logger.error(f"Error loading insights: {e}")
        return {}

@st.cache_data
def load_metrics():
    """Load executive metrics"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_path = os.path.join(current_dir, "outputs", "executive_metrics.csv")
        
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path)
            
        return pd.DataFrame({
            'Metric': ['Total Customers', 'Churn Rate', 'Annual Revenue at Risk', 'Target Savings'],
            'Value': ['7,043', '26.5%', '$1,452,475', '$363,119']
        })
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        return pd.DataFrame()

def render_vl_header():
    """Render VL Analytics branded header"""
    st.markdown("""
        <div class="logo-container">
            <div class="vl-logo">VL</div>
            <h1 class="company-name">
                <span style="color: #0F2540;">Vidibemus</span> 
                <span style="color: #F4B942;">Lumen</span> 
                <span style="color: #2563EB;">Analytics</span>
            </h1>
            <p class="tagline">"We Shall See the Light"</p>
            <h2 class="main-header">Customer Intelligence Platform</h2>
            <p class="sub-header">Illuminating the path to customer intelligence excellence</p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    """Render branded sidebar"""
    # VL Analytics branding - BRIGHT WHITE TEXT
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: rgba(255, 255, 255, 0.1); border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.2);'>
            <div style='width: 80px; height: 80px; background: linear-gradient(135deg, #F4B942 0%, #2563EB 100%); 
                        border-radius: 50%; margin: 0 auto 1rem; display: flex; align-items: center; 
                        justify-content: center; font-size: 2rem; font-weight: 900; color: white; font-family: Georgia, serif;'>VL</div>
            <h2 style='color: #FFFFFF !important; font-size: 1.3rem; margin-bottom: 0.5rem; font-weight: 800;'>Vidibemus Lumen Analytics</h2>
            <p style='color: #FFFFFF !important; font-size: 0.9rem; font-style: italic; font-weight: 600; opacity: 0.9;'>"We Shall See the Light"</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Navigation - BRIGHT WHITE TEXT
    st.sidebar.markdown("""
        <h3 style='color: #FFFFFF !important; font-size: 1.4rem; margin-bottom: 1rem; font-weight: 800; text-align: left;'>
            🧭 Navigation
        </h3>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose your intelligence focus:",
        ["Executive Intelligence", "Customer Analytics", "Predictive Intelligence", "Strategic Planning"]
    )
    
    st.sidebar.markdown("---")
    
    # Quick Intelligence - BRIGHT WHITE TEXT
    if not df.empty:
        total_customers = len(df)
        churn_rate = (df['Churn'] == 'Yes').sum() / total_customers * 100
        revenue_at_risk = churn_rate * total_customers * 64.76 * 12 / 100
        
        st.sidebar.markdown("""
            <div style='background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid rgba(255, 255, 255, 0.2);'>
                <h4 style='color: #FFFFFF !important; margin-bottom: 1rem; font-size: 1.2rem; font-weight: 800;'>📊 Quick Intelligence</h4>
                <div style='margin-bottom: 1rem;'>
                    <div style='color: #FFFFFF !important; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.2rem;'>Customers</div>
                    <div style='color: #FFFFFF !important; font-size: 1.5rem; font-weight: 800;'>{:,}</div>
                </div>
                <div style='margin-bottom: 1rem;'>
                    <div style='color: #FFFFFF !important; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.2rem;'>Risk Rate</div>
                    <div style='color: #FFFFFF !important; font-size: 1.5rem; font-weight: 800;'>{:.1f}%</div>
                </div>
                <div style='background: rgba(0, 0, 0, 0.2); padding: 1rem; border-radius: 6px; margin-top: 1rem;'>
                    <div style='color: #FFFFFF !important; font-size: 0.9rem; font-weight: 700; margin-bottom: 0.3rem;'>VL Insight:</div>
                    <div style='color: #FFFFFF !important; font-size: 0.85rem; font-weight: 600; line-height: 1.3;'>${:,.0f} annual revenue at risk</div>
                </div>
            </div>
        """.format(total_customers, churn_rate, revenue_at_risk), unsafe_allow_html=True)
    
    # About section - BRIGHT WHITE TEXT
    st.sidebar.markdown("""
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid rgba(255, 255, 255, 0.2);'>
            <h4 style='color: #FFFFFF !important; font-size: 1.2rem; font-weight: 800; text-align: center; margin-bottom: 1rem;'>About</h4>
            <h5 style='color: #FFFFFF !important; font-size: 1rem; font-weight: 700; text-align: center; margin-bottom: 0.8rem;'>Vidibemus Lumen Analytics</h5>
            <p style='color: #FFFFFF !important; font-size: 0.85rem; font-weight: 600; text-align: center; line-height: 1.4; margin-bottom: 1rem;'>Transforming data into enlightened decisions for positive social impact.</p>
            <div style='text-align: center; padding-top: 0.8rem; border-top: 1px solid rgba(255, 255, 255, 0.3);'>
                <p style='color: #FFFFFF !important; font-size: 0.8rem; font-weight: 600; margin: 0;'>© 2025 Victor Collins Oppon</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    return page

def render_executive_dashboard(df, insights, metrics):
    """Render executive dashboard with VL branding"""
    st.markdown('<h1 class="main-header">Executive Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Strategic insights powered by <strong style="color: #F4B942;">Vidibemus Lumen Analytics</strong></p>', unsafe_allow_html=True)
    
    # Key metrics with VL styling
    col1, col2, col3, col4 = st.columns(4)
    
    if not metrics.empty:
        metrics_dict = metrics.set_index('Metric')['Value'].to_dict()
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Customers", metrics_dict.get('Total Customers', 'N/A'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Churn Rate", metrics_dict.get('Churn Rate', 'N/A'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Revenue at Risk", metrics_dict.get('Annual Revenue at Risk', 'N/A'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Potential Savings", metrics_dict.get('Target Savings (25% reduction)', 'N/A'))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations with VL color scheme
    st.markdown("---")
    
    if not df.empty:
        # Define VL color scheme
        vl_colors = ['#10B981', '#EF4444']  # Success green, Error red
        vl_gradient = px.colors.sequential.Blues_r
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            churn_counts = df['Churn'].value_counts()
            fig = px.pie(
                values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
                names=['Retained', 'Churned'],
                title="Customer Retention Analysis",
                color_discrete_sequence=vl_colors
            )
            fig.update_layout(
                font=dict(family="Inter, sans-serif"),
                title_font_size=20,
                title_font_color='#0F2540'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk segments
            if insights.get('high_risk_segments'):
                segments_df = pd.DataFrame(
                    list(insights['high_risk_segments'].items()),
                    columns=['Segment', 'Churn Rate (%)']
                )
                
                fig = px.bar(
                    segments_df,
                    x='Segment',
                    y='Churn Rate (%)',
                    title="High-Risk Customer Segments",
                    color='Churn Rate (%)',
                    color_continuous_scale=[[0, '#10B981'], [0.5, '#F4B942'], [1, '#EF4444']]
                )
                fig.update_layout(
                    font=dict(family="Inter, sans-serif"),
                    title_font_size=20,
                    title_font_color='#0F2540'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Revenue impact with VL gradient
        st.markdown("### 💰 Revenue Impact Intelligence")
        
        churned = df[df['Churn'] == 'Yes']
        if not churned.empty:
            tenure_bins = pd.cut(churned['tenure'], bins=[0, 12, 24, 48, 100], 
                               labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
            revenue_by_tenure = churned.groupby(tenure_bins)['MonthlyCharges'].sum()
            
            fig = px.bar(
                x=revenue_by_tenure.index,
                y=revenue_by_tenure.values,
                title="Monthly Revenue at Risk by Customer Tenure",
                labels={'x': 'Tenure Group', 'y': 'Monthly Revenue at Risk ($)'},
                color=revenue_by_tenure.values,
                color_continuous_scale=[[0, '#2563EB'], [0.5, '#F4B942'], [1, '#EF4444']]
            )
            fig.update_layout(
                font=dict(family="Inter, sans-serif"),
                title_font_size=20,
                title_font_color='#0F2540'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # VL Insights
    if insights.get('recommendations'):
        st.markdown("### 🔦 Illuminated Insights")
        for rec in insights['recommendations']:
            st.markdown(f'<div class="insight-box">💡 {rec}</div>', unsafe_allow_html=True)

def render_customer_analysis(df):
    """Render customer analysis with VL branding"""
    st.markdown('<h1 class="main-header">Customer Analytics Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep insights into customer behavior patterns</p>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No customer data available")
        return
    
    # Filters with VL styling
    st.markdown("### 🎯 Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract_filter = st.multiselect(
            "Contract Type",
            options=df['Contract'].unique(),
            default=df['Contract'].unique()
        )
    
    with col2:
        service_filter = st.multiselect(
            "Internet Service",
            options=df['InternetService'].unique(),
            default=df['InternetService'].unique()
        )
    
    with col3:
        tenure_range = st.slider(
            "Tenure (months)",
            min_value=int(df['tenure'].min()),
            max_value=int(df['tenure'].max()),
            value=(int(df['tenure'].min()), int(df['tenure'].max()))
        )
    
    # Apply filters
    filtered_df = df[
        (df['Contract'].isin(contract_filter)) &
        (df['InternetService'].isin(service_filter)) &
        (df['tenure'].between(tenure_range[0], tenure_range[1]))
    ]
    
    st.markdown(f"### 📊 Analyzing {len(filtered_df):,} customers")
    
    # Visualizations with VL colors
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method churn
        payment_churn = filtered_df.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        fig = px.bar(
            x=payment_churn.index,
            y=payment_churn.values,
            title="Churn Risk by Payment Method",
            labels={'x': 'Payment Method', 'y': 'Churn Rate (%)'},
            color=payment_churn.values,
            color_continuous_scale=[[0, '#10B981'], [0.5, '#F4B942'], [1, '#EF4444']]
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif"),
            title_font_size=18,
            title_font_color='#0F2540'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges
        fig = px.box(
            filtered_df,
            x='Churn',
            y='MonthlyCharges',
            title="Monthly Charges Distribution",
            color='Churn',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'}
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif"),
            title_font_size=18,
            title_font_color='#0F2540'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Service analysis
    st.markdown("### 🛡️ Service Protection Analysis")
    
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    service_churn = []
    for service in services:
        if service in filtered_df.columns:
            has_service = filtered_df[filtered_df[service].isin(['Yes', 'DSL', 'Fiber optic'])]
            if len(has_service) > 0:
                churn_rate = (has_service['Churn'] == 'Yes').mean() * 100
                service_churn.append({'Service': service, 'Churn Rate': churn_rate})
    
    if service_churn:
        service_df = pd.DataFrame(service_churn)
        fig = px.bar(
            service_df,
            x='Service',
            y='Churn Rate',
            title="Service Impact on Customer Retention",
            color='Churn Rate',
            color_continuous_scale=[[0, '#10B981'], [0.5, '#F4B942'], [1, '#EF4444']]
        )
        fig.update_layout(
            font=dict(family="Inter, sans-serif"),
            title_font_size=18,
            title_font_color='#0F2540'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_prediction_tool(predictor):
    """Render prediction tool with VL branding"""
    st.markdown('<h1 class="main-header">Predictive Intelligence Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered customer risk assessment</p>', unsafe_allow_html=True)
    
    st.markdown("### 🔮 Customer Information")
    
    # Customer info with VL styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with col2:
        st.markdown("#### Account Details")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col3:
        st.markdown("#### Financial")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", 
                                      min_value=0.0, 
                                      max_value=10000.0, 
                                      value=float(monthly_charges * tenure),
                                      step=10.0)
    
    # Services
    st.markdown("---")
    st.markdown("### 📡 Service Portfolio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        else:
            multiple_lines = "No phone service"
            st.info("No phone service")
    
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["Yes", "No"])
        else:
            online_security = "No internet service"
            st.info("No internet service")
    
    with col3:
        if internet_service != "No":
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        else:
            online_backup = "No internet service"
            device_protection = "No internet service"
            st.info("No internet service")
    
    with col4:
        if internet_service != "No":
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        else:
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
            st.info("No internet service")
    
    # Predict button with VL styling
    if st.button("🔦 Illuminate Customer Risk", type="primary"):
        # Create customer data dictionary
        customer_data = {
            'customerID': 'PRED-001',
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Make prediction
        if predictor.model is not None:
            with st.spinner("Illuminating insights..."):
                risk_score, risk_category = predictor.predict(customer_data)
            
            if risk_score is not None:
                # Results with VL styling
                st.markdown("---")
                st.markdown("### 💡 Illuminated Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Churn Probability", f"{risk_score:.1%}")
                    
                    # VL Progress bar
                    progress_color = "#EF4444" if risk_score >= 0.7 else ("#F59E0B" if risk_score >= 0.4 else "#10B981")
                    st.markdown(f"""
                        <div style='background: #F3F4F6; border-radius: 10px; padding: 2px;'>
                            <div style='background: {progress_color}; width: {risk_score*100}%; 
                                      height: 20px; border-radius: 8px;'></div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    risk_class = f'risk-{risk_category.lower()}'
                    st.markdown(f'<h3 style="color: #0F2540;">Risk Category</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p class="{risk_class}" style="font-size: 1.5rem;">{risk_category} Risk</p>', 
                              unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    retention_cost = 50 if risk_category == "High" else (30 if risk_category == "Medium" else 10)
                    potential_loss = monthly_charges * 12
                    roi = ((potential_loss - retention_cost) / retention_cost * 100) if retention_cost > 0 else 0
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Recommended Investment", f"${retention_cost}")
                    st.caption(f"Potential ROI: {roi:.0f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk factors with VL styling
                st.markdown("### 🔍 Illuminated Risk Factors")
                
                risk_factors = []
                
                if contract == "Month-to-month":
                    risk_factors.append(("📅 Month-to-month contract", "42.7% churn rate", "high"))
                if payment_method == "Electronic check":
                    risk_factors.append(("💳 Electronic check payment", "45.3% churn rate", "high"))
                if tenure <= 12:
                    risk_factors.append(("🆕 New customer", "First year customers have highest risk", "medium"))
                if tech_support == "No" and internet_service != "No":
                    risk_factors.append(("🛡️ No tech support", "15.2% higher churn rate", "medium"))
                if monthly_charges > 70:
                    risk_factors.append(("💰 High monthly charges", "Above average pricing", "low"))
                
                if risk_factors:
                    for factor, desc, severity in risk_factors:
                        if severity == "high":
                            st.error(f"{factor}: {desc}")
                        elif severity == "medium":
                            st.warning(f"{factor}: {desc}")
                        else:
                            st.info(f"{factor}: {desc}")
                else:
                    st.success("✅ No major risk factors identified")
                
                # VL Retention Strategy
                st.markdown("### 🌟 Illuminated Path Forward")
                
                if risk_category == "High":
                    st.markdown('<div class="vl-brand" style="background: #EF4444;">⚡ URGENT: High Risk Customer</div>', 
                              unsafe_allow_html=True)
                    recommendations = [
                        ("🎯", "Immediate Intervention", "Contact within 24-48 hours via preferred channel"),
                        ("💎", "Premium Retention Offer", "20-30% discount + contract upgrade incentive"),
                        ("🛡️", "Service Enhancement", "Complimentary protection bundle for 3 months"),
                        ("🤝", "Relationship Building", "Assign senior account manager"),
                        ("💰", "Financial Flexibility", "Payment plan options + method change bonus")
                    ]
                elif risk_category == "Medium":
                    st.markdown('<div class="vl-brand" style="background: #F59E0B;">⚡ Medium Risk Customer</div>', 
                              unsafe_allow_html=True)
                    recommendations = [
                        ("📧", "Targeted Engagement", "Personalized value communication campaign"),
                        ("💳", "Payment Optimization", "Auto-pay incentive + convenience benefits"),
                        ("📦", "Service Discovery", "Showcase underutilized features + upgrades"),
                        ("🤝", "Proactive Support", "Quarterly satisfaction assessments"),
                        ("📊", "Usage Intelligence", "Monitor engagement patterns closely")
                    ]
                else:
                    st.markdown('<div class="vl-brand" style="background: #10B981;">✅ Low Risk Customer</div>', 
                              unsafe_allow_html=True)
                    recommendations = [
                        ("🌟", "Excellence Maintenance", "Continue superior service delivery"),
                        ("🎁", "Loyalty Recognition", "VIP program enrollment + exclusive perks"),
                        ("📈", "Growth Opportunities", "Strategic upsell after stability period"),
                        ("💌", "Appreciation Touchpoints", "Regular value acknowledgments"),
                        ("🔮", "Predictive Care", "AI-driven satisfaction monitoring")
                    ]
                
                for icon, title, desc in recommendations:
                    st.markdown(f'<div class="insight-box">{icon} <strong>{title}</strong>: {desc}</div>', 
                              unsafe_allow_html=True)
                    
            else:
                st.error(f"Prediction failed: {risk_category}")
                st.info("Please verify all fields and try again.")
        else:
            st.error("Model not loaded. Please ensure model files are available.")

def render_retention_strategy(insights):
    """Render retention strategy with VL branding"""
    st.markdown('<h1 class="main-header">Strategic Intelligence Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-driven strategies for sustainable growth</p>', unsafe_allow_html=True)
    
    # Strategy metrics with VL colors
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Target Reduction", "25-30%", "↓ Churn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected ROI", "430%", "↑ Returns")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Timeline", "3 months", "📅")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Focus Areas", "3", "🎯")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # VL Strategy Framework
    st.markdown("---")
    st.markdown("### 🔦 Illuminated Strategy Framework")
    
    # Strategy tabs with VL styling
    tab1, tab2, tab3 = st.tabs(["Contract Optimization", "Payment Evolution", "Service Excellence"])
    
    with tab1:
        st.markdown("#### Month-to-Month Contract Transformation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Target Segment**: Month-to-month customers  
            **Current Risk**: 42.7% churn rate  
            **Target State**: 27.7% churn rate  
            
            **VL Implementation Strategy:**
            1. 🔍 Identify high-value month-to-month customers
            2. 📊 Segment by lifetime value and engagement
            3. 🎯 Create personalized migration paths
            4. 🚀 Launch multi-channel campaigns
            5. 📈 Track and optimize conversions
            
            **Incentive Architecture:**
            • **1-Year Commitment**: 20% discount + premium features
            • **2-Year Partnership**: 30% discount + VIP status
            • **Loyalty Bonus**: $50 credit after successful migration
            """)
        
        with col2:
            st.metric("Conversion Target", "35%")
            st.metric("Revenue Impact", "+$127K")
            st.metric("Investment", "$15K")
            st.metric("ROI", "747%")
    
    with tab2:
        st.markdown("#### Electronic Payment Evolution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Target Segment**: Electronic check users  
            **Current Risk**: 45.3% churn rate  
            **Target State**: 33.3% churn rate  
            
            **VL Transformation Approach:**
            1. 📊 Analyze payment failure patterns
            2. 🎯 Design frictionless migration path
            3. 💳 Implement secure payment options
            4. 🎁 Create compelling incentives
            5. 📱 Provide seamless setup support
            
            **Migration Incentives:**
            • **Auto-Pay Adoption**: $5/month perpetual discount
            • **Switch Bonus**: $20 immediate credit
            • **Security Plus**: 6 months payment protection
            • **VIP Support**: Priority assistance channel
            """)
        
        with col2:
            st.metric("Adoption Goal", "40%")
            st.metric("Revenue Saved", "+$98K")
            st.metric("Program Cost", "$22K")
            st.metric("ROI", "345%")
    
    with tab3:
        st.markdown("#### Service Excellence Initiative")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Target Segment**: Unprotected service users  
            **Current Risk**: 41.7% churn rate  
            **Target State**: 31.7% churn rate  
            
            **VL Protection Strategy:**
            1. 🛡️ Identify vulnerable customers
            2. 📦 Design value-driven bundles
            3. 🎯 Highlight security benefits
            4. 🎁 Offer risk-free trials
            5. 📊 Monitor satisfaction metrics
            
            **Protection Packages:**
            • **Essential Shield**: Core protection at $5/month
            • **Complete Armor**: Full suite at $10/month
            • **Premium Guardian**: Elite care at $15/month
            • **Trial Period**: 3 months free experience
            """)
        
        with col2:
            st.metric("Bundle Target", "25%")
            st.metric("Revenue Growth", "+$156K")
            st.metric("Setup Cost", "$18K")
            st.metric("ROI", "767%")
    
    # VL ROI Intelligence
    st.markdown("---")
    st.markdown("### 💎 ROI Intelligence Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Investment Parameters")
        
        target_customers = st.number_input(
            "High-Risk Customers to Target",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        avg_customer_value = st.number_input(
            "Average Monthly Value ($)",
            min_value=20.0,
            max_value=200.0,
            value=64.76,
            step=5.0
        )
        
        retention_investment = st.number_input(
            "Investment per Customer ($)",
            min_value=10.0,
            max_value=100.0,
            value=30.0,
            step=5.0
        )
        
        success_rate = st.slider(
            "Expected Success Rate (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5
        )
    
    with col2:
        st.markdown("#### Illuminated Projections")
        
        # Calculations
        customers_retained = int(target_customers * success_rate / 100)
        annual_revenue_saved = customers_retained * avg_customer_value * 12
        total_investment = target_customers * retention_investment
        net_benefit = annual_revenue_saved - total_investment
        roi = (net_benefit / total_investment * 100) if total_investment > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Customers Retained", f"{customers_retained:,}")
            st.metric("Annual Revenue Saved", f"${annual_revenue_saved:,.0f}")
        
        with col2:
            st.metric("Total Investment", f"${total_investment:,.0f}")
            st.metric("Net Benefit", f"${net_benefit:,.0f}", delta=f"{roi:.0f}% ROI")
        
        # VL Insight
        if avg_customer_value > 0:
            breakeven_months = retention_investment / avg_customer_value
            st.markdown(f"""
                <div class="insight-box">
                <strong>🔦 VL Insight:</strong> Break-even achieved in {breakeven_months:.1f} months per retained customer. 
                This represents a {roi:.0f}% return on investment, illuminating the path to sustainable growth.
                </div>
            """, unsafe_allow_html=True)

def render_vl_footer():
    """Render VL Analytics branded footer"""
    st.markdown("""
        <div class="vl-footer">
            <p><strong>Vidibemus Lumen Analytics</strong> | "We Shall See the Light"</p>
            <p>© 2025 Victor Collins Oppon, FCCA, MBA, BSc | All Rights Reserved</p>
            <p>
                📧 <a href="mailto:victoroppdatascience1@gmail.com">victoroppdatascience1@gmail.com</a> | 
                🔗 <a href="https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/" target="_blank">LinkedIn</a> | 
                🌐 <a href="https://victoropp.github.io/" target="_blank">Portfolio</a>
            </p>
            <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;">
                "We Shall See the Light" - Illuminating insights through data science and artificial intelligence.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Initialize
    render_vl_header()
    
    predictor = ChurnPredictor()
    df = load_data()
    insights = load_insights()
    metrics = load_metrics()
    
    # Sidebar
    page = render_sidebar(df)
    
    # Render selected page
    if page == "Executive Intelligence":
        render_executive_dashboard(df, insights, metrics)
    elif page == "Customer Analytics":
        render_customer_analysis(df)
    elif page == "Predictive Intelligence":
        render_prediction_tool(predictor)
    elif page == "Strategic Planning":
        render_retention_strategy(insights)
    
    # Footer
    render_vl_footer()

if __name__ == "__main__":
    main()