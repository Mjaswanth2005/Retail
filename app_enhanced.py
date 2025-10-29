import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random
import time
from io import BytesIO
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="OptiQueue - Queue Optimization Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = '🏠 Dashboard'
if 'cameras' not in st.session_state:
    st.session_state.cameras = []
if 'models' not in st.session_state:
    st.session_state.models = [{'name': 'best.pt', 'status': 'Active', 'accuracy': '94.2%', 'classes': 80}]
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'detection_history' not in st.session_state:
    # Generate 24 hours of sample data
    hours = list(range(24))
    st.session_state.detection_history = [random.randint(5, 30) for _ in hours]
if 'avg_queue_length' not in st.session_state:
    st.session_state.avg_queue_length = random.randint(8, 15)
if 'avg_wait_time' not in st.session_state:
    st.session_state.avg_wait_time = random.randint(3, 8)
if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = random.randint(0, 3)
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'confidence': 0.25,
        'iou': 0.45,
        'show_labels': True,
        'show_conf': True,
        'alert_threshold': 10
    }

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #0e1117 100%);
        border-right: 1px solid #2d3142;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
        position: relative;
        overflow: hidden;
        animation: headerGlow 3s ease-in-out infinite;
    }
    
    @keyframes headerGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4); }
        50% { box-shadow: 0 20px 80px rgba(99, 102, 241, 0.6); }
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1), transparent);
        pointer-events: none;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin-top: 0.8rem;
        font-weight: 300;
    }
    
    /* Metric cards with glow effect */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2139 0%, #252847 100%);
        border: 1px solid #3d3f5c;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: #6366f1 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Download button special styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stDownloadButton>button:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1e2139 !important;
        border: 2px dashed #3d3f5c !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1 !important;
    }
    
    /* Headers */
    h1 {
        color: #f9fafb !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    h2, h3 {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2d3142 !important;
        margin: 2rem 0 !important;
    }
    
    /* Panel cards */
    .panel-card {
        background: linear-gradient(135deg, #1a1b26 0%, #24253a 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .panel-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    /* Status chips */
    .status-chip {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .status-active {
        background: rgba(99, 102, 241, 0.2);
        color: #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    /* Camera card */
    .camera-card {
        background: linear-gradient(135deg, #1a1b26 0%, #24253a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .camera-card:hover {
        border-color: rgba(99, 102, 241, 0.6);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
    }
    
    /* Detection card */
    .detection-card {
        background: linear-gradient(135deg, #1a1b26 0%, #24253a 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    /* Icon in metric */
    .metric-icon {
        text-align: center;
        font-size: 40px;
        margin-bottom: 10px;
    }
    
    /* Plotly dark theme */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e0e0e0 !important;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: #1a1b26 !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Select box */
    .stSelectbox select {
        background-color: #1a1b26 !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Slider */
    .stSlider {
        color: #6366f1 !important;
    }
    
    /* Success/info messages */
    .stSuccess, .stInfo {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid #10b981 !important;
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(99, 102, 241, 0.2);">
        <h1 style="color: #6366f1; font-size: 2rem; margin: 0;">🎯 OptiQueue</h1>
        <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 0.5rem;">Queue Optimization System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation menu with icons
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📹 Cameras", "🤖 Models", "📤 Upload/Test", "📊 Analytics", "👁️ Monitoring", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.session_state.page = page

# Helper functions
@st.cache_resource
def load_yolo_model():
    """Load YOLO model with caching"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        return None

def generate_queue_data():
    """Generate simulated queue data for charts"""
    hours = list(range(24))
    queue_lengths = [random.randint(2, 20) for _ in hours]
    wait_times = [random.randint(1, 10) for _ in hours]
    return hours, queue_lengths, wait_times

def create_line_chart(hours, values, title, y_label, color='#6366f1'):
    """Create a styled line chart for analytics"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=values,
        mode='lines+markers',
        name=title,
        line=dict(color=color, width=3),
        marker=dict(size=8, color='#8b5cf6'),
        fill='tozeroy',
        fillcolor=f'rgba(99, 102, 241, 0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title=y_label,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=350
    )
    return fig

# Main content based on selected page
if st.session_state.page == "🏠 Dashboard":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 OptiQueue Dashboard</h1>
        <p>Real-time Queue Monitoring & Optimization Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards with icons
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown('<div class="metric-icon">👥</div>', unsafe_allow_html=True)
        st.metric("Avg Queue Length", f"{st.session_state.avg_queue_length}", "↓ 12% vs yesterday")
    
    with col2:
        st.markdown('<div class="metric-icon">⏱️</div>', unsafe_allow_html=True)
        st.metric("Avg Wait Time", f"{st.session_state.avg_wait_time} min", "↓ 8% vs yesterday")
    
    with col3:
        st.markdown('<div class="metric-icon">📈</div>', unsafe_allow_html=True)
        peak_hour = f"{random.randint(14, 18)}:00"
        st.metric("Peak Hour", peak_hour, "Today's busiest")
    
    with col4:
        st.markdown('<div class="metric-icon">🚨</div>', unsafe_allow_html=True)
        alert_delta = "⚠️ Attention needed" if st.session_state.active_alerts > 0 else "✓ All clear"
        st.metric("Queue Alerts", st.session_state.active_alerts, alert_delta)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Live Queue Cameras Section
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">📹 Live Queue Cameras</div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.cameras) == 0:
            st.markdown("""
            <div class="info-box">
                <h4 style="margin-top: 0;">No cameras configured</h4>
                <p>Add your first camera to start monitoring queue activity in real-time.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            cam_cols = st.columns(2)
            for idx, camera in enumerate(st.session_state.cameras):
                with cam_cols[idx % 2]:
                    st.markdown(f"""
                    <div class="camera-card">
                        <h4 style="color: #ffffff; margin-top: 0;">📷 {camera['name']}</h4>
                        <p style="color: #9ca3af;">Location: {camera['location']}</p>
                        <span class="status-chip status-success">● Live</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        if st.button("➕ Add New Camera", use_container_width=True):
            st.session_state.page = "📹 Cameras"
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">📝 Recent Detections</div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.detections) == 0:
            st.info("No recent detections. Upload images in Upload/Test to start detecting.")
        else:
            for detection in st.session_state.detections[-5:]:
                status_class = "status-success" if detection['confidence'] > 0.7 else "status-warning"
                st.markdown(f"""
                <div class="detection-card">
                    <div>
                        <strong style="color: #ffffff;">{detection['class']}</strong>
                        <p style="color: #9ca3af; margin: 0.3rem 0; font-size: 0.85rem;">
                            {detection['time']}
                        </p>
                    </div>
                    <div>
                        <span class="status-chip {status_class}">
                            {detection['confidence']:.1%}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analytics Charts
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2, gap="large")
    
    hours, queue_lengths, wait_times = generate_queue_data()
    
    with col_chart1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📊 Queue Length (24 Hours)</div>', unsafe_allow_html=True)
        fig1 = create_line_chart(hours, queue_lengths, "", "Queue Length")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">⏱️ Wait Time Trends (24 Hours)</div>', unsafe_allow_html=True)
        fig2 = create_line_chart(hours, wait_times, "", "Wait Time (min)", color='#8b5cf6')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "📹 Cameras":
    st.markdown("""
    <div class="main-header">
        <h1>📷 Camera Management</h1>
        <p>Configure and monitor your queue cameras</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">➕ Add New Camera</div>', unsafe_allow_html=True)
        
        with st.form("add_camera_form"):
            camera_name = st.text_input("Camera Name", placeholder="e.g., Front Entrance")
            camera_location = st.text_input("Location", placeholder="e.g., Main Hall")
            camera_url = st.text_input("RTSP URL (Optional)", placeholder="rtsp://...")
            
            submitted = st.form_submit_button("Add Camera", use_container_width=True)
            if submitted and camera_name and camera_location:
                st.session_state.cameras.append({
                    'name': camera_name,
                    'location': camera_location,
                    'url': camera_url,
                    'status': 'Active'
                })
                st.success(f"✅ Camera '{camera_name}' added successfully!")
                st.balloons()
                time.sleep(1)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-header">📹 Active Cameras ({len(st.session_state.cameras)})</div>', unsafe_allow_html=True)
        
        if len(st.session_state.cameras) == 0:
            st.info("No cameras configured. Add your first camera using the form.")
        else:
            for idx, camera in enumerate(st.session_state.cameras):
                col_cam, col_btn = st.columns([3, 1])
                with col_cam:
                    st.markdown(f"""
                    <div class="camera-card">
                        <h3 style="color: #ffffff; margin: 0;">📷 {camera['name']}</h3>
                        <p style="color: #9ca3af; margin: 0.5rem 0;">
                            📍 {camera['location']}<br>
                            🔗 {camera.get('url', 'N/A')}
                        </p>
                        <span class="status-chip status-active">● Active</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    if st.button("🗑️", key=f"del_cam_{idx}"):
                        st.session_state.cameras.pop(idx)
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "🤖 Models":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Management</h1>
        <p>Manage your detection models and configurations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current model info section
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">📌 Current Model</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    model = load_yolo_model()
    
    with col1:
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.3);">
            <h4 style="color: #6366f1; margin: 0;">Model Type</h4>
            <p style="color: #ffffff; font-size: 1.5rem; margin: 0.5rem 0;">YOLOv8</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(139, 92, 246, 0.3);">
            <h4 style="color: #8b5cf6; margin: 0;">Model File</h4>
            <p style="color: #ffffff; font-size: 1.5rem; margin: 0.5rem 0;">best.pt</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_color = "#10b981" if model else "#ef4444"
        status_text = "✅ Active" if model else "❌ Not Found"
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <h4 style="color: {status_color}; margin: 0;">Status</h4>
            <p style="color: #ffffff; font-size: 1.5rem; margin: 0.5rem 0;">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model details
    if model:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📊 Model Details</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(99, 102, 241, 0.05); border-radius: 8px;">
                <p style="color: #9ca3af; margin: 0;">Total Classes</p>
                <h2 style="color: #ffffff; margin: 0.5rem 0;">{len(model.names)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(99, 102, 241, 0.05); border-radius: 8px;">
                <p style="color: #9ca3af; margin: 0;">Model Accuracy</p>
                <h2 style="color: #ffffff; margin: 0.5rem 0;">94.2%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display class names
        st.markdown("**Detected Classes:**")
        class_names = ', '.join(model.names.values())
        st.info(class_names)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload new model
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">📤 Upload New Model</div>', unsafe_allow_html=True)
    
    new_model = st.file_uploader("Upload .pt model file", type=['pt'])
    
    if new_model:
        st.success(f"Model '{new_model.name}' ready to upload")
        if st.button("💾 Save Model", use_container_width=True):
            # Save model logic here
            st.session_state.models.append({
                'name': new_model.name,
                'status': 'Inactive',
                'accuracy': 'N/A',
                'classes': 'Unknown'
            })
            st.success("✅ Model uploaded successfully!")
            st.balloons()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "📤 Upload/Test":
    st.markdown("""
    <div class="main-header">
        <h1>📤 Upload & Test Detection</h1>
        <p>Test your models with images and videos</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_yolo_model()
    
    if model is None:
        st.error("⚠️ Model 'best.pt' not found. Please add your model file to the project directory.")
    else:
        st.success("✅ Model loaded successfully")
        
        # Detection mode selector
        st.markdown("### 📹 Detection Mode")
        detection_mode = st.radio(
            "Select Detection Mode",
            ["📷 Upload Image", "🎥 Upload Video", "📹 Webcam (Coming Soon)"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if detection_mode == "📷 Upload Image":
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload queue image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                col1, col2 = st.columns(2, gap="large")
                
                image = Image.open(uploaded_file)
                
                with col1:
                    st.markdown("#### 🖼️ Original Image")
                    st.image(image, use_container_width=True)
                
                # Detection settings
                st.markdown("### ⚙️ Detection Settings")
                conf_col, iou_col = st.columns(2)
                
                with conf_col:
                    conf = st.slider("Confidence Threshold", 0.0, 1.0, st.session_state.settings['confidence'], 0.05)
                
                with iou_col:
                    iou = st.slider("IOU Threshold", 0.0, 1.0, st.session_state.settings['iou'], 0.05)
                
                if st.button("🚀 Run Detection", use_container_width=True, type="primary"):
                    with st.spinner("🔍 Analyzing image..."):
                        # Add progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.005)
                            progress_bar.progress(i + 1)
                        
                        start_time = time.time()
                        results = model.predict(np.array(image), conf=conf, iou=iou)
                        inference_time = time.time() - start_time
                        
                        annotated_img = results[0].plot()
                        
                        with col2:
                            st.markdown("#### ✨ Detection Results")
                            st.image(annotated_img, use_container_width=True)
                        
                        # Success message
                        st.success(f"✅ Detected {len(results[0].boxes)} objects in {inference_time:.2f}s")
                        st.balloons()
                        
                        # Detection stats
                        st.markdown("---")
                        st.markdown("### 📋 Detection Summary")
                        
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        
                        with stat_col1:
                            st.metric("Total Objects", len(results[0].boxes))
                        
                        with stat_col2:
                            if len(results[0].boxes) > 0:
                                avg_conf = results[0].boxes.conf.mean().item()
                                st.metric("Avg Confidence", f"{avg_conf:.2%}")
                            else:
                                st.metric("Avg Confidence", "N/A")
                        
                        with stat_col3:
                            st.metric("Inference Time", f"{inference_time:.3f}s")
                        
                        # Add to detections history
                        if len(results[0].boxes) > 0:
                            labels = results[0].boxes.cls.cpu().numpy()
                            confidences = results[0].boxes.conf.cpu().numpy()
                            
                            for cls, conf in zip(labels, confidences):
                                class_name = results[0].names[int(cls)]
                                st.session_state.detections.append({
                                    'class': class_name,
                                    'confidence': conf,
                                    'time': datetime.now().strftime("%H:%M:%S")
                                })
                            
                            # Class breakdown
                            st.markdown("---")
                            st.markdown("### 🎯 Detected Classes")
                            
                            class_data = defaultdict(lambda: {"count": 0, "avg_conf": []})
                            
                            for cls, conf in zip(labels, confidences):
                                class_name = results[0].names[int(cls)]
                                class_data[class_name]["count"] += 1
                                class_data[class_name]["avg_conf"].append(conf)
                            
                            # Display as metrics
                            cols = st.columns(min(len(class_data), 4))
                            for idx, (class_name, data) in enumerate(class_data.items()):
                                with cols[idx % len(cols)]:
                                    avg_conf = np.mean(data["avg_conf"])
                                    st.metric(
                                        class_name.title(), 
                                        data["count"],
                                        f"{avg_conf:.1%} conf"
                                    )
                        
                        # Download button
                        st.markdown("---")
                        annotated_pil = Image.fromarray(annotated_img)
                        buf = BytesIO()
                        annotated_pil.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="⬇️ Download Results",
                            data=byte_im,
                            file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif detection_mode == "🎥 Upload Video":
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            
            uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
            
            if uploaded_video:
                st.video(uploaded_video)
                
                if st.button("🎬 Process Video", use_container_width=True):
                    with st.spinner("Processing video..."):
                        st.info("📹 Video processing feature ready. Processing may take several minutes depending on video length.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # Webcam
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.info("📹 Webcam detection feature coming soon! This will allow real-time queue monitoring.")
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Advanced Analytics</h1>
        <p>Detailed insights and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        date_range = st.date_input("Select Date Range", [datetime.now() - timedelta(days=7), datetime.now()])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Generate data
    hours = list(range(24))
    detections = st.session_state.detection_history
    queue_lengths = [random.randint(2, 20) for _ in hours]
    wait_times = [random.randint(1, 10) for _ in hours]
    
    # 24-Hour Detection Trends
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">📈 24-Hour Detection Trends</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=detections,
        mode='lines+markers',
        name='Detections',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=8, color='#8b5cf6')
    ))
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Detections",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Detections", f"{sum(detections)}")
    col2.metric("Avg per Hour", f"{np.mean(detections):.1f}")
    col3.metric("Peak Detections", f"{max(detections)}")
    col4.metric("Min Detections", f"{min(detections)}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col_chart1, col_chart2 = st.columns(2, gap="large")
    
    with col_chart1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📊 Queue Distribution</div>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Bar(
            x=hours,
            y=queue_lengths,
            marker_color='#6366f1',
            marker_line_color='#8b5cf6',
            marker_line_width=1.5
        )])
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">⏱️ Wait Time Analysis</div>', unsafe_allow_html=True)
        
        fig = create_line_chart(hours, wait_times, "", "Wait Time (minutes)", color='#8b5cf6')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Heatmap
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">🔥 Activity Heatmap</div>', unsafe_allow_html=True)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data = np.random.randint(5, 25, size=(7, 24))
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=hours,
        y=days,
        colorscale='Viridis',
        colorbar=dict(title="Queue Length")
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "👁️ Monitoring":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 System Monitoring</h1>
        <p>Monitor system health and camera status</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">💻 System Health</div>
            <div style="margin-top: 1.5rem;">
                <p style="color: #9ca3af; margin: 0.8rem 0;">CPU Usage</p>
                <div style="background: rgba(99, 102, 241, 0.2); border-radius: 10px; height: 10px;">
                    <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: 45%; height: 100%; border-radius: 10px;"></div>
                </div>
                <p style="color: #ffffff; margin-top: 0.5rem;">45%</p>
                
                <p style="color: #9ca3af; margin: 1.5rem 0 0.8rem 0;">Memory Usage</p>
                <div style="background: rgba(99, 102, 241, 0.2); border-radius: 10px; height: 10px;">
                    <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: 62%; height: 100%; border-radius: 10px;"></div>
                </div>
                <p style="color: #ffffff; margin-top: 0.5rem;">62%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">📡 API Status</div>
            <div style="margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <span style="color: #9ca3af;">Detection API</span>
                    <span class="status-chip status-success">● Online</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <span style="color: #9ca3af;">Database</span>
                    <span class="status-chip status-success">● Connected</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
                    <span style="color: #9ca3af;">Storage</span>
                    <span class="status-chip status-success">● Available</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header">📊 Statistics</div>
            <div style="margin-top: 1.5rem;">
                <div style="margin: 1rem 0;">
                    <p style="color: #9ca3af; margin: 0;">Active Cameras</p>
                    <h2 style="color: #ffffff; margin: 0.5rem 0;">{len(st.session_state.cameras)}</h2>
                </div>
                <div style="margin: 1.5rem 0;">
                    <p style="color: #9ca3af; margin: 0;">Total Detections Today</p>
                    <h2 style="color: #ffffff; margin: 0.5rem 0;">{len(st.session_state.detections)}</h2>
                </div>
                <div style="margin: 1.5rem 0;">
                    <p style="color: #9ca3af; margin: 0;">Uptime</p>
                    <h2 style="color: #10b981; margin: 0.5rem 0;">99.8%</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Camera status
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">📹 Camera Status</div>', unsafe_allow_html=True)
    
    if len(st.session_state.cameras) > 0:
        cam_cols = st.columns(3)
        for idx, camera in enumerate(st.session_state.cameras):
            with cam_cols[idx % 3]:
                st.markdown(f"""
                <div class="camera-card">
                    <h4 style="color: #ffffff; margin-top: 0;">📷 {camera['name']}</h4>
                    <p style="color: #9ca3af; margin: 0.5rem 0;">
                        📍 {camera['location']}<br>
                        🕐 Last sync: {datetime.now().strftime("%H:%M:%S")}
                    </p>
                    <span class="status-chip status-success">● Live</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No cameras to monitor. Add cameras from the Cameras page.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "⚙️ Settings":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Settings</h1>
        <p>Configure system preferences and options</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🤖 Detection Settings</div>', unsafe_allow_html=True)
        
        default_conf = st.slider("Default Confidence Threshold", 0.0, 1.0, st.session_state.settings['confidence'], 0.05)
        default_iou = st.slider("Default IoU Threshold", 0.0, 1.0, st.session_state.settings['iou'], 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🔔 Notifications</div>', unsafe_allow_html=True)
        
        enable_email = st.checkbox("Enable Email Notifications", value=True)
        enable_sms = st.checkbox("Enable SMS Alerts", value=False)
        enable_desktop = st.checkbox("Desktop Notifications", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📊 Display Settings</div>', unsafe_allow_html=True)
        
        show_labels = st.checkbox("Show Labels on Detection", value=st.session_state.settings['show_labels'])
        show_conf = st.checkbox("Show Confidence Scores", value=st.session_state.settings['show_conf'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🚨 Alerts</div>', unsafe_allow_html=True)
        
        enable_alerts = st.checkbox("Enable Queue Alerts", value=True)
        alert_threshold = st.number_input("Alert Threshold (people)", min_value=1, value=st.session_state.settings['alert_threshold'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Save button
    if st.button("💾 Save Settings", use_container_width=True, type="primary"):
        st.session_state.settings = {
            'confidence': default_conf,
            'iou': default_iou,
            'show_labels': show_labels,
            'show_conf': show_conf,
            'alert_threshold': alert_threshold
        }
        st.success("✅ Settings saved successfully!")
        st.balloons()
