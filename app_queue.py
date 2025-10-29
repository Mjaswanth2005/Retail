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

# Page config
st.set_page_config(
    page_title="OptiQueue - Queue Optimization Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'cameras' not in st.session_state:
    st.session_state.cameras = []
if 'models' not in st.session_state:
    st.session_state.models = [{'name': 'best.pt', 'status': 'Active', 'accuracy': '94.2%'}]
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'avg_queue_length' not in st.session_state:
    st.session_state.avg_queue_length = random.randint(8, 15)
if 'avg_wait_time' not in st.session_state:
    st.session_state.avg_wait_time = random.randint(3, 8)
if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = random.randint(0, 3)

# Custom CSS for dark theme with glowing effects
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #181825 0%, #0e1117 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
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
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin-top: 0.8rem;
        font-weight: 300;
    }
    
    /* KPI Cards with glow effect */
    .kpi-card {
        background: linear-gradient(135deg, #1a1b26 0%, #24253a 100%);
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    0 0 20px rgba(99, 102, 241, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4),
                    0 0 30px rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.6);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
    }
    
    .kpi-title {
        color: #9ca3af;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }
    
    .kpi-value {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-icon {
        font-size: 2rem;
        opacity: 0.8;
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
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }
    
    /* Sidebar navigation buttons */
    .nav-button {
        background: rgba(99, 102, 241, 0.1);
        color: #e0e0e0;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid transparent;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-weight: 500;
    }
    
    .nav-button:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(5px);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-color: #6366f1;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    /* Metrics with dark theme */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1b26 0%, #24253a 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem;
        font-weight: 700;
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
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .status-active {
        background: rgba(99, 102, 241, 0.2);
        color: #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    /* Alert badge */
    .alert-badge {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
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
    
    /* Input fields */
    .stTextInput input, .stSelectbox select {
        background-color: #1a1b26 !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #1a1b26 0%, #24253a 100%);
        border: 2px dashed rgba(99, 102, 241, 0.5);
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Plotly dark theme */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #e0e0e0;
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
    
    # Navigation menu
    menu_items = [
        ("📊", "Dashboard"),
        ("📷", "Cameras"),
        ("🤖", "Models"),
        ("📤", "Upload/Test"),
        ("📈", "Analytics"),
        ("🔍", "Monitoring"),
        ("⚙️", "Settings"),
        ("🚪", "Logout")
    ]
    
    for icon, label in menu_items:
        active_class = "active" if st.session_state.page == label else ""
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

# Helper functions
def generate_queue_data():
    """Generate simulated queue data for charts"""
    hours = list(range(24))
    queue_lengths = [random.randint(2, 20) for _ in hours]
    wait_times = [random.randint(1, 10) for _ in hours]
    return hours, queue_lengths, wait_times

def create_line_chart(hours, values, title, y_label):
    """Create a styled line chart for analytics"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=values,
        mode='lines+markers',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=8, color='#8b5cf6'),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)'
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

def load_yolo_model():
    """Load YOLO model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        return None

# Main content based on selected page
if st.session_state.page == "Dashboard":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 OptiQueue Dashboard</h1>
        <p>Real-time Queue Monitoring & Optimization Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">👥</div>
            <div class="kpi-title">Avg Queue Length</div>
            <div class="kpi-value">{st.session_state.avg_queue_length}</div>
            <p style="color: #10b981; font-size: 0.9rem; margin: 0;">↓ 12% from yesterday</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">⏱️</div>
            <div class="kpi-title">Avg Wait Time</div>
            <div class="kpi-value">{st.session_state.avg_wait_time} min</div>
            <p style="color: #10b981; font-size: 0.9rem; margin: 0;">↓ 8% from yesterday</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        peak_hour = f"{random.randint(14, 18)}:00"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📊</div>
            <div class="kpi-title">Peak Hour</div>
            <div class="kpi-value">{peak_hour}</div>
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">Today's peak time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        alert_color = "#ef4444" if st.session_state.active_alerts > 0 else "#10b981"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🚨</div>
            <div class="kpi-title">Active Alerts</div>
            <div class="kpi-value" style="color: {alert_color};">{st.session_state.active_alerts}</div>
            <p style="color: {alert_color}; font-size: 0.9rem; margin: 0;">
                {"⚠️ Requires attention" if st.session_state.active_alerts > 0 else "✓ All clear"}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
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
            st.session_state.page = "Cameras"
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">📝 Recent Detections</div>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.detections) == 0:
            st.info("No recent detections. Upload images to start detecting.")
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
        fig2 = create_line_chart(hours, wait_times, "", "Wait Time (min)")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Cameras":
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
            camera_url = st.text_input("RTSP URL", placeholder="rtsp://...")
            
            submitted = st.form_submit_button("Add Camera", use_container_width=True)
            if submitted and camera_name and camera_location:
                st.session_state.cameras.append({
                    'name': camera_name,
                    'location': camera_location,
                    'url': camera_url,
                    'status': 'Active'
                })
                st.success(f"✅ Camera '{camera_name}' added successfully!")
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

elif st.session_state.page == "Models":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Management</h1>
        <p>Manage your detection models and configurations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📤 Upload Model</div>', unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader("Upload YOLO Model (.pt)", type=['pt'])
        if uploaded_model:
            st.success(f"Model '{uploaded_model.name}' ready to upload")
            if st.button("Save Model", use_container_width=True):
                st.session_state.models.append({
                    'name': uploaded_model.name,
                    'status': 'Inactive',
                    'accuracy': 'N/A'
                })
                st.success("Model uploaded successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="panel-header">🗂️ Available Models ({len(st.session_state.models)})</div>', unsafe_allow_html=True)
        
        for idx, model in enumerate(st.session_state.models):
            status_class = "status-active" if model['status'] == 'Active' else "status-warning"
            st.markdown(f"""
            <div class="camera-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="color: #ffffff; margin: 0;">🤖 {model['name']}</h3>
                        <p style="color: #9ca3af; margin: 0.5rem 0;">
                            Accuracy: {model['accuracy']} | Status: <span class="status-chip {status_class}">{model['status']}</span>
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Upload/Test":
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
        
        input_type = st.radio("Select Input Type", ["📷 Image", "🎥 Video"], horizontal=True)
        
        if "Image" in input_type:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                col1, col2 = st.columns(2, gap="large")
                
                image = Image.open(uploaded_file)
                
                with col1:
                    st.markdown("#### 🖼️ Original Image")
                    st.image(image, use_container_width=True)
                
                if st.button("🚀 Run Detection", use_container_width=True):
                    with st.spinner("🔍 Analyzing..."):
                        start_time = time.time()
                        results = model.predict(image, conf=0.25)
                        processing_time = time.time() - start_time
                        
                        with col2:
                            st.markdown("#### ✨ Detection Results")
                            annotated = results[0].plot()
                            st.image(annotated, use_container_width=True)
                        
                        # Add to detections
                        if len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                st.session_state.detections.append({
                                    'class': results[0].names[cls_id],
                                    'confidence': conf,
                                    'time': datetime.now().strftime("%H:%M:%S")
                                })
                        
                        # Metrics
                        st.markdown("---")
                        met1, met2, met3 = st.columns(3)
                        with met1:
                            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
                        with met2:
                            st.metric("🎯 Detections", len(results[0].boxes))
                        with met3:
                            if len(results[0].boxes) > 0:
                                avg_conf = np.mean([float(box.conf[0]) for box in results[0].boxes])
                                st.metric("📊 Avg Confidence", f"{avg_conf:.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:  # Video
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            
            if uploaded_video:
                st.video(uploaded_video)
                
                if st.button("🚀 Process Video", use_container_width=True):
                    with st.spinner("🎬 Processing video..."):
                        st.info("Video processing feature ready. Processing may take several minutes.")
            
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Analytics":
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
    hours, queue_lengths, wait_times = generate_queue_data()
    
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
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">⏱️ Wait Time Analysis</div>', unsafe_allow_html=True)
        
        fig = create_line_chart(hours, wait_times, "", "Wait Time (minutes)")
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

elif st.session_state.page == "Monitoring":
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

elif st.session_state.page == "Settings":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Settings</h1>
        <p>Configure system preferences and options</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🎨 Appearance</div>', unsafe_allow_html=True)
        
        theme = st.selectbox("Theme", ["Dark (Current)", "Light"])
        st.slider("UI Scale", 80, 120, 100, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🔔 Notifications</div>', unsafe_allow_html=True)
        
        st.checkbox("Enable Email Notifications", value=True)
        st.checkbox("Enable SMS Alerts", value=False)
        st.checkbox("Desktop Notifications", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🤖 Detection Settings</div>', unsafe_allow_html=True)
        
        st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
        st.number_input("Max Queue Alert", min_value=5, max_value=50, value=15)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🔐 API Configuration</div>', unsafe_allow_html=True)
        
        st.text_input("API Key", type="password", placeholder="Enter API key")
        st.text_input("Webhook URL", placeholder="https://...")
        
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:  # Logout
    st.markdown("""
    <div class="main-header">
        <h1>🚪 Logout</h1>
        <p>Thank you for using OptiQueue</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="panel-card" style="text-align: center; padding: 4rem 2rem;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff;">Are you sure you want to logout?</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚪 Yes, Logout", use_container_width=True):
            st.session_state.clear()
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
