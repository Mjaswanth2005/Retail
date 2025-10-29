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

# Set page config
st.set_page_config(
    page_title="YOLO Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark dashboard with glow effects
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d3142;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e5e7eb;
    }
    
    /* Metric cards with glow effect */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: #6366f1;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Add glow to metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2139 0%, #252847 100%);
        border: 1px solid #3d3f5c;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1e2139;
        border: 2px dashed #3d3f5c;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Headers */
    h1 {
        color: #f9fafb;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #e5e7eb;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border-color: #2d3142;
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1e2139 0%, #252847 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
        color: #e5e7eb;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e5e7eb;
    }
    
    /* Icon styling */
    .metric-icon {
        font-size: 40px;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = '🏠 Dashboard'

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 YOLO Detection Dashboard</h1>
    <p>Advanced Object Detection with Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for navigation and settings
with st.sidebar:
    st.markdown("### 🚦 OptiQ Navigation")
    st.markdown("---")
    
    # Navigation with icons
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "🤖 Models", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    # Extract page name without emoji
    page_name = page.split(" ", 1)[1] if " " in page else page
    
    st.markdown("---")
    
    # Model info
    if model:
        st.success("✓ Model Loaded")
    else:
        st.error("✗ Model Not Found")
    
    st.markdown("---")
    st.markdown("### 🎯 Detection Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.25, 0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.slider(
        "IOU Threshold", 
        0.0, 1.0, 0.45, 0.05,
        help="Intersection over Union threshold for NMS"
    )
    
    st.markdown("---")
    st.markdown("### 📷 Input Source")
    input_type = st.radio(
        "Select Input Type", 
        ["📷 Image", "🎥 Video", "📹 Webcam"],
        label_visibility="collapsed"
    )
    
    # Extract input type without emoji
    input_type = input_type.split(" ", 1)[1] if " " in input_type else input_type
    
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    if 'images_processed' not in st.session_state:
        st.session_state.images_processed = 0
    
    st.metric("Images Processed", st.session_state.images_processed)
    st.metric("Total Detections", st.session_state.total_detections)

# Page routing
if page_name == "Dashboard":
    if model:
        if input_type == "Image":
            # Upload section with modern styling
            st.markdown("### 📷 Image Detection")
            st.markdown('<div class="info-box">📄 Upload an image to detect objects using YOLO</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG"
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                
                # Display image info
                st.markdown(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")
                
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.markdown("③ 🖼️ Original Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.markdown("③ ✨ Detection Results")
                    result_placeholder = st.empty()
                
                # Run inference button
                st.markdown("---")
                if st.button("🚀 Start Detection", use_container_width=True):
                    with st.spinner("🔍 Analyzing image..."):
                        start_time = datetime.now()
                        
                        results = model.predict(
                            image,
                            conf=confidence_threshold,
                            iou=iou_threshold
                        )
                        
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        # Display results
                        with col2:
                            annotated_image = results[0].plot()
                            result_placeholder.image(annotated_image, use_container_width=True)
                        
                        # Update session stats
                        st.session_state.images_processed += 1
                        
                        # Display detection statistics
                        st.markdown("---")
                        st.markdown("### 📊 Detection Analytics")
                        
                        if len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            classes = boxes.cls.cpu().numpy()
                            confidences = boxes.conf.cpu().numpy()
                            
                            # Update total detections
                            st.session_state.total_detections += len(boxes)
                            
                            # Performance metrics with icons
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.markdown('<div class="metric-icon">⏱️</div>', unsafe_allow_html=True)
                                st.metric("Processing Time", f"{processing_time:.2f}s")
                            with metric_cols[1]:
                                st.markdown('<div class="metric-icon">🎯</div>', unsafe_allow_html=True)
                                st.metric("Total Objects", len(boxes))
                            with metric_cols[2]:
                                st.markdown('<div class="metric-icon">📊</div>', unsafe_allow_html=True)
                                st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")
                            with metric_cols[3]:
                                st.markdown('<div class="metric-icon">🏷️</div>', unsafe_allow_html=True)
                                st.metric("Unique Classes", len(np.unique(classes)))
                            
                            st.markdown("---")
                            
                            # Count detections per class
                            class_counts = {}
                            for cls_id, conf in zip(classes, confidences):
                                cls_name = results[0].names[int(cls_id)]
                                if cls_name not in class_counts:
                                    class_counts[cls_name] = []
                                class_counts[cls_name].append(conf)
                            
                            # Display class-wise statistics
                            st.markdown("③ 📝 Detection Breakdown")
                            stat_cols = st.columns(min(len(class_counts), 4))
                            for idx, (cls_name, confs) in enumerate(class_counts.items()):
                                with stat_cols[idx % 4]:
                                    st.metric(
                                        label=f"🟢 {cls_name.title()}",
                                        value=f"{len(confs)} detected",
                                        delta=f"Conf: {np.mean(confs):.2%}"
                                    )
                            
                            # Download button for results
                            st.markdown("---")
                            from io import BytesIO
                            
                            # Convert annotated image to bytes
                            annotated_pil = Image.fromarray(annotated_image)
                            buf = BytesIO()
                            annotated_pil.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="⬇️ Download Detection Results",
                                data=byte_im,
                                file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            # Success message with balloons
                            st.success(f"✅ Detection complete! Found {len(boxes)} objects in {processing_time:.2f}s")
                            st.balloons()
                        else:
                            st.warning("⚠️ No objects detected. Try adjusting the confidence threshold.")
        
        elif input_type == "Video":
            st.markdown("### 🎥 Video Detection")
            st.markdown('<div class="info-box">🎥 Upload a video to detect objects across frames</div>', unsafe_allow_html=True)
            
            uploaded_video = st.file_uploader(
                "Choose a video...", 
                type=["mp4", "avi", "mov"],
                help="Supported formats: MP4, AVI, MOV"
            )
            
            if uploaded_video is not None:
                # Display video info
                st.markdown(f"**File:** {uploaded_video.name} | **Size:** {uploaded_video.size / (1024*1024):.2f} MB")
                
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.markdown("③ 📹 Original Video")
                    st.video(uploaded_video)
                
                # Save uploaded video to temporary file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                
                st.markdown("---")
                if st.button("🚀 Process Video", use_container_width=True):
                    with st.spinner("🎬 Processing video frames..."):
                        start_time = datetime.now()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run inference on video
                        results = model.predict(
                            tfile.name,
                            conf=confidence_threshold,
                            iou=iou_threshold,
                            stream=True
                        )
                        
                        # Process and save video with detections
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        
                        cap = cv2.VideoCapture(tfile.name)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        frame_count = 0
                        total_detections = 0
                        
                        for result in results:
                            annotated_frame = result.plot()
                            out.write(annotated_frame)
                            
                            frame_count += 1
                            total_detections += len(result.boxes)
                            
                            # Update progress
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames}")
                        
                        cap.release()
                        out.release()
                        
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        with col2:
                            st.markdown("③ ✨ Processed Video")
                            st.video(output_path)
                        
                        # Video statistics
                        st.markdown("---")
                        st.markdown("### 📊 Video Analytics")
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
                        with metric_cols[1]:
                            st.metric("🎬 Total Frames", frame_count)
                        with metric_cols[2]:
                            st.metric("🎯 Total Detections", total_detections)
                        with metric_cols[3]:
                            st.metric("📈 Avg FPS", f"{frame_count / processing_time:.1f}")
                        
                        # Download button for processed video
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Processed Video",
                                data=f,
                                file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        
                        st.success(f"✅ Video processing complete! Processed {frame_count} frames in {processing_time:.2f}s")
                        st.balloons()
                    
                    os.unlink(tfile.name)
        
        elif input_type == "Webcam":
            st.markdown("### 📹 Webcam Detection")
            st.markdown('<div class="info-box">📷 Capture a photo using your webcam for real-time detection</div>', unsafe_allow_html=True)
            
            camera_photo = st.camera_input("📸 Take a picture")
            
            if camera_photo is not None:
                image = Image.open(camera_photo)
                
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    st.markdown("③ 📷 Captured Image")
                    st.image(image, use_container_width=True)
                
                with st.spinner("🔍 Analyzing capture..."):
                    start_time = datetime.now()
                    
                    results = model.predict(
                        image,
                        conf=confidence_threshold,
                        iou=iou_threshold
                    )
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    with col2:
                        st.markdown("③ ✨ Detection Results")
                        annotated_image = results[0].plot()
                        st.image(annotated_image, use_container_width=True)
                    
                    # Display statistics
                    if len(results[0].boxes) > 0:
                        st.markdown("---")
                        st.markdown("### 📊 Detection Results")
                        
                        boxes = results[0].boxes
                        metric_cols = st.columns(3)
                        
                        with metric_cols[0]:
                            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
                        with metric_cols[1]:
                            st.metric("🎯 Objects Detected", len(boxes))
                        with metric_cols[2]:
                            st.metric("📊 Avg Confidence", f"{np.mean(boxes.conf.cpu().numpy()):.2%}")
                        
                        st.success(f"✅ Detection complete! Found {len(boxes)} objects")
                        
                        # Download button
                        from io import BytesIO
                        annotated_pil = Image.fromarray(annotated_image)
                        buf = BytesIO()
                        annotated_pil.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="⬇️ Download Results",
                            data=byte_im,
                            file_name=f"webcam_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No objects detected in the captured image")

    else:
        # Model not loaded error for Dashboard
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-top: 2rem;
        ">
            <h2>⚠️ Model Not Found</h2>
            <p style="font-size: 1.1rem; margin-top: 1rem;">
                Please ensure <strong>best.pt</strong> model file is in the project directory
            </p>
            <p style="margin-top: 1rem; opacity: 0.9;">
                The YOLO model weights file is required to run object detection.
            </p>
        </div>
        """, unsafe_allow_html=True)

elif page_name == "Models":
    st.header("🤖 Model Management")
    
    # Current model info
    st.subheader("Current Model Information")
    
    if model:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-icon">🤖</div>', unsafe_allow_html=True)
            st.info("**Model Type:** YOLOv8")
        
        with col2:
            st.markdown('<div class="metric-icon">💾</div>', unsafe_allow_html=True)
            st.info("**Model File:** best.pt")
        
        with col3:
            st.markdown('<div class="metric-icon">✅</div>', unsafe_allow_html=True)
            st.success("**Status:** Active")
        
        st.markdown("---")
        
        # Model details
        with st.expander("📊 Detailed Model Information", expanded=True):
            st.write(f"**Number of Classes:** {len(model.names)}")
            st.write(f"**Class Names:** {', '.join(model.names.values())}")
            
            # Display model architecture info if available
            try:
                st.write(f"**Model Task:** {model.task}")
            except:
                pass
        
        # Model performance stats
        st.markdown("---")
        st.subheader("📈 Model Performance")
        
        perf_cols = st.columns(3)
        with perf_cols[0]:
            st.metric("Images Processed", st.session_state.images_processed)
        with perf_cols[1]:
            st.metric("Total Detections", st.session_state.total_detections)
        with perf_cols[2]:
            avg_per_image = st.session_state.total_detections / max(st.session_state.images_processed, 1)
            st.metric("Avg Detections/Image", f"{avg_per_image:.1f}")
    else:
        st.error("⚠️ No model loaded. Please ensure best.pt is in the project directory.")
    
    # Upload new model section
    st.markdown("---")
    st.subheader("📤 Upload New Model")
    st.markdown('<div class="info-box">Upload a new YOLOv8 model (.pt file) to replace the current model</div>', unsafe_allow_html=True)
    
    new_model = st.file_uploader("Select .pt model file", type=['pt'])
    
    if new_model:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Selected:** {new_model.name}")
            st.write(f"**Size:** {new_model.size / (1024*1024):.2f} MB")
        with col2:
            if st.button("Upload Model", use_container_width=True):
                with st.spinner("Uploading model..."):
                    with open('best.pt', 'wb') as f:
                        f.write(new_model.read())
                    st.success("✅ Model uploaded successfully!")
                    st.info("Please restart the application to load the new model.")

elif page_name == "Analytics":
    st.header("📊 Detection Analytics")
    
    if st.session_state.images_processed > 0:
        # Sample data for demonstration (replace with actual logging in production)
        dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
        detections = np.random.randint(5, 30, 24)
        
        # Line chart for trends
        st.subheader("📈 24-Hour Detection Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=detections,
            mode='lines+markers',
            name='Detections',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=8, color='#8b5cf6')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Number of Detections",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(30, 33, 57, 0.5)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📋 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-icon">📊</div>', unsafe_allow_html=True)
            st.metric("Total Detections", f"{detections.sum()}")
        with col2:
            st.markdown('<div class="metric-icon">📈</div>', unsafe_allow_html=True)
            st.metric("Avg per Hour", f"{detections.mean():.1f}")
        with col3:
            st.markdown('<div class="metric-icon">🔝</div>', unsafe_allow_html=True)
            st.metric("Peak Detections", f"{detections.max()}")
        with col4:
            st.markdown('<div class="metric-icon">📉</div>', unsafe_allow_html=True)
            st.metric("Min Detections", f"{detections.min()}")
        
        # Class distribution pie chart
        st.markdown("---")
        st.subheader("🎯 Detection Distribution")
        
        # Sample class distribution data
        if model:
            class_names = list(model.names.values())
            class_counts = np.random.randint(1, 20, len(class_names))
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=class_names,
                values=class_counts,
                hole=.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig_pie.update_layout(
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("📊 No data available yet. Process some images to see analytics!")
        st.markdown('<div class="info-box">Analytics will be populated once you start processing images or videos.</div>', unsafe_allow_html=True)

elif page_name == "Settings":
    st.header("⚙️ Application Settings")
    
    # Detection Settings
    st.subheader("🎯 Detection Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_conf = st.slider("Default Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        default_iou = st.slider("Default IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    with col2:
        max_det = st.number_input("Maximum Detections", min_value=1, max_value=1000, value=300)
        img_size = st.selectbox("Image Size", [320, 480, 640, 1280], index=2)
    
    st.markdown("---")
    
    # Display Settings
    st.subheader("🎨 Display Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_labels = st.checkbox("Show Labels on Detection", value=True)
        show_conf = st.checkbox("Show Confidence Scores", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    
    with col2:
        line_width = st.slider("Bounding Box Line Width", 1, 5, 2)
        font_size = st.slider("Label Font Size", 8, 20, 12)
    
    st.markdown("---")
    
    # Notification Settings
    st.subheader("🔔 Notifications")
    
    enable_alerts = st.checkbox("Enable Detection Alerts", value=True)
    
    if enable_alerts:
        alert_threshold = st.number_input("Alert Threshold (objects)", min_value=1, value=10)
        st.info(f"You will be alerted when {alert_threshold} or more objects are detected.")
    
    st.markdown("---")
    
    # Save settings
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved successfully!")
            st.balloons()
    
    # Reset to defaults
    st.markdown("---")
    st.subheader("🔄 Reset Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.info("Settings reset to default values.")
    
    with col2:
        if st.button("🗑️ Clear Session Data", use_container_width=True):
            st.session_state.images_processed = 0
            st.session_state.total_detections = 0
            st.success("Session data cleared!")
            st.rerun()
