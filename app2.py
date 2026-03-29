import streamlit as st
from PIL import Image
import numpy as np
import time
import cv2
import threading
import datetime
import os
import logging
import queue
from io import BytesIO
import tempfile

# Configure logging to terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartFlow")

# Page configuration
st.set_page_config(page_title="SmartFlow Traffic System", page_icon="🚦", layout="wide")

# Initialize session state variables
if 'traffic_states' not in st.session_state:
    st.session_state.traffic_states = {
        'lane1': {'light': 'red', 'timer': 60, 'vehicles': 0, 'file': None, 'file_type': None},
        'lane2': {'light': 'red', 'timer': 60, 'vehicles': 0, 'file': None, 'file_type': None},
        'lane3': {'light': 'red', 'timer': 60, 'vehicles': 0, 'file': None, 'file_type': None},
        'lane4': {'light': 'red', 'timer': 60, 'vehicles': 0, 'file': None, 'file_type': None}
    }

if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False

if 'thread_active' not in st.session_state:
    st.session_state.thread_active = False

if 'priority_lane' not in st.session_state:
    st.session_state.priority_lane = None

if 'current_phase' not in st.session_state:
    st.session_state.current_phase = "detection"  # detection, green, yellow, pause

if 'phase_time_left' not in st.session_state:
    st.session_state.phase_time_left = 0

if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

if 'update_needed' not in st.session_state:
    st.session_state.update_needed = False

if 'current_processing_lane' not in st.session_state:
    st.session_state.current_processing_lane = 1

if 'video_frames' not in st.session_state:
    st.session_state.video_frames = {
        'lane1': None,
        'lane2': None,
        'lane3': None,
        'lane4': None
    }

if 'temp_video_paths' not in st.session_state:
    st.session_state.temp_video_paths = {
        'lane1': None,
        'lane2': None,
        'lane3': None,
        'lane4': None
    }

# Constants
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # COCO: bicycle, car, motorcycle, bus, truck
GREEN_PHASE_DURATION = 10  # seconds
YELLOW_PHASE_DURATION = 5  # seconds
DETECTION_PHASE_DURATION = 5  # seconds per lane
PAUSE_PHASE_DURATION = 300  # 5 minutes (300 seconds)
RED_PHASE_DURATION = 60  # seconds

# Thread-safe communication queues
command_queue = queue.Queue()  # Commands to thread
result_queue = queue.Queue()   # Results from thread

# Load YOLOv8 model
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        logger.info("YOLOv8 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded video file to temporary location and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None

def detect_vehicles(image_array, lane_idx, model):
    """Run vehicle detection on a numpy image array"""
    try:
        # Run YOLO detection
        results = model(image_array, verbose=False)[0]
        
        # Count vehicles and create annotated image
        count = 0
        annotated_img = image_array.copy()
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                count += 1
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        logger.info(f"Lane {lane_idx+1}: Detected {count} vehicles")
        return count, annotated_img
        
    except Exception as e:
        logger.error(f"Error in Lane {lane_idx+1}: {str(e)}")
        return 0, None

def extract_video_frame(video_path):
    """Extract a frame from video for detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
            
        # Read a random frame from first half of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            frame_idx = np.random.randint(0, max(1, int(total_frames/2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        
        cap.release()
        return None
    except Exception as e:
        logger.error(f"Error extracting frame from video: {str(e)}")
        return None

def process_traffic_data():
    """Main processing function to update traffic signals and timers"""
    # Get the current state from session state
    traffic_states = st.session_state.traffic_states.copy()
    current_phase = st.session_state.current_phase
    phase_time_left = max(0, st.session_state.phase_time_left - 1)  # Decrement timer
    priority_lane = st.session_state.priority_lane
    
    # Update timers for all lanes
    for i in range(4):
        lane_key = f'lane{i+1}'
        if traffic_states[lane_key]['timer'] > 0:
            traffic_states[lane_key]['timer'] -= 1
    
    # Process different phases
    if current_phase == "green" and phase_time_left <= 0:
        # Green phase completed, transition to yellow
        logger.info(f"GREEN phase complete for Lane {priority_lane}")
        current_phase = "yellow"
        phase_time_left = YELLOW_PHASE_DURATION
        
        # Update the light of priority lane to yellow
        lane_key = f'lane{priority_lane}'
        traffic_states[lane_key]['light'] = 'yellow'
        traffic_states[lane_key]['timer'] = YELLOW_PHASE_DURATION
        logger.info(f"Starting YELLOW phase for Lane {priority_lane} ({YELLOW_PHASE_DURATION}s)")
        
    elif current_phase == "yellow" and phase_time_left <= 0:
        # Yellow phase completed, transition to detection
        logger.info(f"YELLOW phase complete for Lane {priority_lane}")
        current_phase = "detection"
        phase_time_left = DETECTION_PHASE_DURATION
        st.session_state.current_processing_lane = 1  # Start with lane 1
        logger.info(f"Starting DETECTION phase for Lane 1 ({DETECTION_PHASE_DURATION}s)")
        
    elif current_phase == "detection" and phase_time_left <= 0:
        # Current lane detection complete, move to next lane or pause
        current_lane = st.session_state.current_processing_lane
        
        if current_lane < 4:
            # Move to next lane
            current_lane += 1
            st.session_state.current_processing_lane = current_lane
            phase_time_left = DETECTION_PHASE_DURATION
            logger.info(f"Starting DETECTION phase for Lane {current_lane} ({DETECTION_PHASE_DURATION}s)")
        else:
            # All lanes processed, transition to pause phase
            current_phase = "pause"
            phase_time_left = PAUSE_PHASE_DURATION  # 5 minutes pause
            logger.info(f"All lanes processed. Starting PAUSE phase ({PAUSE_PHASE_DURATION}s)")
            
    elif current_phase == "pause" and phase_time_left <= 0:
        # Pause phase completed, transition back to detection
        logger.info("PAUSE phase complete")
        current_phase = "detection"
        phase_time_left = DETECTION_PHASE_DURATION
        st.session_state.current_processing_lane = 1  # Start with lane 1
        logger.info(f"Starting DETECTION phase for Lane 1 ({DETECTION_PHASE_DURATION}s)")
    
    # Update session state with new values
    st.session_state.traffic_states = traffic_states
    st.session_state.current_phase = current_phase
    st.session_state.phase_time_left = phase_time_left
    st.session_state.update_needed = True
    
    # Return if we need to run detection on current lane
    return (current_phase == "detection" and 
            phase_time_left <= 0 and 
            st.session_state.current_processing_lane <= 4)

def run_detection_for_current_lane(model):
    """Run detection for the current lane being processed"""
    current_lane = st.session_state.current_processing_lane
    lane_key = f'lane{current_lane}'
    file = st.session_state.traffic_states[lane_key]['file']
    file_type = st.session_state.traffic_states[lane_key]['file_type']
    
    if file:
        try:
            img_array = None
            
            # Process based on file type
            if file_type == "image":
                # Reset file pointer for images
                file.seek(0)
                # Load image
                image = Image.open(file).convert("RGB")
                img_array = np.array(image)
            elif file_type == "video":
                # Use the stored video path
                video_path = st.session_state.temp_video_paths[lane_key]
                if video_path:
                    img_array = extract_video_frame(video_path)
            
            if img_array is not None:
                # Run detection
                count, annotated_img = detect_vehicles(img_array, current_lane-1, model)
                
                # Update session state
                st.session_state.traffic_states[lane_key]['vehicles'] = count
                st.session_state[f"annotated_image_{current_lane-1}"] = annotated_img
                logger.info(f"Lane {current_lane}: Detection complete - {count} vehicles")
                return True
                
        except Exception as e:
            logger.error(f"Error processing Lane {current_lane}: {str(e)}")
    
    return False

def detect_vehicles_for_all_lanes(model):
    """Detect vehicles for all lanes and update priority list."""
    for lane_idx in range(4):
        lane_key = f'lane{lane_idx+1}'
        file = st.session_state.traffic_states[lane_key]['file']
        file_type = st.session_state.traffic_states[lane_key]['file_type']
        img_array = None

        if file:
            try:
                if file_type == "image":
                    file.seek(0)
                    image = Image.open(file).convert("RGB")
                    img_array = np.array(image)
                elif file_type == "video":
                    video_path = st.session_state.temp_video_paths[lane_key]
                    if video_path:
                        img_array = extract_video_frame(video_path)
                if img_array is not None:
                    count, annotated_img = detect_vehicles(img_array, lane_idx, model)
                    st.session_state.traffic_states[lane_key]['vehicles'] = count
                    st.session_state[f"annotated_image_{lane_idx}"] = annotated_img
                    logger.info(f"Lane {lane_idx+1}: Detection complete - {count} vehicles")
            except Exception as e:
                logger.error(f"Error processing Lane {lane_idx+1}: {str(e)}")
        else:
            st.session_state.traffic_states[lane_key]['vehicles'] = 0
            st.session_state[f"annotated_image_{lane_idx}"] = None

    # After all detections, update the priority list
    update_priority_list()

def update_priority_list():
    """Update priority list based on vehicle counts and set green light for top lane"""
    # Get all lanes and their vehicle counts
    lane_with_counts = [(i+1, st.session_state.traffic_states[f'lane{i+1}']['vehicles']) 
                       for i in range(4)]
    
    # Sort lanes by vehicle count (highest first)
    lane_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Reset all lights to red
    for i in range(4):
        lane_key = f'lane{i+1}'
        st.session_state.traffic_states[lane_key]['light'] = 'red'
        st.session_state.traffic_states[lane_key]['timer'] = RED_PHASE_DURATION
    
    # Set priority lane to green (highest count)
    if lane_with_counts:
        highest_priority_lane = lane_with_counts[0][0]
        lane_key = f'lane{highest_priority_lane}'
        st.session_state.traffic_states[lane_key]['light'] = 'green'
        st.session_state.traffic_states[lane_key]['timer'] = GREEN_PHASE_DURATION
        
        # Update phase information
        st.session_state.current_phase = "green"
        st.session_state.phase_time_left = GREEN_PHASE_DURATION
        st.session_state.priority_lane = highest_priority_lane
        
        logger.info(f"Priority updated. Top Lane: {highest_priority_lane} with {lane_with_counts[0][1]} vehicles")
        logger.info(f"Starting GREEN phase for Lane {highest_priority_lane}")
    
    st.session_state.update_needed = True

def traffic_control_worker(model):
    """Background worker function that runs in a separate thread"""
    logger.info("Traffic control worker thread started")
    
    while True:
        # Check if we should exit
        try:
            if not command_queue.empty():
                command = command_queue.get_nowait()
                if command == "STOP":
                    logger.info("Received STOP command, exiting thread")
                    break
        except queue.Empty:
            pass
        
        # Process traffic data
        should_run_detection = process_traffic_data()
        
        # Run detection for current lane if needed
        if should_run_detection:
            detection_success = run_detection_for_current_lane(model)
            
            # If we just finished lane 4, update priority list
            if st.session_state.current_processing_lane == 4 and detection_success:
                detect_vehicles_for_all_lanes(model)
        
        # Sleep for 1 second
        time.sleep(1)
    
    logger.info("Traffic control worker thread stopped")

def update_ui():
    """Update the UI with the latest traffic information"""
    # Update image containers and vehicle metrics
    for i in range(4):
        lane_num = i+1
        lane_key = f'lane{i+1}'
        
        # Update annotated images if available
        if f"annotated_image_{i}" in st.session_state and st.session_state[f"annotated_image_{i}"] is not None:
            image_containers[i].image(
                st.session_state[f"annotated_image_{i}"],
                caption=f"Lane {lane_num}: {st.session_state.traffic_states[lane_key]['vehicles']} vehicles detected",
                use_container_width=True
            )
            vehicle_metrics[i].metric(
                f"Vehicles in Lane {lane_num}",
                st.session_state.traffic_states[lane_key]['vehicles']
            )
        
        # Update traffic light indicators
        light_state = st.session_state.traffic_states[lane_key]['light']
        timer_value = st.session_state.traffic_states[lane_key]['timer']
        
        red_indicator, yellow_indicator, green_indicator, timer_display = traffic_containers[i]
        red_indicator.markdown("🔴" if light_state == 'red' else "⚫")
        yellow_indicator.markdown("🟡" if light_state == 'yellow' else "⚫")
        green_indicator.markdown("🟢" if light_state == 'green' else "⚫")
        timer_display.markdown(f"**Timer:** {timer_value}s")
    
    # Update priority list display
    lane_with_counts = [(i+1, st.session_state.traffic_states[f'lane{i+1}']['vehicles']) 
                       for i in range(4)]
    lane_with_counts.sort(key=lambda x: x[1], reverse=True)
    
    priority_text = """
    <div style=" padding: 10px; border-radius: 5px;">
        <h4>Current Priority Order</h4>
        <table style="width:100%">
            <tr><th>Priority</th><th>Lane</th><th>Vehicle Count</th><th>Status</th></tr>
    """
    
    for i, (lane, count) in enumerate(lane_with_counts, 1):
        lane_key = f'lane{lane}'
        light_state = st.session_state.traffic_states[lane_key]['light']
        
        # Determine status display
        if light_state == 'green':
            status = '<span style="color: green; font-weight: bold;">GREEN</span>'
        elif light_state == 'yellow':
            status = '<span style="color: #FFD700; font-weight: bold;">YELLOW</span>'
        else:
            status = '<span style="color: red; font-weight: bold;">RED</span>'
            
        # Highlight the current priority lane
        highlight = "background-color: lightgreen; color:black;" if lane == st.session_state.priority_lane else ""
        priority_text += f'<tr style="{highlight}"><td>{i}</td><td>Lane {lane}</td><td>{count} vehicles</td><td>{status}</td></tr>'
    
    priority_text += """
        </table>
    </div>
    """
    priority_list.markdown(priority_text, unsafe_allow_html=True)
    
    # Update phase information
    phase_info = ""
    if st.session_state.current_phase == "detection":
        current_lane = st.session_state.current_processing_lane
        phase_info = f"<h4>Current Phase: <span style='color: blue;'>DETECTION</span> for Lane {current_lane} (Time Left: {st.session_state.phase_time_left}s)</h4>"
    elif st.session_state.current_phase == "green":
        phase_info = f"<h4>Current Phase: <span style='color: green;'>GREEN</span> for Lane {st.session_state.priority_lane} (Time Left: {st.session_state.phase_time_left}s)</h4>"
    elif st.session_state.current_phase == "yellow":
        phase_info = f"<h4>Current Phase: <span style='color: #FFD700;'>YELLOW</span> for Lane {st.session_state.priority_lane} (Time Left: {st.session_state.phase_time_left}s)</h4>"
    elif st.session_state.current_phase == "pause":
        phase_info = f"<h4>Current Phase: <span style='color: purple;'>PAUSE</span> (Time Left: {st.session_state.phase_time_left}s)</h4>"
    
    phase_display.markdown(phase_info, unsafe_allow_html=True)

def start_traffic_system(model):
    """Start the traffic control system"""
    if model is None:
        logger.error("Cannot start system, model failed to load")
        return False
    
    logger.info("Starting traffic control system")
    
    # Set initial state
    st.session_state.detection_running = True
    st.session_state.thread_active = True
    st.session_state.current_phase = "detection"
    st.session_state.phase_time_left = DETECTION_PHASE_DURATION
    st.session_state.current_processing_lane = 1
    
    # Start the worker thread
    worker = threading.Thread(target=traffic_control_worker, args=(model,))
    worker.daemon = True
    worker.start()
    
    logger.info("Traffic control system started")
    return True

def stop_traffic_system():
    """Stop the traffic control system"""
    logger.info("Stopping traffic control system")
    
    # Send stop command to worker thread
    command_queue.put("STOP")
    
    # Reset state
    st.session_state.detection_running = False
    st.session_state.thread_active = False
    
    # Reset all lights to red
    for i in range(4):
        lane_key = f'lane{i+1}'
        st.session_state.traffic_states[lane_key]['light'] = 'red'
        st.session_state.traffic_states[lane_key]['timer'] = RED_PHASE_DURATION
    
    # Reset phase information
    st.session_state.current_phase = "detection"
    st.session_state.phase_time_left = 0
    
    logger.info("Traffic control system stopped")

# Clean up temporary files when the app exits
def cleanup_temp_files():
    """Remove temporary video files when app exits"""
    for lane_key, path in st.session_state.temp_video_paths.items():
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.info(f"Removed temporary file: {path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {path}: {str(e)}")

# Register cleanup handler
import atexit
atexit.register(cleanup_temp_files)

# Now begin the UI construction
st.title("🚦 SmartFlow Traffic Management System")

# Display the current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Current Date and Time: {current_time}")
st.caption("Current User's Login: dhairyagothi")

# Phase information display
phase_display = st.empty()

# Load model at the start
model = load_model()

# Main control button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.session_state.detection_running:
        start_button = st.button("🛑 Stop Traffic System", type="secondary", use_container_width=True)
    else:
        start_button = st.button("🚦 Start Traffic System", type="primary", use_container_width=True)

# Create a container for videos and controls
image_containers = []
vehicle_metrics = []

with st.container():
    # Create 2x2 grid for videos
    video_cols = st.columns(2)
    
    # First row of videos
    for i in range(2):
        lane_num = i+1
        lane_key = f'lane{i+1}'
        with video_cols[i]:
            st.markdown(f"### Lane {lane_num}")
            
            # Add file format selection
            file_type = st.radio(f"File Type for Lane {lane_num}", 
                               ["Image", "Video"],
                               horizontal=True,
                               key=f"file_type_{lane_num}")
            
            # Based on file type, show appropriate uploader
            if file_type == "Image":
                uploaded_file = st.file_uploader(f"Upload image for Lane {lane_num}", 
                                            type=["jpg", "png", "jpeg"], 
                                            key=f"file_{lane_num}")
                
                if uploaded_file:
                    # Store the file in session state
                    st.session_state.traffic_states[lane_key]['file'] = uploaded_file
                    st.session_state.traffic_states[lane_key]['file_type'] = "image"
                    
                    # Display original image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    # Create a container for the image that can be updated
                    image_container = st.empty()
                    image_container.image(image, caption=f"Lane {lane_num} Image", use_column_width=True)
                    image_containers.append(image_container)
                    
                    # Create a container for the vehicle metric
                    vehicle_metric = st.empty()
                    vehicle_metrics.append(vehicle_metric)
                    vehicle_metric.metric(f"Vehicles in Lane {lane_num}", 0)
                    
                    logger.info(f"Uploaded image for Lane {lane_num}")
                else:
                    # Create empty containers
                    image_container = st.empty()
                    image_containers.append(image_container)
                    
                    vehicle_metric = st.empty()
                    vehicle_metrics.append(vehicle_metric)
                    
            else:  # Video
                uploaded_file = st.file_uploader(f"Upload video for Lane {lane_num}", 
                                            type=["mp4", "avi", "mov", "mkv"], 
                                            key=f"video_{lane_num}")
                
                if uploaded_file:
                    # Save video to temp file and store path
                    video_path = save_uploaded_file(uploaded_file)
                    if video_path:
                        st.session_state.traffic_states[lane_key]['file'] = uploaded_file
                        st.session_state.traffic_states[lane_key]['file_type'] = "video"
                        st.session_state.temp_video_paths[lane_key] = video_path
                        
                        # Display video player
                        st.video(video_path)
                        
                        # Extract a frame for preview
                        preview_frame = extract_video_frame(video_path)
                        
                        # Create a container for the detection results
                        image_container = st.empty()
                        if preview_frame is not None:
                            image_container.image(preview_frame, caption=f"Lane {lane_num} Preview Frame", use_column_width=True)
                        image_containers.append(image_container)
                        
                        # Create a container for the vehicle metric
                        vehicle_metric = st.empty()
                        vehicle_metrics.append(vehicle_metric)
                        vehicle_metric.metric(f"Vehicles in Lane {lane_num}", 0)
                        
                        logger.info(f"Uploaded video for Lane {lane_num}")
                    else:
                        st.error(f"Failed to process video for Lane {lane_num}")
                        # Create empty containers
                        image_container = st.empty()
                        image_containers.append(image_container)
                        
                        vehicle_metric = st.empty()
                        vehicle_metrics.append(vehicle_metric)
                else:
                    # Create empty containers
                    image_container = st.empty()
                    image_containers.append(image_container)
                    
                    vehicle_metric = st.empty()
                    vehicle_metrics.append(vehicle_metric)
    
    # Second row of videos
    video_cols2 = st.columns(2)
    for i in range(2):
        lane_num = i+3
        lane_key = f'lane{lane_num}'
        with video_cols2[i]:
            st.markdown(f"### Lane {lane_num}")
            
            # Add file format selection
            file_type = st.radio(f"File Type for Lane {lane_num}", 
                               ["Image", "Video"],
                               horizontal=True,
                               key=f"file_type_{lane_num}")
            
            # Based on file type, show appropriate uploader
            if file_type == "Image":
                uploaded_file = st.file_uploader(f"Upload image for Lane {lane_num}", 
                                            type=["jpg", "png", "jpeg"], 
                                            key=f"file_{lane_num}")
                
                if uploaded_file:
                    # Store the file in session state
                    st.session_state.traffic_states[lane_key]['file'] = uploaded_file
                    st.session_state.traffic_states[lane_key]['file_type'] = "image"
                    
                    # Display original image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    # Create a container for the image that can be updated
                    image_container = st.empty()
                    image_container.image(image, caption=f"Lane {lane_num} Image", use_column_width=True)
                    image_containers.append(image_container)
                    
                    # Create a container for the vehicle metric
                    vehicle_metric = st.empty()
                    vehicle_metrics.append(vehicle_metric)
                    vehicle_metric.metric(f"Vehicles in Lane {lane_num}", 0)
                    
                    logger.info(f"Uploaded image for Lane {lane_num}")
                else:
                    # Create empty containers
                    image_container = st.empty()
                    image_containers.append(image_container)
                    
                    vehicle_metric = st.empty()
                    vehicle_metrics.append(vehicle_metric)
                    
            else:  # Video
                uploaded_file = st.file_uploader(f"Upload video for Lane {lane_num}", 
                                            type=["mp4", "avi", "mov", "mkv"], 
                                            key=f"video_{lane_num}")
                
                if uploaded_file:
                    # Save video to temp file and store path
                    video_path = save_uploaded_file(uploaded_file)
                    if video_path:
                        st.session_state.traffic_states[lane_key]['file'] = uploaded_file
                        st.session_state.traffic_states[lane_key]['file_type'] = "video"
                        st.session_state.temp_video_paths[lane_key] = video_path
                        
                        # Display video player
                        st.video(video_path)
                        
                        # Extract a frame for preview
                        preview_frame = extract_video_frame(video_path)
                        
                        # Create a container for the detection results
                        image_container = st.empty()
                        if preview_frame is not None:
                            image_container.image(preview_frame, caption=f"Lane {lane_num} Preview Frame", use_column_width=True)
                        image_containers.append(image_container)
                        
                        # Create a container for the vehicle metric
                        vehicle_metric = st.empty()
                        vehicle_metrics.append(vehicle_metric)
                        vehicle_metric.metric(f"Vehicles in Lane {lane_num}", 0)
                        
                        logger.info(f"Uploaded video for Lane {lane_num}")
                    else:
                        st.error(f"Failed to process video for Lane {lane_num}")
                        # Create empty containers
                        image_container = st.empty()
                        image_containers.append(image_container)
                        
                        vehicle_metric = st.empty()
                        vehicle_metrics.append(vehicle_metric)
                else:
                    # Create empty containers
                    image_container = st.empty()
                    image_containers.append(image_container)
                    
                    vehicle_metric = st.empty()
                    vehicle_metrics.append(vehicle_metric)

# Traffic Control Panel
st.markdown("---")
st.subheader("Traffic Control Panel")

# Create traffic light containers - fix duplicated display issue
traffic_containers = []
control_cols = st.columns(4)
for i in range(4):
    lane_num = i+1
    lane_key = f'lane{i+1}'
    with control_cols[i]:
        st.markdown(f"### Lane {lane_num} Status")
        
        # Create placeholders for traffic light indicators (single display)
        red_indicator = st.empty()
        yellow_indicator = st.empty()
        green_indicator = st.empty()
        
        # Get current state
        light_state = st.session_state.traffic_states[lane_key]['light']
        
        # Set initial state
        red_indicator.markdown("🔴" if light_state == 'red' else "⚫")
        yellow_indicator.markdown("🟡" if light_state == 'yellow' else "⚫")
        green_indicator.markdown("🟢" if light_state == 'green' else "⚫")
        
        # Timer container
        timer_container = st.empty()
        timer_container.markdown(f"**Timer:** {st.session_state.traffic_states[lane_key]['timer']}s")
        
        # Store containers for updates
        traffic_containers.append((red_indicator, yellow_indicator, green_indicator, timer_container))

# Priority list display
st.markdown("---")
st.subheader("Lane Priority List")
priority_list = st.empty()

# System status display


# Handle button clicks
if start_button:
    if not st.session_state.detection_running:
        success = start_traffic_system(model)
        if success:
            detect_vehicles_for_all_lanes(model)
            st.experimental_rerun()  # Force UI update
    else:
        stop_traffic_system()
        st.experimental_rerun()  # Force UI update

# Initialize logging
logger.info("SmartFlow Traffic Management System initialized")
logger.info("Upload images/videos for each lane and press Start to begin detection")

# Periodic UI updates via polling loop
if st.session_state.detection_running and time.time() - st.session_state.last_update > 0.5:
    update_ui()
    st.session_state.last_update = time.time()
    st.session_state.update_needed = False