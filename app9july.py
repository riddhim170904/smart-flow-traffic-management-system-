import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import logging
import tempfile
import serial
import serial.tools.list_ports
import os
import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartTraffic")

st.set_page_config(page_title="Smart Traffic System", page_icon="🚦", layout="wide")

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**Current Date and Time:** {current_time}")
st.markdown("**Current User's Login:** dhairyagothi")

LANE_COUNT = 4
GREEN_TIME = 15
YELLOW_TIME = 5
RED_TIME = 75

if "traffic_states" not in st.session_state:
    st.session_state.traffic_states = {
        f'lane{i+1}': {'light': 'red', 'timer': RED_TIME}
        for i in range(LANE_COUNT)
    }

if "lane_timers" not in st.session_state:
    st.session_state.lane_timers = [RED_TIME for _ in range(LANE_COUNT)]
    logger.info(f"Initialized lane timers: {st.session_state.lane_timers}")

if "last_timer_update" not in st.session_state:
    st.session_state.last_timer_update = time.time()
    logger.info("Initialized last_timer_update.")

if "last_priority_lane" not in st.session_state:
    st.session_state.last_priority_lane = None

if "priority_order" not in st.session_state:
    st.session_state.priority_order = list(range(1, LANE_COUNT+1))  # [1,2,3,4] initially

if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = {
        f'lane{i+1}': {'vehicles': 0, 'file': None, 'file_type': None}
        for i in range(LANE_COUNT)
    }

if 'priority_lane' not in st.session_state:
    st.session_state.priority_lane = None

if 'temp_video_paths' not in st.session_state:
    st.session_state.temp_video_paths = {
        f'lane{i+1}': None for i in range(LANE_COUNT)
    }

if 'green_timer_active' not in st.session_state:
    st.session_state.green_timer_active = False

if 'cycle_lanes_used' not in st.session_state:
    st.session_state.cycle_lanes_used = set()

VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None

def detect_vehicles(image_array, model):
    try:
        results = model(image_array, verbose=False)[0]
        count = 0
        annotated_img = image_array.copy()
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return count, annotated_img
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return 0, image_array

def extract_video_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            frame_idx = np.random.randint(0, max(1, int(total_frames/2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        cap.release()
        return None
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        return None

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description:
            return port.device
    return None

def initialize_arduino():
    try:
        port = find_arduino_port()
        if port:
            arduino = serial.Serial(port, 9600, timeout=1)
            time.sleep(3)  # Give Arduino time to reset
            # Send test command to verify connection
            arduino.write(b"TEST\n")
            arduino.flush()
            logger.info(f"Arduino connected on port {port}")
            return arduino
        else:
            logger.error("No Arduino found")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Arduino: {str(e)}")
        return None

def send_green_signal(arduino, lane):
    try:
        if arduino and arduino.is_open:
            # Send GREEN signal to turn on LED on pin 9
            command = "GREEN\n"
            arduino.write(command.encode())
            arduino.flush()  # Ensure data is sent immediately
            logger.info(f"🟢 GREEN signal sent to Arduino pin 9 for Lane {lane} - LED will stay ON for {GREEN_TIME} seconds")
            
            # Always start/restart timer for this lane
            st.session_state.green_timer_active = True
            timer_thread = threading.Thread(target=turn_off_green_after_delay, args=(arduino, GREEN_TIME, lane))
            timer_thread.daemon = True
            timer_thread.start()
            return True
        else:
            logger.error(f"Arduino not connected - Cannot send GREEN signal for Lane {lane}")
    except Exception as e:
        logger.error(f"Error sending GREEN command to Arduino: {str(e)}")
    return False

def turn_off_green_after_delay(arduino, delay_seconds, lane):
    try:
        time.sleep(delay_seconds)
        if arduino and arduino.is_open:
            command = "RED\n"
            arduino.write(command.encode())
            arduino.flush()
            logger.info(f"🔴 GREEN signal turned OFF for Lane {lane} after {delay_seconds} seconds - LED turned OFF")
        st.session_state.green_timer_active = False
    except Exception as e:
        logger.error(f"Error in timer thread for Lane {lane}: {str(e)}")
        st.session_state.green_timer_active = False

def send_light_signal(arduino, lane, color):
    """
    Send a command to Arduino to set the light color for a specific lane.
    lane: 1-4
    color: 'red', 'yellow', 'green'
    """
    try:
        if arduino and arduino.is_open:
            command = f"L{lane}_{color.upper()}\n"
            arduino.write(command.encode())
            arduino.flush()
            logger.info(f"Signal sent: {command.strip()} (Lane {lane}, {color})")
            return True
        else:
            logger.error(f"Arduino not connected - Cannot send {color.upper()} signal for Lane {lane}")
    except Exception as e:
        logger.error(f"Error sending {color.upper()} command to Arduino: {str(e)}")
    return False

def analyze_traffic_and_update_priority():
    model = load_model()
    if not model:
        st.error("Failed to load detection model. Please check your installation.")
        return
    for i in range(LANE_COUNT):
        lane_key = f'lane{i+1}'
        lane_data = st.session_state.traffic_data[lane_key]
        file = lane_data['file']
        file_type = lane_data['file_type']
        if file:
            try:
                img_array = None
                if file_type == "image":
                    file.seek(0)
                    image = Image.open(file).convert("RGB")
                    img_array = np.array(image)
                elif file_type == "video":
                    video_path = st.session_state.temp_video_paths[lane_key]
                    if video_path:
                        img_array = extract_video_frame(video_path)
                if img_array is not None:
                    count, annotated_img = detect_vehicles(img_array, model)
                    st.session_state.traffic_data[lane_key]['vehicles'] = count
                    st.session_state[f"annotated_image_{i}"] = annotated_img
                    logger.info(f"Lane {i+1}: Detected {count} vehicles")
            except Exception as e:
                logger.error(f"Error processing Lane {i+1}: {str(e)}")
    # Update lane priorities after detection
    update_lane_priority(force_switch=True)

def cleanup_temp_files():
    for lane_key, path in st.session_state.temp_video_paths.items():
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.info(f"Removed temporary file: {path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {path}: {str(e)}")

import atexit
atexit.register(cleanup_temp_files)

if 'arduino' not in st.session_state:
    st.session_state.arduino = initialize_arduino()

st.title("🚦 Smart Traffic Management System")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Traffic & Update Signal", type="primary", use_container_width=True)

with st.container():
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)
    for i, cols in enumerate([row1_cols, row2_cols]):
        for j, col in enumerate(cols):
            lane_idx = i*2 + j
            lane_num = lane_idx + 1
            lane_key = f'lane{lane_num}'
            with col:
                st.markdown(f"### Lane {lane_num}")
                file_type = st.radio(f"File Type", ["Image", "Video"], horizontal=True, key=f"file_type_{lane_num}")
                if file_type == "Image":
                    uploaded_file = st.file_uploader(f"Upload image for Lane {lane_num}", 
                                                type=["jpg", "png", "jpeg"], 
                                                key=f"file_{lane_num}")
                    if uploaded_file:
                        st.session_state.traffic_data[lane_key]['file'] = uploaded_file
                        st.session_state.traffic_data[lane_key]['file_type'] = "image"
                        image = Image.open(uploaded_file).convert("RGB")
                        st.image(image, caption=f"Lane {lane_num} Image", use_container_width=True)
                else:
                    uploaded_file = st.file_uploader(f"Upload video for Lane {lane_num}", 
                                                type=["mp4", "avi", "mov", "mkv"], 
                                                key=f"video_{lane_num}")
                    if uploaded_file:
                        video_path = save_uploaded_file(uploaded_file)
                        if video_path:
                            st.session_state.traffic_data[lane_key]['file'] = uploaded_file
                            st.session_state.traffic_data[lane_key]['file_type'] = "video"
                            st.session_state.temp_video_paths[lane_key] = video_path
                            st.video(video_path)
                            preview_frame = extract_video_frame(video_path)
                            if preview_frame is not None:
                                st.image(
                                    preview_frame,
                                    caption=f"Preview Frame from Lane {lane_num} Video",
                                    use_container_width=True
                                )
                        else:
                            st.error(f"Failed to process video for Lane {lane_num}")

def reset_lane_timers(next_green_idx, prev_green_idx=None, reset_cycle=False):
    for i in range(LANE_COUNT):
        lane_num = i + 1
        if i == next_green_idx:
            st.session_state.lane_timers[i] = GREEN_TIME
            st.session_state.traffic_states[f'lane{lane_num}']['light'] = 'green'
            st.session_state.traffic_states[f'lane{lane_num}']['timer'] = GREEN_TIME
            send_light_signal(st.session_state.arduino, lane_num, 'green')
        elif prev_green_idx is not None and i == prev_green_idx:
            st.session_state.lane_timers[i] = RED_TIME
            st.session_state.traffic_states[f'lane{lane_num}']['light'] = 'red'
            st.session_state.traffic_states[f'lane{lane_num}']['timer'] = RED_TIME
            send_light_signal(st.session_state.arduino, lane_num, 'red')
        else:
            # Set to red if not green or previous green
            st.session_state.traffic_states[f'lane{lane_num}']['light'] = 'red'
            send_light_signal(st.session_state.arduino, lane_num, 'red')
    logger.info(f"Timers updated: Lane {next_green_idx+1} is GREEN ({GREEN_TIME}s), Lane {prev_green_idx+1 if prev_green_idx is not None else '-'} is set to RED ({RED_TIME}s)")
    if reset_cycle:
        st.session_state.cycle_lanes_used = set([next_green_idx])
    else:
        st.session_state.cycle_lanes_used.add(next_green_idx)

# -- MAIN PRIORITY UPDATE FUNCTION --
def update_lane_priority(force_switch=False):
    lane_vehicle_counts = [(i+1, st.session_state.traffic_data[f'lane{i+1}']['vehicles']) for i in range(LANE_COUNT)]
    sorted_lanes = sorted(lane_vehicle_counts, key=lambda x: x[1], reverse=True)
    st.session_state.priority_order = [lane for lane, _ in sorted_lanes]
    if not sorted_lanes:
        return
    current_priority_lane = st.session_state.priority_lane
    # Find lanes not used in this cycle
    unused_lanes = [lane for lane, _ in sorted_lanes if (lane-1) not in st.session_state.cycle_lanes_used]
    if not unused_lanes:
        # All lanes have had green, start new cycle and re-evaluate priorities
        next_priority_lane = sorted_lanes[0][0]
        prev_green_idx = current_priority_lane - 1 if current_priority_lane is not None else None
        reset_lane_timers(next_priority_lane-1, prev_green_idx, reset_cycle=True)
        st.session_state.priority_lane = next_priority_lane
        logger.info(f"🚦 New cycle: Switched priority to Lane {next_priority_lane} (vehicles={sorted_lanes[0][1]})")
        send_green_signal(st.session_state.arduino, next_priority_lane)
    else:
        # Pick highest-priority unused lane
        next_priority_lane = unused_lanes[0]
        prev_green_idx = current_priority_lane - 1 if current_priority_lane is not None else None
        reset_lane_timers(next_priority_lane-1, prev_green_idx)
        st.session_state.priority_lane = next_priority_lane
        logger.info(f"🚦 Cycle: Switched priority to Lane {next_priority_lane} (vehicles={dict(sorted_lanes)[next_priority_lane]})")
        send_green_signal(st.session_state.arduino, next_priority_lane)

# -- TIMER LOGIC AND AUTO-SWITCH --
def auto_switch_lane():
    now = time.time()
    elapsed = now - st.session_state.last_timer_update
    if elapsed >= 1.0:  # Update every 1 second for integer countdown
        # decrement timers
        for i in range(LANE_COUNT):
            if st.session_state.lane_timers[i] > 0:
                st.session_state.lane_timers[i] -= 1  # Decrease by 1 second
                st.session_state.traffic_states[f'lane{i+1}']['timer'] = st.session_state.lane_timers[i]
                # Only log every 5 seconds to reduce log spam
                if st.session_state.lane_timers[i] % 5 == 0:
                    logger.info(f"Lane {i+1} countdown: {int(st.session_state.lane_timers[i])} sec remaining")
        st.session_state.last_timer_update = now

        # Check for yellow phase transition (2 seconds before green ends)
        if st.session_state.priority_lane is not None:
            cur_idx = st.session_state.priority_lane - 1
            lane_num = st.session_state.priority_lane
            if st.session_state.lane_timers[cur_idx] == YELLOW_TIME:
                # Start yellow phase
                st.session_state.traffic_states[f'lane{lane_num}']['light'] = 'yellow'
                send_light_signal(st.session_state.arduino, lane_num, 'yellow')
                logger.info(f"Lane {lane_num} entering YELLOW phase ({YELLOW_TIME}s)")
            elif st.session_state.lane_timers[cur_idx] <= 0:
                # End of yellow, switch to next lane
                logger.info(f"Lane {lane_num} timer expired, switching lane.")
                update_lane_priority(force_switch=True)

# -- Call auto-switch each rerun --
if st.session_state.priority_lane is not None:
    auto_switch_lane()

# --- Detection results area with countdown timers ---
st.markdown("---")
st.subheader("Traffic Analysis Results")

st.markdown("""
<style>
.detection-card {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.vehicle-count {
    font-size: 28px;
    font-weight: bold;
    color: #1E88E5;
}
.green-light {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
    border: 3px solid #2E7D32;
    color: white;
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.4);
}
.yellow-light {
    background: linear-gradient(135deg, #FFC107, #FFD54F);
    border: 3px solid #F57F17;
    color: #333;
    box-shadow: 0 0 20px rgba(255, 193, 7, 0.4);
}
.red-light {
    background: linear-gradient(135deg, #F44336, #EF5350);
    border: 3px solid #C62828;
    color: white;
    box-shadow: 0 0 20px rgba(244, 67, 54, 0.4);
}
.timer-display {
    font-size: 24px;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}
.lane-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

detection_cols = st.columns(4)
for i in range(4):
    lane_num = i + 1
    lane_key = f'lane{i+1}'
    with detection_cols[i]:
        # Get traffic light status
        light_state = st.session_state.traffic_states[lane_key]['light']
        
        # Determine card styling based on light state
        if light_state == 'green':
            card_class = "detection-card green-light"
            status_text = "🟢 GREEN"
            phase_text = "VEHICLES PASSING"
        elif light_state == 'yellow':
            card_class = "detection-card yellow-light"
            status_text = "🟡 YELLOW"
            phase_text = "PREPARE TO STOP"
        else:
            card_class = "detection-card red-light"
            status_text = "🔴 RED"
            phase_text = "VEHICLES WAITING"
            
        timer_value = int(st.session_state.lane_timers[i])
        is_priority = st.session_state.priority_lane == lane_num
        
        st.markdown(f"""
        <div class="{card_class}">
            <div class="lane-title"> Lane {lane_num} {'  PRIORITY' if is_priority else ''}</div>
            <p><strong>Vehicles:</strong> <span class="vehicle-count">{st.session_state.traffic_data[lane_key]['vehicles']}</span></p>
            <p><strong>Status:</strong> {status_text}</p>
            <p><strong>Phase:</strong> {phase_text}</p>
            <p><strong>Time:</strong> <span class="timer-display">{timer_value} sec</span></p>
        </div>
        """, unsafe_allow_html=True)
        if f"annotated_image_{i}" in st.session_state and st.session_state[f"annotated_image_{i}"] is not None:
            st.image(
                st.session_state[f"annotated_image_{i}"],
                caption=f"Lane {lane_num}: {st.session_state.traffic_data[lane_key]['vehicles']} vehicles detected",
                use_container_width=True
            )

# --- Visualization ---
st.markdown("---")
st.subheader("Lane Priority Visualization")
if st.session_state.priority_lane:
    lane_priorities = [(lane, st.session_state.traffic_data[f'lane{lane}']['vehicles']) 
        for lane in st.session_state.priority_order]
    labels = [f"Lane {lane}" for lane, _ in lane_priorities]
    values = [count for _, count in lane_priorities]
    priority_lane = st.session_state.priority_lane
    max_count = max(values) if values else 1
    for rank, (lane, count) in enumerate(lane_priorities, 1):
        bar_width = int(100 * count / max_count) if max_count > 0 else 0
        bar_color = "#4CAF50" if lane == priority_lane else "#F44336"
        st.markdown(f"""
        <div style="margin-bottom: 10px; display: flex; align-items: center;">
            <div style="width: 80px; text-align: center; margin-right: 10px;">Lane {lane}</div>
            <div style="width: {bar_width}%; background-color: {bar_color}; height: 30px; 
                 display: flex; align-items: center; padding-left: 10px; color: white;">
                {count} vehicles
            </div>
            <div style="margin-left: 10px;">
                {' 🟢 GREEN (Active)' if lane == priority_lane else ' 🟡 YELLOW' if st.session_state.traffic_states[f'lane{lane}']['light'] == 'yellow' else ' 🔴 RED'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    if st.session_state.green_timer_active:
        st.info(f"⏱️ Green signal will automatically turn off after {GREEN_TIME} seconds")
else:
    st.warning("No lane has been analyzed yet. Click 'Analyze Traffic' to detect vehicles and set priority.")

# --- Arduino status ---
st.markdown("---")
st.subheader("Arduino Status")
if st.session_state.arduino and st.session_state.arduino.is_open:
    st.success("✅ Arduino Connected - Ready to send signals to pin 9")
else:
    st.error("❌ Arduino Not Connected - Cannot send signals")

if analyze_button:
    with st.spinner("Analyzing traffic in all lanes..."):
        analyze_traffic_and_update_priority()
        st.experimental_rerun()

# --- Real-time rerun for timers, so lane switches are faster ---
if st.session_state.priority_lane is not None and any(t > 0 for t in st.session_state.lane_timers):
    time.sleep(0.2)  # Moderate sleep time for good responsiveness
    st.experimental_rerun()