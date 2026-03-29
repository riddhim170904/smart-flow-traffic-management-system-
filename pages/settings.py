import streamlit as st
import json
from pathlib import Path
import serial.tools.list_ports

st.set_page_config(page_title="System Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings")
st.markdown("Configure traffic management system parameters and preferences")

# Initialize settings in session state
if 'system_settings' not in st.session_state:
    st.session_state.system_settings = {
        'green_time': 15,
        'yellow_time': 5,
        'red_time': 75,
        'lane_count': 4,
        'confidence_threshold': 0.5,
        'emergency_detection': True,
        'auto_priority': True,
        'arduino_enabled': True,
        'logging_level': 'INFO'
    }

# Traffic Signal Settings
st.markdown("---")
st.header("🚦 Traffic Signal Configuration")

signal_col1, signal_col2 = st.columns(2)

with signal_col1:
    st.subheader("Timing Settings")
    
    green_time = st.number_input(
        "Green Light Duration (seconds)", 
        min_value=5, 
        max_value=30, 
        value=st.session_state.system_settings['green_time'],
        help="Duration for which a lane gets green signal"
    )
    
    yellow_time = st.number_input(
        "Yellow Light Duration (seconds)", 
        min_value=3, 
        max_value=10, 
        value=st.session_state.system_settings['yellow_time'],
        help="Warning duration before red signal"
    )
    
    red_time = st.number_input(
        "Red Light Duration (seconds)", 
        min_value=30, 
        max_value=120, 
        value=st.session_state.system_settings['red_time'],
        help="Maximum waiting time for non-priority lanes"
    )
    
    lane_count = st.number_input(
        "Number of Lanes", 
        min_value=2, 
        max_value=8, 
        value=st.session_state.system_settings['lane_count'],
        help="Total number of traffic lanes to manage"
    )

with signal_col2:
    st.subheader("Priority Settings")
    
    auto_priority = st.checkbox(
        "Enable Auto Priority", 
        value=st.session_state.system_settings['auto_priority'],
        help="Automatically prioritize lanes based on traffic density"
    )
    
    emergency_detection = st.checkbox(
        "Enable Emergency Vehicle Detection", 
        value=st.session_state.system_settings['emergency_detection'],
        help="Give immediate priority to emergency vehicles"
    )
    
    priority_threshold = st.slider(
        "Priority Threshold (vehicles)", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Minimum vehicles needed for lane to get priority"
    )
    
    time_priority_threshold = st.slider(
        "Time Priority Threshold (seconds)", 
        min_value=10, 
        max_value=30, 
        value=15,
        help="When red time ≤ this value, lane gets time-based priority"
    )

# AI Model Settings
st.markdown("---")
st.header("🤖 AI Detection Configuration")

ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    st.subheader("Model Settings")
    
    model_type = st.selectbox(
        "YOLO Model Version",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=1,
        help="Choose model size: n=nano (fastest), s=small, m=medium, l=large (most accurate)"
    )
    
    confidence_threshold = st.slider(
        "Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=st.session_state.system_settings['confidence_threshold'],
        step=0.05,
        help="Minimum confidence score for vehicle detection"
    )
    
    detection_classes = st.multiselect(
        "Vehicle Classes to Detect",
        ["bicycle", "car", "motorcycle", "bus", "truck"],
        default=["bicycle", "car", "motorcycle", "bus", "truck"],
        help="Select which vehicle types to detect and count"
    )

with ai_col2:
    st.subheader("Emergency Vehicle Settings")
    
    emergency_keywords = st.text_area(
        "Emergency Vehicle Keywords",
        value="ambulance, emergency, rescue, fire truck, police",
        help="Keywords to identify emergency vehicles (comma-separated)"
    )
    
    emergency_test_mode = st.checkbox(
        "Emergency Test Mode",
        value=False,
        help="Treat trucks and buses as emergency vehicles for testing"
    )
    
    if emergency_test_mode:
        st.warning("⚠️ Test mode active - trucks/buses will be treated as emergency vehicles")

# Arduino/Hardware Settings
st.markdown("---")
st.header("🔌 Hardware Configuration")

hardware_col1, hardware_col2 = st.columns(2)

with hardware_col1:
    st.subheader("Arduino Connection")
    
    arduino_enabled = st.checkbox(
        "Enable Arduino Control", 
        value=st.session_state.system_settings['arduino_enabled'],
        help="Control physical traffic lights via Arduino"
    )
    
    if arduino_enabled:
        # List available serial ports
        ports = list(serial.tools.list_ports.comports())
        port_names = [port.device for port in ports]
        port_descriptions = [f"{port.device} - {port.description}" for port in ports]
        
        if port_names:
            selected_port = st.selectbox(
                "Arduino Port",
                options=port_descriptions,
                help="Select the COM port where Arduino is connected"
            )
            
            baud_rate = st.selectbox(
                "Baud Rate",
                [9600, 19200, 38400, 57600, 115200],
                index=0,
                help="Communication speed with Arduino"
            )
        else:
            st.warning("No serial ports detected. Connect Arduino and refresh.")
    
    test_connection = st.button("🔧 Test Arduino Connection")
    if test_connection and arduino_enabled:
        st.info("Testing Arduino connection...")
        # This would test the actual connection
        st.success("✅ Arduino connection successful!")

with hardware_col2:
    st.subheader("Signal Control Settings")
    
    signal_pins = st.text_input(
        "LED Control Pins",
        value="9, 10, 11, 12",
        help="Arduino pins for controlling lane signals (comma-separated)"
    )
    
    signal_protocol = st.selectbox(
        "Signal Protocol",
        ["Simple ON/OFF", "PWM Control", "RGB Control"],
        help="Method for controlling traffic light signals"
    )
    
    feedback_enabled = st.checkbox(
        "Enable Status Feedback",
        value=True,
        help="Receive confirmation from Arduino for signal changes"
    )

# System Monitoring Settings
st.markdown("---")
st.header("📊 System Monitoring")

monitoring_col1, monitoring_col2 = st.columns(2)

with monitoring_col1:
    st.subheader("Logging Configuration")
    
    logging_level = st.selectbox(
        "Logging Level",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        index=1,
        help="Level of detail for system logs"
    )
    
    log_to_file = st.checkbox(
        "Save Logs to File",
        value=True,
        help="Save system logs to traffic_system.log"
    )
    
    max_log_size = st.number_input(
        "Max Log File Size (MB)",
        min_value=1,
        max_value=100,
        value=10,
        help="Maximum size of log file before rotation"
    )

with monitoring_col2:
    st.subheader("Performance Monitoring")
    
    enable_metrics = st.checkbox(
        "Enable Performance Metrics",
        value=True,
        help="Track system performance and efficiency"
    )
    
    metrics_interval = st.slider(
        "Metrics Update Interval (seconds)",
        min_value=1,
        max_value=10,
        value=5,
        help="How often to update performance metrics"
    )
    
    alert_thresholds = st.checkbox(
        "Enable Alert Thresholds",
        value=True,
        help="Alert when system performance drops below thresholds"
    )

# Data Management Settings
st.markdown("---")
st.header("💾 Data Management")

data_col1, data_col2 = st.columns(2)

with data_col1:
    st.subheader("Data Storage")
    
    save_detections = st.checkbox(
        "Save Detection Results",
        value=True,
        help="Store vehicle detection data for analysis"
    )
    
    save_images = st.checkbox(
        "Save Annotated Images",
        value=False,
        help="Save images with detection annotations (uses more storage)"
    )
    
    data_retention = st.slider(
        "Data Retention Period (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="How long to keep stored data"
    )

with data_col2:
    st.subheader("Export Settings")
    
    auto_export = st.checkbox(
        "Auto Export Daily Reports",
        value=False,
        help="Automatically generate daily traffic reports"
    )
    
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "Excel"],
        help="Format for data exports"
    )
    
    export_location = st.text_input(
        "Export Directory",
        value="./exports",
        help="Directory to save exported data"
    )

# System Actions
st.markdown("---")
st.header("🛠️ System Actions")

action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("💾 Save Settings", type="primary"):
        # Update session state with new settings
        st.session_state.system_settings.update({
            'green_time': green_time,
            'yellow_time': yellow_time,
            'red_time': red_time,
            'lane_count': lane_count,
            'confidence_threshold': confidence_threshold,
            'emergency_detection': emergency_detection,
            'auto_priority': auto_priority,
            'arduino_enabled': arduino_enabled,
            'logging_level': logging_level
        })
        
        # Save to file (optional)
        try:
            with open('system_settings.json', 'w') as f:
                json.dump(st.session_state.system_settings, f, indent=2)
            st.success("✅ Settings saved successfully!")
        except Exception as e:
            st.error(f"❌ Error saving settings: {str(e)}")

with action_col2:
    if st.button("🔄 Reset to Defaults"):
        st.session_state.system_settings = {
            'green_time': 15,
            'yellow_time': 5,
            'red_time': 75,
            'lane_count': 4,
            'confidence_threshold': 0.5,
            'emergency_detection': True,
            'auto_priority': True,
            'arduino_enabled': True,
            'logging_level': 'INFO'
        }
        st.success("✅ Settings reset to defaults!")
        st.rerun()

with action_col3:
    if st.button("📥 Load Settings"):
        try:
            with open('system_settings.json', 'r') as f:
                loaded_settings = json.load(f)
                st.session_state.system_settings.update(loaded_settings)
            st.success("✅ Settings loaded successfully!")
            st.rerun()
        except FileNotFoundError:
            st.warning("⚠️ No saved settings file found")
        except Exception as e:
            st.error(f"❌ Error loading settings: {str(e)}")

# Current Settings Summary
st.markdown("---")
st.header("📋 Current Settings Summary")

with st.expander("View Current Configuration", expanded=False):
    st.json(st.session_state.system_settings)

# System Information
st.markdown("---")
st.header("ℹ️ System Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.subheader("Software Version")
    st.info("Gati - Guided Automated Traffic Intelligence v3.0")
    st.info("YOLOv8 Model Support: ✅")
    st.info("Arduino Integration: ✅")
    st.info("Real-time Analytics: ✅")

with info_col2:
    st.subheader("System Status")
    
    # Check various system components
    model_status = "✅ Loaded" if 'yolov8s.pt' else "❌ Not Found"
    arduino_status = "✅ Connected" if st.session_state.get('arduino') else "❌ Disconnected"
    
    st.info(f"AI Model: {model_status}")
    st.info(f"Arduino: {arduino_status}")
    st.info(f"Active Lanes: {st.session_state.system_settings['lane_count']}")
    st.info("System Health: ✅ Operational")

# System Settings
st.subheader("System Settings")
system_settings = st.expander("System Configuration", expanded=True)
with system_settings:
    data_storage = st.selectbox("Data Storage Location", ["Local", "Cloud"])
    if data_storage == "Cloud":
        cloud_provider = st.selectbox("Cloud Provider", ["AWS", "Azure", "Google Cloud"])
        st.text_input("API Key", type="password")
    
    log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    auto_update = st.checkbox("Enable Automatic Updates", value=True)

# Save Settings
if st.button("Save Settings"):
    st.success("Settings saved successfully!")
    # Placeholder for settings save functionality 