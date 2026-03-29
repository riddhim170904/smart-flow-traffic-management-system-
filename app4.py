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

# --- Basic logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SmartTraffic")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Smart Traffic System", page_icon="🚦", layout="wide")

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**Current Date and Time:** {current_time}")
st.markdown("**Current User's Login:** dhairyagothi")

# --- Timer logic for lanes (no detection, just timer UI) ---
LANE_COUNT = 4
GREEN_TIME = 15
RED_TIME = 75

# Initialize timers and green lane
if 'timers' not in st.session_state:
    st.session_state.timers = [RED_TIME] * LANE_COUNT
    logger.info(f"Initialized timers: {st.session_state.timers}")

if 'green_lane' not in st.session_state:
    st.session_state.green_lane = 0  # Lane 0 is green initially
    logger.info(f"Initialized green_lane: Lane {st.session_state.green_lane + 1}")

if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
    logger.info("Initialized last_update time.")

# On first run, set only green lane to GREEN_TIME, others to RED_TIME
if 'initialized_once' not in st.session_state:
    for i in range(LANE_COUNT):
        st.session_state.timers[i] = GREEN_TIME if i == st.session_state.green_lane else RED_TIME
    st.session_state.initialized_once = True

# Button to manually advance green to next lane
if st.button("Next Lane"):
    prev_green = st.session_state.green_lane
    st.session_state.timers[prev_green] = RED_TIME
    st.session_state.green_lane = (prev_green + 1) % LANE_COUNT
    st.session_state.timers[st.session_state.green_lane] = GREEN_TIME
    logger.info(f"Green lane switched to Lane {st.session_state.green_lane + 1}")

# Real-time timer update
current_time = time.time()
elapsed = current_time - st.session_state.last_update
if elapsed >= 1:
    for i in range(LANE_COUNT):
        if st.session_state.timers[i] > 0:
            st.session_state.timers[i] -= 1
            logger.info(f"Lane {i+1} countdown: {st.session_state.timers[i]} sec remaining")
    st.session_state.last_update = current_time

    # If green lane reached 0, auto-advance
    if st.session_state.timers[st.session_state.green_lane] == 0:
        prev_green = st.session_state.green_lane
        st.session_state.timers[prev_green] = RED_TIME
        st.session_state.green_lane = (prev_green + 1) % LANE_COUNT
        st.session_state.timers[st.session_state.green_lane] = GREEN_TIME
        logger.info(f"Green lane automatically switched to Lane {st.session_state.green_lane + 1}")

# --- Display lane countdowns and status ---
lane_cols = st.columns(LANE_COUNT)
for i, col in enumerate(lane_cols):
    with col:
        color = "🟢 GREEN" if i == st.session_state.green_lane else "🔴 RED"
        st.markdown(f"### Lane {i+1}")
        st.markdown(f"**{color}**")
        st.metric(label="Time Remaining", value=f"{st.session_state.timers[i]} sec")

# Only rerun if at least one timer is running
if any(t > 0 for t in st.session_state.timers):
    time.sleep(1)
    st.experimental_rerun()