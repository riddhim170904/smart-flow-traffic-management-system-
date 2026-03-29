import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Traffic Analytics", page_icon="📈", layout="wide")

st.title("📈 Traffic Analytics & Insights")
st.markdown("Comprehensive traffic pattern analysis and performance insights")

# Initialize analytics data in session state
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = {
        'historical_data': [],
        'lane_performance': {f'lane{i+1}': [] for i in range(4)},
        'priority_switches': [],
        'emergency_events': [],
        'efficiency_metrics': []
    }

# Constants from main app
LANE_COUNT = 4
GREEN_TIME = 15
RED_TIME = 75

# Date range selector
st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
with col2:
    end_date = st.date_input("End Date", datetime.now())
with col3:
    analysis_type = st.selectbox("Analysis Type", ["Real-time", "Historical", "Predictive"])

# Current system analytics
st.markdown("---")
st.header("Current System Performance")

# Get current traffic data
current_data = st.session_state.get('traffic_data', {})
priority_order = st.session_state.get('priority_order', [1, 2, 3, 4])
current_priority = st.session_state.get('priority_lane', None)

# Calculate analytics metrics
total_vehicles = sum(current_data.get(f'lane{i+1}', {}).get('vehicles', 0) for i in range(LANE_COUNT))
total_emergency = sum(current_data.get(f'lane{i+1}', {}).get('emergency_vehicles', 0) for i in range(LANE_COUNT))
avg_vehicles_per_lane = total_vehicles / LANE_COUNT if LANE_COUNT > 0 else 0

# Performance metrics row
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric(
        label="Total Vehicles Detected",
        value=total_vehicles,
        delta=f"Avg: {avg_vehicles_per_lane:.1f}/lane"
    )

with metrics_col2:
    st.metric(
        label="Priority Switches Today",
        value=len(st.session_state.analytics_data['priority_switches']),
        delta="Optimal" if len(st.session_state.analytics_data['priority_switches']) < 20 else "High"
    )

with metrics_col3:
    st.metric(
        label="Emergency Events",
        value=total_emergency,
        delta="🚨 Active" if total_emergency > 0 else "✅ Clear"
    )

with metrics_col4:
    efficiency = min(100, (total_vehicles / (LANE_COUNT * 8)) * 100) if total_vehicles > 0 else 0
    st.metric(
        label="System Efficiency",
        value=f"{efficiency:.1f}%",
        delta="Excellent" if efficiency > 80 else "Good" if efficiency > 60 else "Needs Attention"
    )

# Traffic patterns analysis
st.markdown("---")
st.header("Traffic Patterns Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Lane Performance", "Priority Analysis", "Time-based Patterns", "Emergency Events"])

with tab1:
    st.subheader("Lane-by-Lane Performance")
    
    # Create performance comparison
    lane_perf_col1, lane_perf_col2 = st.columns(2)
    
    with lane_perf_col1:
        st.write("**Current Vehicle Distribution**")
        lane_data = []
        for i in range(LANE_COUNT):
            lane_key = f'lane{i+1}'
            vehicles = current_data.get(lane_key, {}).get('vehicles', 0)
            emergency = current_data.get(lane_key, {}).get('emergency_vehicles', 0)
            
            # Get traffic light status
            traffic_states = st.session_state.get('traffic_states', {})
            light_status = traffic_states.get(lane_key, {}).get('light', 'red')
            
            lane_data.append({
                'Lane': f'Lane {i+1}',
                'Vehicles': vehicles,
                'Emergency': emergency,
                'Status': light_status.upper(),
                'Priority Rank': priority_order.index(i+1) + 1 if (i+1) in priority_order else 'N/A'
            })
        
        df_lanes = pd.DataFrame(lane_data)
        st.dataframe(df_lanes, use_container_width=True)
    
    with lane_perf_col2:
        st.write("**Lane Utilization Chart**")
        if total_vehicles > 0:
            # Create a simple bar chart using Streamlit
            lane_vehicles = [current_data.get(f'lane{i+1}', {}).get('vehicles', 0) for i in range(LANE_COUNT)]
            lane_names = [f'Lane {i+1}' for i in range(LANE_COUNT)]
            
            chart_data = pd.DataFrame({
                'Lane': lane_names,
                'Vehicles': lane_vehicles
            })
            st.bar_chart(chart_data.set_index('Lane'))
        else:
            st.info("No vehicle data available. Start traffic analysis to see utilization.")

with tab2:
    st.subheader("Priority Decision Analysis")
    
    priority_col1, priority_col2 = st.columns(2)
    
    with priority_col1:
        st.write("**Current Priority Logic**")
        if current_priority:
            current_lane_data = current_data.get(f'lane{current_priority}', {})
            vehicles = current_lane_data.get('vehicles', 0)
            emergency = current_lane_data.get('emergency_vehicles', 0)
            
            # Determine priority reason
            if emergency > 0:
                reason = f"🚨 Emergency vehicles detected ({emergency})"
                priority_type = "Emergency Priority"
            elif vehicles >= 5:
                reason = f"🚗 High traffic volume ({vehicles} vehicles)"
                priority_type = "Traffic Volume Priority"
            else:
                reason = "⏰ Time-based rotation"
                priority_type = "Time Priority"
            
            st.success(f"**Active Lane:** {current_priority}")
            st.info(f"**Priority Type:** {priority_type}")
            st.write(f"**Reason:** {reason}")
        else:
            st.warning("No active priority lane")
    
    with priority_col2:
        st.write("**Priority Queue Status**")
        for rank, lane in enumerate(priority_order, 1):
            lane_data = current_data.get(f'lane{lane}', {})
            vehicles = lane_data.get('vehicles', 0)
            emergency = lane_data.get('emergency_vehicles', 0)
            
            if lane == current_priority:
                status = "🟢 ACTIVE"
            elif rank <= 2:
                status = "🟡 HIGH PRIORITY"
            else:
                status = "🔴 WAITING"
            
            st.write(f"**{rank}.** Lane {lane} - {vehicles} vehicles {status}")
            if emergency > 0:
                st.write(f"   🚨 {emergency} emergency vehicle(s)")

with tab3:
    st.subheader("Time-based Traffic Patterns")
    
    time_col1, time_col2 = st.columns(2)
    
    with time_col1:
        st.write("**Signal Timing Analysis**")
        if 'lane_timers' in st.session_state:
            timer_data = []
            for i, timer in enumerate(st.session_state.lane_timers):
                phase = "Green" if (current_priority and i == (current_priority - 1)) else "Red"
                timer_data.append({
                    'Lane': f'Lane {i+1}',
                    'Current Timer': f"{timer}s",
                    'Time Remaining': f"{max(0, timer)}s",
                    'Phase': phase
                })
            
            df_timers = pd.DataFrame(timer_data)
            st.dataframe(df_timers, use_container_width=True)
        else:
            st.info("Timer data not available")
    
    with time_col2:
        st.write("**Cycle Efficiency Metrics**")
        # Calculate cycle metrics
        if current_priority and 'lane_timers' in st.session_state:
            active_timer = st.session_state.lane_timers[current_priority - 1] if current_priority <= len(st.session_state.lane_timers) else 0
            
            st.metric("Current Cycle Progress", f"{GREEN_TIME - active_timer}s / {GREEN_TIME}s")
            st.metric("Cycle Efficiency", f"{min(100, ((GREEN_TIME - active_timer) / GREEN_TIME) * 100):.1f}%")
            
            # Average wait time estimation
            avg_wait = sum(st.session_state.lane_timers) / len(st.session_state.lane_timers)
            st.metric("Avg Wait Time", f"{avg_wait:.1f}s")
        else:
            st.info("Cycle data not available")

with tab4:
    st.subheader("Emergency Vehicle Events")
    
    if total_emergency > 0:
        st.error(f"🚨 **ACTIVE EMERGENCY SITUATION** - {total_emergency} emergency vehicle(s) detected")
        
        emergency_col1, emergency_col2 = st.columns(2)
        
        with emergency_col1:
            st.write("**Emergency Vehicle Locations**")
            for i in range(LANE_COUNT):
                lane_key = f'lane{i+1}'
                emergency_count = current_data.get(lane_key, {}).get('emergency_vehicles', 0)
                if emergency_count > 0:
                    st.write(f"🚨 **Lane {i+1}:** {emergency_count} emergency vehicle(s)")
        
        with emergency_col2:
            st.write("**Emergency Response Actions**")
            if current_priority:
                emergency_in_priority = current_data.get(f'lane{current_priority}', {}).get('emergency_vehicles', 0)
                if emergency_in_priority > 0:
                    st.success("✅ Emergency lane has priority")
                else:
                    st.warning("⚠️ Emergency vehicle in non-priority lane")
            
            st.info("📍 Emergency protocol: Immediate priority given to emergency vehicles")
    else:
        st.success("✅ No emergency vehicles detected")
        
        # Historical emergency events (simulated)
        st.write("**Recent Emergency Events**")
        st.info("No recent emergency events recorded")

# Advanced analytics section
st.markdown("---")
st.header("Advanced Analytics")

advanced_col1, advanced_col2 = st.columns(2)

with advanced_col1:
    st.subheader("Traffic Flow Optimization")
    
    # Calculate optimization score
    if total_vehicles > 0:
        # Factors for optimization score
        vehicle_distribution = np.std([current_data.get(f'lane{i+1}', {}).get('vehicles', 0) for i in range(LANE_COUNT)])
        emergency_response = 100 if total_emergency == 0 else 50  # Lower score if emergencies not handled
        
        optimization_score = max(0, 100 - (vehicle_distribution * 10))
        
        st.metric("Traffic Distribution Score", f"{optimization_score:.1f}/100")
        st.metric("Emergency Response Score", f"{emergency_response}/100")
        
        overall_score = (optimization_score + emergency_response) / 2
        st.metric("Overall Optimization Score", f"{overall_score:.1f}/100")
        
        if overall_score > 80:
            st.success("🟢 Excellent traffic flow optimization")
        elif overall_score > 60:
            st.warning("🟡 Good optimization with room for improvement")
        else:
            st.error("🔴 Traffic flow needs attention")
    else:
        st.info("Start traffic analysis to see optimization metrics")

with advanced_col2:
    st.subheader("System Recommendations")
    
    recommendations = []
    
    # Generate recommendations based on current state
    if total_vehicles > 0:
        # Check for uneven distribution
        vehicle_counts = [current_data.get(f'lane{i+1}', {}).get('vehicles', 0) for i in range(LANE_COUNT)]
        max_vehicles = max(vehicle_counts)
        min_vehicles = min(vehicle_counts)
        
        if max_vehicles - min_vehicles > 3:
            recommendations.append("🔄 Consider redistributing traffic - uneven lane utilization detected")
        
        # Check for emergency vehicles
        if total_emergency > 0:
            recommendations.append("🚨 Emergency vehicles detected - ensure priority lanes are optimized")
        
        # Check for long wait times
        if 'lane_timers' in st.session_state:
            long_waits = [i+1 for i, timer in enumerate(st.session_state.lane_timers) if timer > 60]
            if long_waits:
                recommendations.append(f"⏰ Long wait times in lanes: {', '.join(map(str, long_waits))}")
        
        # Traffic volume recommendations
        if total_vehicles > 15:
            recommendations.append("📈 High traffic volume - consider shorter green light cycles")
        elif total_vehicles < 5:
            recommendations.append("📉 Low traffic volume - consider longer green light cycles")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
    else:
        st.success("✅ System operating optimally - no recommendations at this time")

# Data export section
st.markdown("---")
st.header("Data Export & Reporting")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("📊 Export Current Data"):
        # Create current data summary
        current_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_vehicles': total_vehicles,
            'lane_data': current_data,
            'priority_order': priority_order,
            'current_priority': current_priority,
            'emergency_events': total_emergency
        }
        st.json(current_summary)

with export_col2:
    if st.button("📈 Generate Report"):
        st.info("Generating comprehensive traffic report...")
        st.success("Report generated successfully! (Feature in development)")

with export_col3:
    if st.button("🔄 Refresh Analytics"):
        st.rerun()

# Real-time updates
if total_vehicles > 0 or current_priority is not None:
    # Store current data for historical analysis
    current_time = datetime.now()
    st.session_state.analytics_data['historical_data'].append({
        'timestamp': current_time,
        'total_vehicles': total_vehicles,
        'priority_lane': current_priority,
        'emergency_count': total_emergency
    })
    
    # Keep only recent data (last 100 entries)
    if len(st.session_state.analytics_data['historical_data']) > 100:
        st.session_state.analytics_data['historical_data'] = st.session_state.analytics_data['historical_data'][-100:]
st.info("Traffic density heatmap will be implemented here")

# Export options
st.subheader("Export Data")
export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "PDF"])
if st.button("Export Data"):
    st.info("Export functionality will be implemented here") 