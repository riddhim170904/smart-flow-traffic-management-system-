import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Traffic Dashboard", page_icon="📊", layout="wide")

# Initialize session state for dashboard data
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = {
        'timestamps': [],
        'lane_data': {f'lane{i+1}': [] for i in range(4)},
        'total_vehicles': [],
        'avg_wait_time': [],
        'system_efficiency': []
    }

if 'last_dashboard_update' not in st.session_state:
    st.session_state.last_dashboard_update = time.time()

# Constants from main app
LANE_COUNT = 4
GREEN_TIME = 15
YELLOW_TIME = 5
RED_TIME = 75

st.title("🚦 Traffic Monitoring Dashboard")

# Current system status section
st.markdown("---")
st.header("Real-Time System Status")

# Create metrics row
col1, col2, col3, col4 = st.columns(4)

# Get current traffic data from main app session state
current_total_vehicles = 0
active_lanes = 0
emergency_alerts = 0

if 'traffic_data' in st.session_state:
    for i in range(LANE_COUNT):
        lane_key = f'lane{i+1}'
        if lane_key in st.session_state.traffic_data:
            current_total_vehicles += st.session_state.traffic_data[lane_key].get('vehicles', 0)
            if st.session_state.traffic_data[lane_key].get('vehicles', 0) > 0:
                active_lanes += 1
            emergency_alerts += st.session_state.traffic_data[lane_key].get('emergency_vehicles', 0)

# Current priority lane
current_priority = st.session_state.get('priority_lane', 'None')

# System efficiency calculation (simplified)
max_possible_vehicles = LANE_COUNT * 10  # Assume max 10 vehicles per lane
efficiency = min(100, (current_total_vehicles / max_possible_vehicles) * 100) if max_possible_vehicles > 0 else 0

with col1:
    st.metric(
        label="Total Vehicles",
        value=current_total_vehicles,
        delta=f"{active_lanes} active lanes"
    )

with col2:
    st.metric(
        label="Priority Lane",
        value=f"Lane {current_priority}" if current_priority != 'None' else "None",
        delta="Active" if current_priority != 'None' else "Idle"
    )

with col3:
    st.metric(
        label="Emergency Alerts",
        value=emergency_alerts,
        delta="🚨 Active" if emergency_alerts > 0 else "✅ Clear"
    )

with col4:
    st.metric(
        label="System Efficiency",
        value=f"{efficiency:.1f}%",
        delta="Optimal" if efficiency > 70 else "Needs attention"
    )

# Live traffic status cards
st.markdown("---")
st.header("Lane Status Overview")

lane_cols = st.columns(4)
for i in range(LANE_COUNT):
    lane_num = i + 1
    lane_key = f'lane{lane_num}'
    
    with lane_cols[i]:
        # Get traffic light status
        traffic_states = st.session_state.get('traffic_states', {})
        lane_timers = st.session_state.get('lane_timers', [RED_TIME] * LANE_COUNT)
        
        if lane_key in traffic_states:
            light_state = traffic_states[lane_key]['light']
            timer_value = max(0, int(lane_timers[i])) if i < len(lane_timers) else 0
        else:
            light_state = 'red'
            timer_value = RED_TIME
        
        # Get vehicle count
        traffic_data = st.session_state.get('traffic_data', {})
        vehicle_count = traffic_data.get(lane_key, {}).get('vehicles', 0)
        
        # Determine status color and icon
        if light_state == 'green':
            status_color = "#4CAF50"
            status_icon = "🟢"
            status_text = "GREEN"
        elif light_state == 'yellow':
            status_color = "#FFC107"
            status_icon = "🟡"
            status_text = "YELLOW"
        else:
            status_color = "#F44336"
            status_icon = "🔴"
            status_text = "RED"
        
        st.markdown(f"""
        <div style="
            border: 2px solid {status_color};
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            background-color: rgba({255 if status_color == '#F44336' else 76 if status_color == '#4CAF50' else 255}, 
                                    {67 if status_color == '#F44336' else 175 if status_color == '#4CAF50' else 193}, 
                                    {54 if status_color == '#F44336' else 80 if status_color == '#4CAF50' else 7}, 0.1);
        ">
            <h3>{status_icon} Lane {lane_num}</h3>
            <p><strong>Status:</strong> {status_text}</p>
            <p><strong>Vehicles:</strong> {vehicle_count}</p>
            <p><strong>Timer:</strong> {timer_value}s</p>
        </div>
        """, unsafe_allow_html=True)

# Real-time charts section
st.markdown("---")
st.header("Real-Time Analytics")

# Update dashboard data
current_time = datetime.now()
if current_time.timestamp() - st.session_state.last_dashboard_update > 5:  # Update every 5 seconds
    st.session_state.dashboard_data['timestamps'].append(current_time)
    
    # Add current lane data
    for i in range(LANE_COUNT):
        lane_key = f'lane{i+1}'
        vehicles = st.session_state.get('traffic_data', {}).get(lane_key, {}).get('vehicles', 0)
        st.session_state.dashboard_data['lane_data'][lane_key].append(vehicles)
    
    st.session_state.dashboard_data['total_vehicles'].append(current_total_vehicles)
    st.session_state.dashboard_data['avg_wait_time'].append(np.random.uniform(30, 90))  # Simulated wait time
    st.session_state.dashboard_data['system_efficiency'].append(efficiency)
    
    # Keep only last 20 data points
    max_points = 20
    for key in st.session_state.dashboard_data:
        if isinstance(st.session_state.dashboard_data[key], list):
            st.session_state.dashboard_data[key] = st.session_state.dashboard_data[key][-max_points:]
        elif isinstance(st.session_state.dashboard_data[key], dict):
            for lane_key in st.session_state.dashboard_data[key]:
                st.session_state.dashboard_data[key][lane_key] = st.session_state.dashboard_data[key][lane_key][-max_points:]
    
    st.session_state.last_dashboard_update = current_time.timestamp()

# Create charts with improved styling and features
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🚗 Real-Time Vehicle Count by Lane")
    
    if st.session_state.dashboard_data['timestamps']:
        # Create DataFrame for lane data
        df_lanes = pd.DataFrame({
            'Time': st.session_state.dashboard_data['timestamps'],
            **st.session_state.dashboard_data['lane_data']
        })
        
        # Create enhanced line chart with better styling
        fig_lanes = go.Figure()
        
        # Define colors for each lane
        lane_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        lane_names = ['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4']
        
        for i, (lane_key, color, name) in enumerate(zip(['lane1', 'lane2', 'lane3', 'lane4'], lane_colors, lane_names)):
            fig_lanes.add_trace(go.Scatter(
                x=df_lanes['Time'],
                y=df_lanes[lane_key],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=6, symbol='circle'),
                hovertemplate=f'<b>{name}</b><br>' +
                             'Time: %{x}<br>' +
                             'Vehicles: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        fig_lanes.update_layout(
            title=dict(
                text="Vehicle Count Trends by Lane",
                font=dict(size=16, color='#2E86AB')
            ),
            xaxis=dict(
                title="Time",
                gridcolor='#E5E5E5',
                showgrid=True
            ),
            yaxis=dict(
                title="Number of Vehicles",
                gridcolor='#E5E5E5',
                showgrid=True
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_lanes, use_container_width=True)
    else:
        st.info("📊 No data available yet. Start traffic analysis to see real-time charts.")

with chart_col2:
    st.subheader("📈 System Performance Metrics")
    
    if st.session_state.dashboard_data['timestamps']:
        # Create DataFrame for system metrics
        df_system = pd.DataFrame({
            'Time': st.session_state.dashboard_data['timestamps'],
            'Total Vehicles': st.session_state.dashboard_data['total_vehicles'],
            'Efficiency (%)': st.session_state.dashboard_data['system_efficiency'],
            'Avg Wait Time': st.session_state.dashboard_data['avg_wait_time']
        })
        
        # Create enhanced dual-axis chart
        fig_system = go.Figure()
        
        # Add total vehicles trace
        fig_system.add_trace(go.Scatter(
            x=df_system['Time'],
            y=df_system['Total Vehicles'],
            mode='lines+markers',
            name='Total Vehicles',
            yaxis='y',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Total Vehicles</b><br>' +
                         'Time: %{x}<br>' +
                         'Count: %{y}<br>' +
                         '<extra></extra>'
        ))
        
        # Add efficiency trace
        fig_system.add_trace(go.Scatter(
            x=df_system['Time'],
            y=df_system['Efficiency (%)'],
            mode='lines+markers',
            name='System Efficiency',
            yaxis='y2',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>System Efficiency</b><br>' +
                         'Time: %{x}<br>' +
                         'Efficiency: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
        
        fig_system.update_layout(
            title=dict(
                text="Performance Overview",
                font=dict(size=16, color='#2E86AB')
            ),
            xaxis=dict(
                title="Time",
                gridcolor='#E5E5E5',
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text="Total Vehicles", font=dict(color='#FF6B6B')),
                side="left",
                gridcolor='#E5E5E5',
                showgrid=True,
                tickfont=dict(color='#FF6B6B')
            ),
            yaxis2=dict(
                title=dict(text="Efficiency (%)", font=dict(color='#4ECDC4')),
                side="right",
                overlaying="y",
                tickfont=dict(color='#4ECDC4'),
                range=[0, 100]
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_system, use_container_width=True)
    else:
        st.info("📊 No data available yet. Start traffic analysis to see real-time charts.")

# Additional advanced visualizations
st.markdown("---")
st.header("📊 Advanced Traffic Analytics")

# Create three columns for additional charts
adv_col1, adv_col2, adv_col3 = st.columns(3)

with adv_col1:
    st.subheader("🎯 Current Lane Distribution")
    
    if st.session_state.dashboard_data['timestamps']:
        # Get current vehicle distribution
        current_lane_data = {}
        for i in range(LANE_COUNT):
            lane_key = f'lane{i+1}'
            vehicles = st.session_state.get('traffic_data', {}).get(lane_key, {}).get('vehicles', 0)
            current_lane_data[f'Lane {i+1}'] = vehicles
        
        if sum(current_lane_data.values()) > 0:
            # Create pie chart for current distribution
            fig_pie = px.pie(
                values=list(current_lane_data.values()),
                names=list(current_lane_data.keys()),
                title="Current Vehicle Distribution",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>' +
                             'Vehicles: %{value}<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>'
            )
            fig_pie.update_layout(
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                title=dict(font=dict(size=14, color='#2E86AB'))
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("🚫 No vehicles detected")
    else:
        st.info("📊 Waiting for data...")

with adv_col2:
    st.subheader("⏱️ Traffic Light Status")
    
    # Create traffic light status chart
    traffic_states = st.session_state.get('traffic_states', {})
    lane_timers = st.session_state.get('lane_timers', [RED_TIME] * LANE_COUNT)
    
    lane_status_data = []
    for i in range(LANE_COUNT):
        lane_key = f'lane{i+1}'
        if lane_key in traffic_states:
            light_state = traffic_states[lane_key]['light']
            timer_value = max(0, int(lane_timers[i])) if i < len(lane_timers) else 0
        else:
            light_state = 'red'
            timer_value = RED_TIME
        
        lane_status_data.append({
            'Lane': f'Lane {i+1}',
            'Status': light_state.upper(),
            'Timer': timer_value,
            'Color': '#4CAF50' if light_state == 'green' else '#FFC107' if light_state == 'yellow' else '#F44336'
        })
    
    # Create horizontal bar chart for timers
    fig_status = go.Figure()
    
    for data in lane_status_data:
        fig_status.add_trace(go.Bar(
            y=[data['Lane']],
            x=[data['Timer']],
            orientation='h',
            name=data['Status'],
            marker_color=data['Color'],
            text=[f"{data['Timer']}s ({data['Status']})"],
            textposition='inside',
            hovertemplate=f'<b>{data["Lane"]}</b><br>' +
                         f'Status: {data["Status"]}<br>' +
                         f'Time Remaining: {data["Timer"]}s<br>' +
                         '<extra></extra>',
            showlegend=False
        ))
    
    fig_status.update_layout(
        title=dict(
            text="Lane Timer Status",
            font=dict(size=14, color='#2E86AB')
        ),
        xaxis_title="Time Remaining (seconds)",
        yaxis_title="Lane",
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_status, use_container_width=True)

with adv_col3:
    st.subheader("🚨 Priority & Alerts")
    
    # Priority ranking visualization
    priority_order = st.session_state.get('priority_order', [1, 2, 3, 4])
    priority_data = []
    
    for rank, lane in enumerate(priority_order, 1):
        lane_data = st.session_state.get('traffic_data', {}).get(f'lane{lane}', {})
        vehicle_count = lane_data.get('vehicles', 0)
        emergency_count = lane_data.get('emergency_vehicles', 0)
        
        priority_data.append({
            'Rank': rank,
            'Lane': f'Lane {lane}',
            'Vehicles': vehicle_count,
            'Emergency': emergency_count,
            'Priority_Score': (4 - rank + 1) * 25  # Convert rank to score for visualization
        })
    
    # Create priority ranking chart
    fig_priority = go.Figure()
    
    # Add priority bars
    fig_priority.add_trace(go.Bar(
        x=[d['Lane'] for d in priority_data],
        y=[d['Priority_Score'] for d in priority_data],
        name='Priority Score',
        marker_color=['#FF6B6B', '#FF8E53', '#FFC107', '#4ECDC4'],
        text=[f"Rank {d['Rank']}" for d in priority_data],
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>' +
                     'Priority Rank: %{text}<br>' +
                     'Vehicles: ' + str([d['Vehicles'] for d in priority_data]) + '<br>' +
                     '<extra></extra>'
    ))
    
    # Add emergency indicators
    emergency_y = [d['Priority_Score'] + 10 for d in priority_data if d['Emergency'] > 0]
    emergency_x = [d['Lane'] for d in priority_data if d['Emergency'] > 0]
    
    if emergency_x:
        fig_priority.add_trace(go.Scatter(
            x=emergency_x,
            y=emergency_y,
            mode='markers+text',
            name='Emergency Alert',
            marker=dict(
                symbol='star',
                size=15,
                color='red'
            ),
            text=['🚨'] * len(emergency_x),
            textposition='middle center',
            showlegend=False
        ))
    
    fig_priority.update_layout(
        title=dict(
            text="Lane Priority Ranking",
            font=dict(size=14, color='#2E86AB')
        ),
        xaxis_title="Lane",
        yaxis_title="Priority Score",
        yaxis=dict(range=[0, 110]),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_priority, use_container_width=True)
st.markdown("---")
st.header("Traffic Flow Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.subheader("Current Priority Order")
    priority_order = st.session_state.get('priority_order', [1, 2, 3, 4])
    for rank, lane in enumerate(priority_order, 1):
        lane_data = st.session_state.get('traffic_data', {}).get(f'lane{lane}', {})
        vehicle_count = lane_data.get('vehicles', 0)
        st.markdown(f"**{rank}.** Lane {lane} - {vehicle_count} vehicles")

with summary_col2:
    st.subheader("System Alerts")
    if emergency_alerts > 0:
        st.error(f"🚨 {emergency_alerts} Emergency vehicle(s) detected!")
    
    # Check for long waiting times
    if 'lane_timers' in st.session_state:
        long_wait_lanes = []
        for i, timer in enumerate(st.session_state.lane_timers):
            if timer > 60:  # More than 60 seconds
                long_wait_lanes.append(i + 1)
        
        if long_wait_lanes:
            st.warning(f"⏰ Long wait times in Lane(s): {', '.join(map(str, long_wait_lanes))}")
        else:
            st.success("✅ All lanes operating normally")
    else:
        st.info("📊 System monitoring active")

with summary_col3:
    st.subheader("Performance Metrics")
    if st.session_state.dashboard_data['timestamps']:
        avg_vehicles = np.mean(st.session_state.dashboard_data['total_vehicles'][-5:])  # Last 5 readings
        avg_efficiency = np.mean(st.session_state.dashboard_data['system_efficiency'][-5:])
        st.metric("Avg Vehicles (5min)", f"{avg_vehicles:.1f}")
        st.metric("Avg Efficiency (5min)", f"{avg_efficiency:.1f}%")
    else:
        st.info("Collecting performance data...")

# Auto-refresh
st.markdown("---")
refresh_col1, refresh_col2 = st.columns([3, 1])

with refresh_col1:
    st.info("📡 Dashboard auto-refreshes every 5 seconds. Data updates when traffic analysis is active.")

with refresh_col2:
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Auto-refresh functionality
if current_total_vehicles > 0 or st.session_state.get('priority_lane') is not None:
    time.sleep(2)  # Small delay for smooth updates
    st.rerun() 