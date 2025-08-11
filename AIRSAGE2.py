import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import threading
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
import io
from statsmodels.tsa.seasonal import seasonal_decompose

# ========================================
# AIR QUALITY MONITORING CONFIGURATION
# ========================================
POLLUTANTS = {
    'PM2.5': {'unit': 'µg/m³', 'thresholds': [12, 35, 55, 150], 'color': '#e63946'},
    'PM10': {'unit': 'µg/m³', 'thresholds': [54, 154, 254, 354], 'color': '#457b9d'},
    'O3': {'unit': 'ppb', 'thresholds': [54, 70, 85, 105], 'color': '#1d3557'},
    'NO2': {'unit': 'ppb', 'thresholds': [53, 100, 360, 649], 'color': '#588157'},
    'CO': {'unit': 'ppm', 'thresholds': [4.4, 9.4, 12.4, 15.4], 'color': '#d4a373'},
    'SO2': {'unit': 'ppb', 'thresholds': [35, 75, 185, 304], 'color': '#c1121f'}
}

# ========================================
# AIR QUALITY MONITOR CLASS
# ========================================
class AirQualityMonitor:
    def __init__(self):
        self.monitoring_active = False
        self.all_data = []
        self.start_time = None
        self.location = None  # Changed from zip_code to more generic location
        self.location_type = None  # 'zip' or 'latlon'
        self.last_update = None
        self.wind_data = []
        self._data_lock = threading.Lock()

    def fetch_airnow_data(self, api_key, location, location_type):
        if location_type == 'zip':
            url = f"https://www.airnowapi.org/aq/observation/zipCode/current/?format=application/json&zipCode={location}&distance=50&API_KEY={api_key}"
        else:  # lat/lon
            lat, lon = location
            url = f"https://www.airnowapi.org/aq/observation/latLong/current/?format=application/json&latitude={lat}&longitude={lon}&distance=50&API_KEY={api_key}"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
        except Exception as e:
            st.error(f"Network Error: {str(e)}")
            return None

    def collect_data(self, api_key, location, location_type, interval_minutes):
        while self.monitoring_active:
            data = self.fetch_airnow_data(api_key, location, location_type)
            if data:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with self._data_lock:
                    for entry in data:
                        if entry['ParameterName'] in POLLUTANTS:
                            entry['timestamp'] = timestamp
                            self.all_data.append(entry)
                        elif 'Wind' in entry.get('ParameterName', ''):
                            self.wind_data.append({
                                'timestamp': timestamp,
                                'direction': entry.get('WindDirection'),
                                'speed': entry.get('WindSpeed'),
                                'units': entry.get('Unit')
                            })
                    self.last_update = timestamp
            time.sleep(interval_minutes * 60)

    def get_latest_data(self):
        with self._data_lock:
            return self.all_data.copy(), self.wind_data.copy()

    def stop_monitoring(self):
        self.monitoring_active = False
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m"

    def generate_wind_plot(self):
        if not self.wind_data:
            return None
        
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
    
        df = pd.DataFrame(self.wind_data)
        if df.empty:
            return None
        
        df['direction_deg'] = df['direction'].map(wind_dir_map)
    
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
    
        speed_by_dir = df.groupby('direction')['speed'].mean()
    
        for direction, speed in speed_by_dir.items():
            angle = np.deg2rad(wind_dir_map[direction])
            ax.bar(angle, speed, width=np.pi/8, color='#1f77b4', alpha=0.7)
    
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(np.linspace(0, 2*np.pi, 16, endpoint=False))
        ax.set_xticklabels(list(wind_dir_map.keys()))
    
    # Updated location display logic
        if self.location_type == 'zip':
            location_str = f"ZIP: {self.location}"
        else:
            lat, lon = self.location
            location_str = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
    
        ax.set_title(f"Wind Rose Analysis\n{location_str}", pad=20)
        plt.tight_layout()
    
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        buf.seek(0)
        return buf

# ========================================
# CUSTOM CSS & STYLING
# ========================================
def inject_custom_css():
    st.markdown(
        """
        <style>
            .main {background-color:#f8fafc;}
            .gradient-title{
                font-family:'Helvetica Neue',sans-serif;
                font-size:4rem;
                font-weight:900;
                background:linear-gradient(135deg,#4361ee,#3a0ca3);
                -webkit-background-clip:text;
                background-clip:text;
                color:transparent;
                margin-bottom:.5rem;
                text-align:center;
            }
            .elegant-card{
                border-radius:12px;padding:25px;background:#fff;
                box-shadow:0 4px 20px rgba(0,0,0,0.08);
                margin-bottom:15px;border-left:4px solid #4361ee;
                transition:all .3s ease;
            }
            .elegant-card:hover{
                transform:translateY(-3px);
                box-shadow:0 6px 25px rgba(0,0,0,0.12);
            }
            .logo-container{
                display:flex;justify-content:center;align-items:center;margin-top:20px;
            }
            [data-testid="stSidebar"]{
                background:linear-gradient(180deg,#3a4a8a 0%,#1e2b5e 100%);
                color:#fff;
            }
            [data-testid="stSidebar"] p,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2{
                color:#fff !important;
            }
            @media(max-width:768px){
                .gradient-title{font-size:2.8rem;}
                .logo-container img{width:140px;}
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ========================================
# MONITORING PAGE
# ========================================
def show_monitor():
    st.header("🌫️ Real-Time Air Quality Monitoring")
    
    # Initialize session state
    if 'monitor' not in st.session_state:
        st.session_state.monitor = AirQualityMonitor()
        st.session_state.monitoring = False
        st.session_state.last_update = None

    # Monitoring controls
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("🔑 AirNow API Key", type="password", key="monitor_api_key")
    
    location_type = st.radio("Location Input Method", ['ZIP Code', 'Latitude/Longitude'], horizontal=True)
    
    if location_type == 'ZIP Code':
        location_input = st.text_input("📌 ZIP Code", "90001", key="monitor_zip_code")
        location = location_input
        location_type_code = 'zip'
    else:
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=34.05, format="%.6f", key="monitor_lat")
        with col2:
            lon = st.number_input("Longitude", value=-118.25, format="%.6f", key="monitor_lon")
        location = (lat, lon)
        location_type_code = 'latlon'
    
    interval = st.slider("⏱️ Update Interval (minutes)", 1, 60, 5, key="monitor_interval")
    
    if st.button("🚀 Start Monitoring", key="start_monitoring") and not st.session_state.monitoring:
        if api_key and location:
            st.session_state.monitor = AirQualityMonitor()
            st.session_state.monitor.location = location
            st.session_state.monitor.location_type = location_type_code
            st.session_state.monitor.monitoring_active = True
            st.session_state.monitor.start_time = datetime.now()
            st.session_state.monitoring = True
            
            monitor_thread = threading.Thread(
                target=st.session_state.monitor.collect_data,
                args=(api_key, location, location_type_code, interval),
                daemon=True
            )
            monitor_thread.start()
            st.success("✅ Monitoring started!")
        else:
            st.warning("⚠️ Please enter both API key and location information")
            
    if st.button("🛑 Stop Monitoring", key="stop_monitoring") and st.session_state.monitoring:
        duration = st.session_state.monitor.stop_monitoring()
        st.session_state.monitoring = False
        st.success(f"📊 Monitoring stopped after {duration}")

    # Live Data Display
    if st.session_state.monitoring:
        if st.session_state.monitor.location_type == 'zip':
            location_header = f"ZIP: {st.session_state.monitor.location}"
        else:
            lat, lon = st.session_state.monitor.location
            location_header = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        
        st.header(f"📡 Live Monitoring: {location_header}")
        
        data_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        while st.session_state.monitoring:
            air_data, wind_data = st.session_state.monitor.get_latest_data()
            
            with data_placeholder.container():
                st.subheader("📈 Live Data Stream")
                if air_data:
                    df = pd.DataFrame(air_data)
                    st.dataframe(
                        df.sort_values('timestamp', ascending=False),
                        height=min(300, 35 * len(df))
                    )
                    st.caption(f"Showing {len(df)} reading(s)")
                else:
                    st.info("Collecting initial data...")
            
            with metrics_placeholder.container():
                st.subheader("🔍 Current Readings")
                cols = st.columns(len(POLLUTANTS))
                for i, (poll, info) in enumerate(POLLUTANTS.items()):
                    with cols[i]:
                        poll_data = [d for d in air_data if d['ParameterName'] == poll]
                        if poll_data:
                            latest = poll_data[-1]
                            st.metric(
                                label=poll,
                                value=f"{latest['AQI']} {info['unit']}",
                                help=f"Threshold: {info['thresholds'][1]} {info['unit']}"
                            )
                        else:
                            st.metric(label=poll, value="N/A")
            
            time.sleep(1)
            
            if not st.session_state.monitoring:
                break

    # Report Generation - This will now show all components regardless of location type
    elif not st.session_state.monitoring and st.session_state.monitor.all_data:
        display_reports()

def display_reports():
    # Get the appropriate location display string
    if st.session_state.monitor.location_type == 'zip':
        location_str = f"ZIP: {st.session_state.monitor.location}"
    else:
        lat, lon = st.session_state.monitor.location
        location_str = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
    
    st.header(f"📊 Final Report for {location_str}")
    air_data, wind_data = st.session_state.monitor.get_latest_data()
    
    tab1, tab2, tab3 = st.tabs(["📉 Trends", "⚠️ Violations", "🌬️ Wind Analysis"])
    
    with tab1:
        st.subheader("🕰️ Air Quality Trends")
        df = pd.DataFrame(air_data)
        if not df.empty:
            fig = px.line(df, x='timestamp', y='AQI', color='ParameterName',
                         color_discrete_map={k: v['color'] for k, v in POLLUTANTS.items()},
                         labels={'AQI': 'AQI Value', 'timestamp': 'Time'},
                         title=f"Air Quality Trends for {location_str}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No air quality data available")
    
    with tab2:
        st.subheader("🚨 Threshold Violations")
        violations = {}
        for poll in POLLUTANTS:
            subset = [d for d in air_data if d['ParameterName'] == poll]
            if subset:
                violations[poll] = (sum(1 for d in subset 
                                     if d['AQI'] > POLLUTANTS[poll]['thresholds'][1]) / len(subset)) * 100
        
        if violations:
            fig = px.bar(x=list(violations.keys()), y=list(violations.values()),
                         color=list(violations.keys()),
                         color_discrete_map={k: v['color'] for k, v in POLLUTANTS.items()},
                         labels={'x': 'Pollutant', 'y': 'Violation %'},
                         title="% Time Above Safety Thresholds")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No violation data available")
    
    with tab3:
        st.subheader("🧭 Wind Direction Analysis")
        wind_plot = st.session_state.monitor.generate_wind_plot()
        if wind_plot:
            st.image(wind_plot)
            if wind_data:
                wind_df = pd.DataFrame(wind_data)
                st.dataframe(wind_df)
        else:
            st.warning("No wind data available in the API response")
    
    st.subheader("💾 Download Options")
    df = pd.DataFrame(air_data)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"air_quality_{location_str.replace(':', '').replace(', ', '_')}.csv",
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="📥 Download JSON",
            data=df.to_json(indent=2),
            file_name=f"air_quality_{location_str.replace(':', '').replace(', ', '_')}.json",
            mime='application/json'
        )
    
    st.subheader("🔔 Pollutant Alerts Summary")
    for poll, info in POLLUTANTS.items():
        subset = [d for d in air_data if d['ParameterName'] == poll]
        if subset:
            max_aqi = max(d['AQI'] for d in subset)
            threshold_index = next((i for i, thresh in enumerate(info['thresholds']) 
                                 if max_aqi <= thresh), len(info['thresholds']))
            alert_levels = ["✅ Good", "⚠️ Moderate", "🚸 Unhealthy for Sensitive Groups", 
                          "❗ Unhealthy", "❌ Very Unhealthy"]
            st.write(f"{poll}: Max AQI {max_aqi} - {alert_levels[threshold_index]}")
# ========================================
# HOME PAGE
# ========================================
def show_home():
    st.markdown('<h3 style="text-align:center;font-weight:900;color:#3a0ca3;">AirSage</h3>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size:1.3rem;color:#555;text-align:center;">
            Machine learning and AI‑powered Pollution Predictor,<br/>
            for cleaner air and healthier communities
        </p>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown(
            """
            <div class="elegant-card">
                <h3>🌬️ Smarter Air Quality Monitoring</h3>
                <p>
                    Our dual-mode system provides both historical analysis and real-time monitoring capabilities.
                    <strong>Historical Mode</strong> analyzes past data to reveal trends and predict future conditions,
                    while <strong>Live Mode</strong> gives you up-to-the-minute air quality insights.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
        """
            <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <img src="data:image/jpeg;base64,{img}" style="width: 220px; margin-top: 30px; border-radius: 12px;" />
            </div>
            """.format(img=base64.b64encode(open(r"C:\\Users\\SNEHA\\Documents\\Python\\WhatsApp Image 2025-07-08 at 13.10.59.jpeg", "rb").read()).decode()),
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="text-align:center;margin:40px 0 15px;font-size:1.8rem;font-weight:800;color:#3a0ca3;">
            🌎 Breathe Better, Live Better
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Capabilities - Split into two sections
        # ... (keep the top part with title/logo the same) ...

    st.subheader("✨ Key Capabilities")

    # Historical Data Analysis Section
    st.markdown("##### 🗃️ Historical Data Analysis")
    historical_features = [
        ("📈 Trend Detection", "Live and interactive air quality visualizations powered by user-uploaded datasets. Helps users monitor critical pollutant levels in a dynamic environment.Identifies seasonal patterns and underlying pollution sources through feature importance and distribution visualizations, enabling better decision-making."),
        ("🔮 Future Forecasting", "Predicts future concentrations like PM2.5, NO2, and O₃ with high accuracy up to 48 hours in advance, using machine learning models trained on historical pollution data."),
        ("📊 Feature Importance", "Discover which factors most impact air quality and generates smart alerts when pollutant levels are predicted to cross safety thresholds, helping users take timely preventive measures.")
    ]
    for title, desc in historical_features:
        st.markdown(
            f"""
            <div class="elegant-card" style="border-left:4px solid #4361ee;">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Real-Time Analysis Section  
    st.markdown("##### ⚡ Real-Time Monitoring")
    realtime_features = [
        ("🌡️ Live Measurements", "Continuous second-by-second monitoring of current conditions with live data collection visualization."),
        ("🚨 Instant Alerts", "Get alerts immediately when pollutants exceed safe levels."),
        ("🌪️ Wind Analysis", "Visualize how wind patterns affect pollution distribution with graph analysis.")
    ]
    for title, desc in realtime_features:
        st.markdown(
            f"""
            <div class="elegant-card" style="border-left:4px solid #3a0ca3;">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # How It Works - Separate workflows
    st.subheader("🔧 How It Works")
    
    st.markdown("#### For Historical Data Analysis")
    st.markdown(
        """
        <div class="elegant-card">
            <div style="display:flex;justify-content:space-between;flex-wrap:wrap;">
                <div style="width:32%;margin-bottom:15px;text-align:center;">
                    <h4>1. Upload Data</h4>
                    <p style="font-size:3rem;">📤</p>
                    <p>Import your CSV with historical air quality measurements</p>
                </div>
                <div style="width:32%;margin-bottom:15px;text-align:center;">
                    <h4>2. Select Target</h4>
                    <p style="font-size:3rem;">⚙️</p>
                    <p>Choose which pollutant to analyze (PM2.5, O₃, etc.)</p>
                </div>
                <div style="width:32%;margin-bottom:15px;text-align:center;">
                    <h4>3. Get Insights</h4>
                    <p style="font-size:3rem;">📊</p>
                    <p>Receive predictions, trends, and health recommendations</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("#### For Real-Time Monitoring")
    st.markdown(
        """
        <div class="elegant-card">
            <div style="display:flex;justify-content:space-between;flex-wrap:wrap;">
                <div style="width:24%;margin-bottom:15px;text-align:center;">
                    <h4>1. Enter API Key</h4>
                    <p style="font-size:3rem;">🔑</p>
                    <p>Provide your AirNow API credentials</p>
                </div>
                <div style="width:24%;margin-bottom:15px;text-align:center;">
                    <h4>2. Set Location</h4>
                    <p style="font-size:3rem;">📍</p>
                    <p>Enter ZIP code for monitoring</p>
                </div>
                <div style="width:24%;margin-bottom:15px;text-align:center;">
                    <h4>3. Configure</h4>
                    <p style="font-size:3rem;">⏱️</p>
                    <p>Set update frequency (1-60 minutes)</p>
                </div>
                <div style="width:24%;margin-bottom:15px;text-align:center;">
                    <h4>4. Monitor</h4>
                    <p style="font-size:3rem;">📡</p>
                    <p>View live data stream and alerts</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================================
# ABOUT PAGE
# ========================================
def show_about():
    st.markdown('<h3 style="text-align:center;font-weight:900;color:#3a0ca3;">About Our Mission</h3>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="elegant-card">
            <h3>🌱 Why Clean Air Matters</h3>
            <p>
                Air pollution leads to over 7 million premature deaths annually according to WHO.
                Our platform helps predict and prevent such outcomes by equipping planners and researchers with AI-powered insights to improve air quality
                before it becomes harmful.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("👥 The Team Behind the Technology")
    team = [
        {"name": "Sarbatriki Jana", "role": "AI-ML & Data Science Enthusiast", "bio": "Tech Lead - AI modelling, Prompt Engineering . Pursuing B.Tech in ECE,3rd year."},
        {"name": "Saraddyuti Chakrabarty", "role": "Data Science Enthusiast", "bio": "Data Preprocessing, Prompt Engineering. Pursuing B.Tech in ECE,3rd year."},
    ]
    cols = st.columns(2)
    for idx, member in enumerate(team):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="elegant-card" style="text-align:center;">
                    <div style="font-size:3rem;">👩‍💻</div>
                    <h4>{member['name']}</h4>
                    <p style="color:#4361ee;font-weight:600;">{member['role']}</p>
                    <p style="font-size:.9rem;color:#666;">{member['bio']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("🛠️ Technology Stack")
    st.markdown(
        """
        <div class="elegant-card">
            <div style="display:flex;justify-content:space-between;flex-wrap:wrap;">
                <div style="width:48%;margin-bottom:15px;">
                    <h4>Core Framework</h4>
                    <ul>
                        <li>Python 3.10</li>
                        <li>Streamlit 1.23</li>
                        <li>Pandas 2.0</li>
                        <li>Plotly 5.15</li>
                    </ul>
                </div>
                <div style="width:48%;">
                    <h4>Machine Learning</h4>
                    <ul>
                        <li>Random Forest Regression</li>
                        <li>Feature‑importance Analysis</li>
                        <li>Hyper‑parameter Optimisation</li>
                        <li>SHAP Value Explanations</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================================
# PREDICT PAGE
# ========================================
def show_predict():
    st.header("📊 Air Quality Predictor")

    uploaded_file = st.file_uploader("Upload your air‑quality data (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            @st.cache_data
            def load_data(file):
                try:
                    df = pd.read_csv(file)
                except:
                    try:
                        df = pd.read_csv(file, encoding='latin1')
                    except:
                        df = pd.read_csv(file, encoding='utf-8')
                
                df.columns = df.columns.str.strip()
                df = df.loc[:, ~df.columns.str.contains("^Unnamed\d+$")]
                
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                    except:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            pass
                
                df.dropna(axis=1, how='all', inplace=True)
                df.dropna(how='all', inplace=True)
                
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    try:
                        df['datetime'] = pd.to_datetime(df[date_cols[0]])
                        df = df.sort_values('datetime')
                    except:
                        pass
                return df

            df = load_data(uploaded_file)

            with st.expander("🔍 Data Preview", expanded=True):
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape}")
                st.write("Columns:", df.columns.tolist())

            def numeric_predictable(col):
                return df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 10

            targets = [c for c in df.columns if numeric_predictable(c)]
            if not targets:
                st.error("No suitable numeric columns found for prediction 🤷‍♂️")
                return

            target_col = st.selectbox("Select target variable to predict:", targets)

            st.subheader("📈 Pollution Trend Analysis")
            
            if 'datetime' in df.columns:
                st.markdown("### Time Series Trend")
                window_size = st.slider("Smoothing window size:", 1, 30, 7, 
                                      help="Larger values create smoother trend lines")
                
                df['smoothed'] = df[target_col].rolling(window=window_size, center=True).mean()
                
                fig = px.line(df, x='datetime', y=[target_col, 'smoothed'],
                             title=f"{target_col} Trend Over Time",
                             labels={'value': target_col, 'datetime': 'Date'},
                             template='plotly_white')
                
                fig.add_hline(y=50, line_dash="dot", line_color="green", 
                             annotation_text="Good", annotation_position="bottom right")
                fig.add_hline(y=100, line_dash="dot", line_color="yellow", 
                             annotation_text="Moderate", annotation_position="bottom right")
                fig.add_hline(y=200, line_dash="dot", line_color="red", 
                             annotation_text="Unhealthy", annotation_position="bottom right")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Seasonal Patterns")
                try:
                    ts = df.set_index('datetime')[target_col].interpolate().dropna()
                    ts = ts.asfreq('D').fillna(method='ffill')
                    
                    if len(ts) > 30:
                        decomposition = seasonal_decompose(ts, period=30)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                                              name='Trend', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                                              name='Seasonal', line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                                              name='Residual', line=dict(color='red')))
                        fig.update_layout(title='Time Series Decomposition',
                                        xaxis_title='Date',
                                        yaxis_title=target_col)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **How to interpret this:**
                        - **Trend**: Shows the long-term direction of pollution levels
                        - **Seasonal**: Reveals repeating patterns (daily/weekly/monthly cycles)
                        - **Residual**: Random fluctuations not explained by trend or seasonality
                        """)
                except Exception as e:
                    st.warning(f"Seasonal decomposition couldn't be performed: {str(e)}")
            else:
                st.warning("No datetime column found - trend analysis requires a date/time column")

            st.subheader("🤖 Model Training")
            if st.button("Train Prediction Model", type="primary"):
                with st.spinner("Training AI model..."):
                    try:
                        exclude_cols = [target_col]
                        if 'datetime' in df.columns:
                            exclude_cols.append('datetime')
                        
                        X = df.drop(columns=exclude_cols)
                        
                        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_cols) == 0:
                            st.error("No numeric columns available for modeling. Please check your data.")
                            return
                        
                        X = X[numeric_cols]
                        y = df[target_col]

                        mask = ~y.isna()
                        X, y = X[mask], y[mask]

                        X = SimpleImputer(strategy="mean").fit_transform(X)
                        X = StandardScaler().fit_transform(X)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        model = RandomForestRegressor(
                            n_estimators=200, max_depth=12, random_state=42
                        ).fit(X_train, y_train)

                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        st.success("Model trained successfully!")
                        c1, c2 = st.columns(2)
                        c1.metric("R² Score", f"{r2:.4f}", 
                                help="Higher is better (1.0 is perfect)")
                        c2.metric("Mean Squared Error", f"{mse:.2f}", 
                                help="Lower is better")

                        feat_imp = (
                            pd.DataFrame(
                                {"Feature": numeric_cols,
                                 "Importance": model.feature_importances_}
                            )
                            .sort_values("Importance", ascending=False)
                        )
                        st.plotly_chart(
                            px.bar(feat_imp, x="Importance", y="Feature", 
                                  title="Feature Importance (What drives pollution?)"),
                            use_container_width=True
                        )

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=y_test, y=y_pred, mode="markers",
                                name="Predictions", marker=dict(color="#4361ee")
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode="lines", name="Perfect Prediction",
                                line=dict(color="red", dash="dash"),
                            )
                        )
                        fig.update_layout(
                            title="Actual vs Predicted Values",
                            xaxis_title="Actual", yaxis_title="Predicted"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        forecast = model.predict([X[-1]])[0]
                        if forecast > 200:
                            alert, color = "🔴 Unhealthy Air Quality Expected!", "#ff4d4d"
                        elif forecast > 100:
                            alert, color = "🟠 Moderate Air Quality Expected", "#ff9966"
                        else:
                            alert, color = "🟢 Good Air Quality Expected", "#66cc99"

                        st.markdown(
                            f"""
                            <div style="background:{color}20;padding:15px;border-radius:10px;border-left:4px solid {color};">
                                <h3>Next {target_col} Prediction: <strong>{forecast:.2f}</strong></h3>
                                <p style="font-size:1.2rem;">{alert}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except ValueError as e:
                        if "could not convert string to float" in str(e):
                            st.error("Model training failed: Your data contains non-numeric values that can't be processed. Please remove or encode text columns.")
                        else:
                            st.error(f"Model training failed: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# ========================================
# MAIN APP
# ========================================
def main():
    inject_custom_css()

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:30px;">
                <h1 style="color:white;font-size:2.5rem;">AirSage</h1>
                <p style="color:#d1d5db;">Predictive Air Analytics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigate", ["🏠 Home", "🌫️ Monitor", "📊 Predict", "ℹ️ About"], label_visibility="collapsed")

    if page == "🏠 Home":
        show_home()
    elif page == "ℹ️ About":
        show_about()
    elif page == "📊 Predict":
        show_predict()
    else:
        show_monitor()

if __name__ == "__main__":
    main()