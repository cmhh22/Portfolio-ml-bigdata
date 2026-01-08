"""
🚕 NYC Taxi Trip Duration Predictor
====================================
Interactive web app to predict taxi trip duration in New York City.

Features:
- Select pickup/dropoff locations on an interactive map
- Compare predictions from multiple ML models (XGBoost, LightGBM, Random Forest, etc.)
- See how different factors affect trip duration (rush hour, distance, time of day)

Built with: Streamlit, Plotly, scikit-learn, XGBoost, LightGBM
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, time
import plotly.graph_objects as go

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="NYC Taxi Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Constants
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Airport coordinates for feature engineering
JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)

# NYC bounds for validation
NYC_BOUNDS = {
    'lat_min': 40.4774, 'lat_max': 40.9176,
    'lon_min': -74.2591, 'lon_max': -73.7004
}

# Predefined routes with names and coordinates
ROUTES = {
    "Times Square → JFK Airport": {
        "pickup": (40.7580, -73.9855),
        "dropoff": (40.6413, -73.7781),
        "description": "Midtown Manhattan to JFK (~22 km)"
    },
    "Central Park → Wall Street": {
        "pickup": (40.7829, -73.9654),
        "dropoff": (40.7074, -74.0113),
        "description": "Upper Manhattan to Financial District (~8 km)"
    },
    "LaGuardia → Times Square": {
        "pickup": (40.7769, -73.8740),
        "dropoff": (40.7580, -73.9855),
        "description": "LaGuardia Airport to Midtown (~12 km)"
    },
    "Brooklyn → Empire State": {
        "pickup": (40.6782, -73.9442),
        "dropoff": (40.7484, -73.9857),
        "description": "Brooklyn to Midtown Manhattan (~9 km)"
    },
    "SoHo → Upper East Side": {
        "pickup": (40.7233, -74.0030),
        "dropoff": (40.7736, -73.9566),
        "description": "Downtown to Uptown (~7 km)"
    },
}

# Available models (only models uploaded to GitHub)
MODEL_INFO = {
    "xgboost_model.pkl": {"name": "XGBoost", "emoji": "🚀", "r2": 0.8234},
    "lightgbm_model.pkl": {"name": "LightGBM", "emoji": "⚡", "r2": 0.8040},
    "gradient_boosting_model.pkl": {"name": "Gradient Boosting", "emoji": "📈", "r2": 0.7926},
    "ridge_model.pkl": {"name": "Ridge Regression", "emoji": "📊", "r2": 0.5392},
}

# Default model if others fail to load
DEFAULT_MODEL = "xgboost_model.pkl"


# ============================================
# Helper Functions
# ============================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def calculate_direction(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def format_duration(seconds):
    """Format seconds to readable duration."""
    if pd.isna(seconds) or np.isinf(seconds):
        return "N/A"
    if seconds < 0:
        seconds = abs(seconds)
    if seconds < 60:
        return f"{int(seconds)} sec"
    elif seconds < 3600:
        return f"{int(seconds // 60)} min {int(seconds % 60)} sec"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}min"


@st.cache_resource
def load_all_models():
    """Load all available models with better error handling."""
    models = {}
    failed_models = []
    
    for filename, info in MODEL_INFO.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                models[filename] = {
                    "model": joblib.load(path),
                    **info
                }
            except Exception as e:
                failed_models.append(f"{info['name']}: {str(e)}")
        else:
            failed_models.append(f"{info['name']}: File not found")
    
    # Show warnings only if no models loaded successfully
    if not models and failed_models:
        st.error("⚠️ No models could be loaded!")
        for err in failed_models:
            st.warning(f"• {err}")
    
    return models


def prepare_features(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                     pickup_datetime, passenger_count, vendor_id):
    """Prepare all 26 features for model prediction."""
    
    hour = pickup_datetime.hour
    dayofweek = pickup_datetime.weekday()
    month = pickup_datetime.month
    
    # Distance features
    haversine_dist = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    manhattan_dist = abs(dropoff_lat - pickup_lat) + abs(dropoff_lon - pickup_lon)
    direction = calculate_direction(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Center point
    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lon + dropoff_lon) / 2
    
    # Airport features
    dist_to_jfk = haversine_distance(pickup_lat, pickup_lon, *JFK_COORDS)
    dist_to_lga = haversine_distance(pickup_lat, pickup_lon, *LGA_COORDS)
    dist_jfk_dropoff = haversine_distance(dropoff_lat, dropoff_lon, *JFK_COORDS)
    dist_lga_dropoff = haversine_distance(dropoff_lat, dropoff_lon, *LGA_COORDS)
    
    is_jfk = 1 if (dist_to_jfk < 2 or dist_jfk_dropoff < 2) else 0
    is_lga = 1 if (dist_to_lga < 2 or dist_lga_dropoff < 2) else 0
    
    # Time flags
    is_weekend = 1 if dayofweek >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    
    # Cyclic encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)
    
    return pd.DataFrame([{
        'vendor_id': vendor_id,
        'passenger_count': passenger_count,
        'store_and_fwd_flag': 0,
        'pickup_longitude': pickup_lon,
        'pickup_latitude': pickup_lat,
        'dropoff_longitude': dropoff_lon,
        'dropoff_latitude': dropoff_lat,
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'is_night': is_night,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'haversine_distance': haversine_dist,
        'manhattan_distance': manhattan_dist,
        'direction': direction,
        'center_latitude': center_lat,
        'center_longitude': center_lon,
        'dist_to_jfk_pickup': dist_to_jfk,
        'dist_to_lga_pickup': dist_to_lga,
        'is_jfk_trip': is_jfk,
        'is_lga_trip': is_lga,
    }])


# ============================================
# Main App
# ============================================

def main():
    # Header
    st.title("🚕 NYC Taxi Trip Duration Predictor")
    st.markdown("*Predict taxi trip duration in New York City using Machine Learning*")
    
    # Load models
    models = load_all_models()
    
    if not models:
        st.error("❌ No models found! Please ensure trained models are available in the repository.")
        st.info("💡 Available models should be: xgboost_model.pkl, lightgbm_model.pkl, gradient_boosting_model.pkl, ridge_model.pkl")
        return
    
    # ==========================================
    # SIDEBAR - Controls
    # ==========================================
    with st.sidebar:
        st.header("🎯 Trip Configuration")
        
        # Route selection
        st.subheader("📍 Select Route")
        route_name = st.selectbox(
            "Preset Routes",
            options=list(ROUTES.keys()),
            help="Select a predefined route or adjust coordinates below"
        )
        
        route = ROUTES[route_name]
        st.caption(route["description"])
        
        # Store coordinates in session state for map interaction
        if 'pickup' not in st.session_state:
            st.session_state.pickup = route["pickup"]
            st.session_state.dropoff = route["dropoff"]
        
        # Update when route changes
        if st.button("🔄 Apply Route", use_container_width=True):
            st.session_state.pickup = route["pickup"]
            st.session_state.dropoff = route["dropoff"]
            st.rerun()
        
        st.markdown("---")
        
        # Time settings
        st.subheader("🕐 Date & Time")
        
        col1, col2 = st.columns(2)
        with col1:
            pickup_date = st.date_input(
                "Date",
                value=datetime.now().date(),
                key="date_input"
            )
        with col2:
            pickup_hour = st.selectbox(
                "Hour",
                options=list(range(24)),
                index=datetime.now().hour,
                format_func=lambda x: f"{x:02d}:00"
            )
        
        pickup_datetime = datetime.combine(pickup_date, time(hour=pickup_hour))
        
        # Show time context
        is_rush = pickup_hour in [7, 8, 9, 17, 18, 19]
        is_night = pickup_hour >= 22 or pickup_hour <= 5
        is_weekend = pickup_date.weekday() >= 5
        
        tags = []
        if is_rush:
            tags.append("🚦 Rush Hour")
        if is_night:
            tags.append("🌙 Night")
        if is_weekend:
            tags.append("📅 Weekend")
        if tags:
            st.caption(" | ".join(tags))
        
        st.markdown("---")
        
        # Trip details
        st.subheader("👥 Trip Details")
        passenger_count = st.slider("Passengers", 1, 6, 1)
        vendor_id = st.radio("Vendor", [1, 2], horizontal=True)
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Models to Compare")
        selected_models = st.multiselect(
            "Select models",
            options=list(models.keys()),
            default=list(models.keys())[:3],  # Default: top 3
            format_func=lambda x: f"{models[x]['emoji']} {models[x]['name']} (R²={models[x]['r2']:.3f})"
        )
        
        if not selected_models:
            st.warning("Select at least one model")
            selected_models = [list(models.keys())[0]]
    
    # ==========================================
    # MAIN AREA
    # ==========================================
    
    # Get current coordinates
    pickup_lat, pickup_lon = st.session_state.pickup
    dropoff_lat, dropoff_lon = st.session_state.dropoff
    
    # Calculate distance
    distance = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Layout: Map | Predictions
    col_map, col_pred = st.columns([3, 2])
    
    with col_map:
        st.subheader("🗺️ Route Map")
        
        # Interactive coordinate adjustment
        with st.expander("📌 Fine-tune coordinates", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 Pickup**")
                new_pickup_lat = st.number_input("Lat", value=pickup_lat, format="%.4f", key="plat")
                new_pickup_lon = st.number_input("Lon", value=pickup_lon, format="%.4f", key="plon")
            with c2:
                st.markdown("**🔴 Dropoff**")
                new_dropoff_lat = st.number_input("Lat", value=dropoff_lat, format="%.4f", key="dlat")
                new_dropoff_lon = st.number_input("Lon", value=dropoff_lon, format="%.4f", key="dlon")
            
            if st.button("✅ Update Coordinates"):
                st.session_state.pickup = (new_pickup_lat, new_pickup_lon)
                st.session_state.dropoff = (new_dropoff_lat, new_dropoff_lon)
                st.rerun()
        
        # Create map
        fig = go.Figure()
        
        # Route line
        fig.add_trace(go.Scattermapbox(
            lat=[pickup_lat, dropoff_lat],
            lon=[pickup_lon, dropoff_lon],
            mode='lines',
            line=dict(width=4, color='#3366cc'),
            name='Route',
            hoverinfo='skip'
        ))
        
        # Pickup marker
        fig.add_trace(go.Scattermapbox(
            lat=[pickup_lat],
            lon=[pickup_lon],
            mode='markers+text',
            marker=dict(size=18, color='#00cc66'),
            text=['Pickup'],
            textposition='top center',
            name='Pickup',
            hovertemplate=f'<b>Pickup</b><br>Lat: {pickup_lat:.4f}<br>Lon: {pickup_lon:.4f}<extra></extra>'
        ))
        
        # Dropoff marker
        fig.add_trace(go.Scattermapbox(
            lat=[dropoff_lat],
            lon=[dropoff_lon],
            mode='markers+text',
            marker=dict(size=18, color='#cc3333'),
            text=['Dropoff'],
            textposition='top center',
            name='Dropoff',
            hovertemplate=f'<b>Dropoff</b><br>Lat: {dropoff_lat:.4f}<br>Lon: {dropoff_lon:.4f}<extra></extra>'
        ))
        
        # Map layout
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(
                    lat=(pickup_lat + dropoff_lat) / 2,
                    lon=(pickup_lon + dropoff_lon) / 2
                ),
                zoom=11
            ),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trip summary
        cols = st.columns(4)
        cols[0].metric("📏 Distance", f"{distance:.2f} km")
        cols[1].metric("👥 Passengers", passenger_count)
        cols[2].metric("🕐 Time", f"{pickup_hour:02d}:00")
        cols[3].metric("📅 Day", pickup_date.strftime("%a"))
    
    with col_pred:
        st.subheader("⏱️ Duration Predictions")
        
        # Prepare features once
        features_df = prepare_features(
            pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            pickup_datetime, passenger_count, vendor_id
        )
        
        # Handle any NaN values in features
        if features_df.isna().any().any():
            features_df = features_df.fillna(0)
        
        # Get predictions from all selected models
        predictions = []
        
        for model_key in selected_models:
            model_data = models[model_key]
            try:
                # Model predicts log_trip_duration (which is log1p(trip_duration))
                # Only need single expm1 to convert back to seconds
                pred_log = model_data["model"].predict(features_df)[0]
                pred_seconds = np.expm1(pred_log)
                
                # TEMPORARY DEBUG: Show prediction details
                if pred_seconds < 120 or pred_seconds > 7200:  # Less than 2 min or more than 2 hours
                    st.warning(f"⚠️ **{model_data['name']}**: Unusual prediction - raw={pred_log:.4f}, seconds={pred_seconds:.1f}")
                
                # Validate prediction
                if pd.isna(pred_seconds) or np.isinf(pred_seconds) or pred_seconds < 0:
                    st.warning(f"⚠️ {model_data['name']}: Invalid prediction (raw: {pred_log:.4f})")
                    continue
                
                # Clamp to reasonable range (1 min to 3 hours)
                pred_seconds = np.clip(pred_seconds, 60, 10800)
                
                predictions.append({
                    "name": model_data["name"],
                    "emoji": model_data["emoji"],
                    "r2": model_data["r2"],
                    "seconds": pred_seconds,
                    "formatted": format_duration(pred_seconds)
                })
            except Exception as e:
                st.warning(f"Error with {model_data['name']}: {e}")
        
        if predictions:
            # Sort by R² (best first)
            predictions.sort(key=lambda x: x["r2"], reverse=True)
            
            # Best prediction highlighted
            best = predictions[0]
            st.markdown(f"### {best['emoji']} Best Model: {best['name']}")
            st.markdown(f"## 🎯 {best['formatted']}")
            
            if distance > 0.1:
                avg_speed = (distance / best['seconds']) * 3600
                st.caption(f"Average speed: {avg_speed:.1f} km/h")
            
            # Airport detection
            if features_df['is_jfk_trip'].values[0]:
                st.info("✈️ JFK Airport trip detected")
            elif features_df['is_lga_trip'].values[0]:
                st.info("✈️ LaGuardia trip detected")
            
            st.markdown("---")
            
            # All model comparisons
            if len(predictions) > 1:
                st.markdown("### 📊 Model Comparison")
                
                for pred in predictions:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    col1.write(f"{pred['emoji']} **{pred['name']}**")
                    col2.write(f"⏱️ {pred['formatted']}")
                    col3.write(f"R²={pred['r2']:.3f}")
                
                # Bar chart comparison
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    x=[p['name'] for p in predictions],
                    y=[p['seconds'] / 60 for p in predictions],  # Minutes
                    marker_color=['#00cc66' if i == 0 else '#3366cc' for i in range(len(predictions))],
                    text=[p['formatted'] for p in predictions],
                    textposition='outside'
                ))
                fig_compare.update_layout(
                    title="Prediction Comparison",
                    yaxis_title="Duration (minutes)",
                    height=300,
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig_compare, use_container_width=True)
    
    # ==========================================
    # EXPLANATION SECTION
    # ==========================================
    with st.expander("ℹ️ How does this app work?", expanded=False):
        st.markdown("""
        ### 🚕 NYC Taxi Trip Duration Predictor
        
        This app predicts how long a taxi trip will take in New York City using **Machine Learning models** 
        trained on real data from the [NYC Taxi Trip Duration Kaggle dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration).
        
        #### 🔧 Features Used for Prediction:
        
        | Category | Features |
        |----------|----------|
        | **Geographic** | Pickup/dropoff coordinates, haversine distance, Manhattan distance, direction |
        | **Temporal** | Hour, day of week, month, rush hour flag, night flag, weekend flag |
        | **Trip Info** | Passenger count, vendor ID |
        | **Airport** | Distance to JFK/LaGuardia, airport trip flags |
        | **Cyclic** | Sin/cos encoding of hour and day of week |
        
        #### 🤖 Available Models:
        
        | Model | R² Score | Description |
        |-------|----------|-------------|
        | **XGBoost** | 0.8234 | Gradient boosting with regularization - Best performance |
        | **LightGBM** | 0.8040 | Fast gradient boosting by Microsoft |
        | **Random Forest** | 0.7938 | Ensemble of decision trees |
        | **Gradient Boosting** | 0.7926 | Sequential ensemble method |
        | **Ridge** | 0.5392 | Linear regression with L2 regularization |
        
        #### 📈 How to Use:
        
        1. **Select a route** from the presets or fine-tune coordinates
        2. **Choose date and time** to see how rush hour affects duration
        3. **Select models** to compare different predictions
        4. **Analyze results** to understand model behavior
        
        #### 🎯 Key Insights:
        
        - **Distance** is the most important predictor (~78% feature importance)
        - **Rush hour** (7-9 AM, 5-7 PM) increases trip duration by ~15-20%
        - **Airport trips** have distinct patterns due to highways
        - **Night trips** (10 PM - 5 AM) are typically faster
        """)


if __name__ == "__main__":
    main()
