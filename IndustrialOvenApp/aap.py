import streamlit as st
from utils.weather import get_weather
from utils.predictor import predict_heating_time
from utils.trainer import prepare_training_data, create_features, train_model
from utils.analyzer import analyze_all_files
import glob

st.set_page_config(page_title="Industrial Oven Dashboard", layout="wide")
st.title("🔥 Industrial Oven Heat-Up Dashboard")

# Dropdown for mode selection
mode = st.selectbox("Select Mode", [
    "Heat-up Time (All Zones)",
    "Zone-wise Heat-up Details",
    "Train Model",
    "Analyze Historical Data"
])

# Mode 1: All Zones Prediction
if mode == "Heat-up Time (All Zones)":
    st.subheader("🔥 Predict Heating Time for All Zones")
    current_temp = st.number_input("Enter Current Oven Temperature (°C)", value=30.0)
    for sensor in ['WU311', 'WU312', 'WU314', 'WU321', 'WU322', 'WU323']:
        try:
            time = predict_heating_time(sensor, current_temp)
            st.write(f"**{sensor}** → {time + 10:.1f} minutes")
        except Exception as e:
            st.error(f"{sensor} prediction failed: {str(e)}")

# Mode 2: Zone-wise Details
elif mode == "Zone-wise Heat-up Details":
    st.subheader("🔍 Zone-wise Prediction")
    sensor = st.selectbox("Select Sensor", ['WU311', 'WU312', 'WU314', 'WU321', 'WU322', 'WU323'])
    current_temp = st.number_input("Enter Current Oven Temperature (°C)", value=30.0)
    try:
        time = predict_heating_time(sensor, current_temp)
        weather = get_weather()
        st.metric("Predicted Heating Time", f"{time + 10:.1f} minutes")
        st.write(f"Weather: {weather['temp']}°C, {weather['humidity']}% humidity, {weather['conditions']}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Mode 3: Train Model
elif mode == "Train Model":
    st.subheader("📊 Train ML Model")
    if st.button("Start Training"):
        csv_files = glob.glob("data/Research Data CED OVEN/*.CSV")
        oven_data = prepare_training_data(csv_files)
        features = create_features(oven_data)
        if not features.empty:
            model, feature_names = train_model(features)
            st.success("Model trained and saved successfully!")
        else:
            st.error("No valid training data found.")

# Mode 4: Analyze Historical Data
elif mode == "Analyze Historical Data":
    st.subheader("📈 Historical Heating Curve Analysis")
    if st.button("Run Analysis"):
        try:
            report_path, summary_path = analyze_all_files()
            st.success("Analysis complete!")
            st.write(f"📄 Report saved at: `{report_path}`")
            st.write(f"📄 Summary saved at: `{summary_path}`")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
