import pandas as pd
import joblib
from utils.weather import get_weather

SENSOR_TARGETS = {
    'WU311': 160, 'WU312': 190, 'WU314': 190,
    'WU321': 190, 'WU322': 190, 'WU323': 190
}

def predict_heating_time(sensor, current_temp, model_path="oven_time_predictor.pkl"):
    model, features = joblib.load(model_path)
    weather = get_weather()

    input_data = pd.DataFrame({
        'start_temp': [current_temp],
        'ambient_temp': [weather['temp']],
        'humidity': [weather['humidity']],
        'target_temp': [SENSOR_TARGETS[sensor]],
        **{f'sensor_{s}': [1 if s == sensor else 0] for s in SENSOR_TARGETS}
    })

    final_input = input_data[features]
    return model.predict(final_input)[0]
