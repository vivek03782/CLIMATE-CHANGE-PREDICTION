import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz

# --- CONSTANTS ---
API_KEY = 'be45314bce4008639d5a2ae0e56f4ba2'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'
CSV_PATH = 'D:/weather.csv'

# --- HELPER FUNCTIONS ---

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'WindGustDir': data['wind'].get('deg', 0),
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind'].get('gust', data['wind']['speed'])
    }

# Cache the model training so it doesn't re-train every time you click a button
@st.cache_resource
def load_and_train_models():
    # Read Historical Data
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.dropna()
        df = df.drop_duplicates() # Fixed missing parentheses here
    except FileNotFoundError:
        st.error(f"Could not find data at {CSV_PATH}. Please check the path.")
        return None, None, None, None, None

    # Prepare Data for Rain Model
    le = LabelEncoder()
    df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
    df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

    X_rain = df[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure' , 'Temp']]
    y_rain = df['RainTomorrow']

    # Train Rain Model
    rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rain_model.fit(X_rain, y_rain)

    # Prepare & Train Regression Models
    def prepare_regression_data(data, feature):
        X, y = [], []
        for i in range(len(data) - 1):
            X.append(data[feature].iloc[i])
            y.append(data[feature].iloc[i+1])
        return np.array(X).reshape(-1, 1), np.array(y)

    X_temp, y_temp = prepare_regression_data(df, 'Temp')
    X_humidity, y_humidity = prepare_regression_data(df, 'Humidity')

    temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    temp_model.fit(X_temp, y_temp)

    hum_model = RandomForestRegressor(n_estimators=100, random_state=42)
    hum_model.fit(X_humidity, y_humidity)

    return rain_model, temp_model, hum_model, le, df

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Weather Predictor", page_icon="🌤️", layout="centered")

st.title("🌤️ AI Weather Predictor")
st.write("Enter a city name to get current weather conditions and AI-powered future predictions.")

# Load models behind the scenes
rain_model, temp_model, hum_model, le, historical_data = load_and_train_models()

# User Input
city_input = st.text_input('Enter City Name:', placeholder='e.g., London, Tokyo, Karachi')

if st.button("Get Weather & Predictions"):
    if not city_input:
        st.warning("Please enter a city name.")
    elif rain_model is None:
        st.error("Models failed to load. Check your CSV file.")
    else:
        with st.spinner('Fetching weather and running models...'):
            current_weather = get_current_weather(city_input)
            
            if current_weather is None:
                st.error("City not found. Please check the spelling.")
            else:
                # --- DISPLAY CURRENT WEATHER ---
                st.subheader(f"📍 {current_weather['city']}, {current_weather['country']}")
                st.write(f"**Condition:** {current_weather['description'].title()}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Temperature", f"{current_weather['current_temp']} °C", f"Feels like {current_weather['feels_like']} °C")
                col2.metric("Humidity", f"{current_weather['humidity']} %")
                col3.metric("Wind Speed", f"{current_weather['WindGustSpeed']} m/s")

                # --- PREPARE DATA FOR PREDICTION ---
                wind_deg = current_weather['WindGustDir'] % 360
                compass_points = [
                    ("N", 348.75, 360), ("N", 0, 11.25), ("NNE", 11.25, 33.75),
                    ("NE", 33.75, 56.25), ("ENE", 56.25, 78.75), ("E", 78.75, 101.25),
                    ("ESE", 101.25, 123.75), ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75),
                    ("S", 168.75, 191.25), ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25),
                    ("WSW", 236.25, 258.75), ("W", 258.75, 281.25), ("WNW", 281.25, 303.75),
                    ("NW", 303.75, 326.25), ("NNW", 326.25, 348.75)
                ]
                compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
                compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

                current_data = pd.DataFrame([{
                    'MinTemp': current_weather['temp_min'],
                    'MaxTemp': current_weather['temp_max'],
                    'WindGustDir': compass_direction_encoded,
                    'WindGustSpeed': current_weather['WindGustSpeed'],
                    'Humidity': current_weather['humidity'],
                    'Pressure': current_weather['pressure'],
                    'Temp': current_weather['current_temp']
                }])

                # --- RUN PREDICTIONS ---
                rain_prediction = rain_model.predict(current_data)[0]
                future_temp = predict_future(temp_model, current_weather['current_temp'])
                future_humidity = predict_future(hum_model, current_weather['humidity'])

                # --- DISPLAY PREDICTIONS ---
                st.divider()
                st.subheader("🔮 AI Predictions")
                
                if rain_prediction == 1:
                    st.warning("🌧️ **Rain Prediction:** It is likely to rain tomorrow.")
                else:
                    st.success("☀️ **Rain Prediction:** No rain expected tomorrow.")

                # Generate future times
                timezone = pytz.timezone('Asia/Karachi')
                now = datetime.now(timezone)
                next_hour = now + timedelta(hours=1)
                next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
                future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

                # Build a dataframe for the tables/charts
                predictions_df = pd.DataFrame({
                    "Time": future_times,
                    "Temperature (°C)": [round(t, 1) for t in future_temp],
                    "Humidity (%)": [round(h, 1) for h in future_humidity]
                })

                col_t, col_h = st.columns(2)
                with col_t:
                    st.write("**Next 5 Hours: Temperature**")
                    st.dataframe(predictions_df[["Time", "Temperature (°C)"]].set_index("Time"), use_container_width=True)
                
                with col_h:
                    st.write("**Next 5 Hours: Humidity**")
                    st.dataframe(predictions_df[["Time", "Humidity (%)"]].set_index("Time"), use_container_width=True)