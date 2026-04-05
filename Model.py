import requests
import pandas as pd
import numpy as np
import sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pandas as pd
import pytz
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor




API_KEY = 'be45314bce4008639d5a2ae0e56f4ba2'
BASE_URL= 'https://api.openweathermap.org/data/2.5/'   # BASE URL FOR MAKING API REQUESTS



# FETCH CURRENT WEATHER DATA

import requests

BASE_URL = "https://api.openweathermap.org/data/2.5/"
API_KEY = "be45314bce4008639d5a2ae0e56f4ba2"   # replace with your actual key

def get_current_weather(city):

    # construct the API request URL properly
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)  # send the GET request
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
        'WindGustDir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'WindGustSpeed': data['wind'].get('gust', data['wind']['speed'])  # safer
    }



#  READ HISTORICAL DATA

def read_historoical_data():
  df = pd.read_csv("D:/weather.csv")
  df = df.dropna()
  df = df.drop_duplicates
  return df


# PREPARE DATA FOR TRAINING

def prepare_data(data):
  le = LabelEncoder() #create a LabelEncoder instance
  data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  #define the feature variable and target variables

  X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure' , 'Temp']]
  y = data['RainTomorrow'] #target variable

  return X, y, le   #return feture variable, target variable and the label encoder


# TRAIN RAIN PREDICTION MODEL

def train_rain_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train) #train the model

  y_pred = model.predict(X_test) #to make predictions on test set

  print("Mean Squared Error for Rain Model")

  print(mean_squared_error(y_test, y_pred))

  return model


# PREPARE REGGRESSION DATA 


def prepare_regression_data(data, feature):
 X, y = [], [] #initialize list for feature and target values

 for i in range(len(data) - 1):
   X.append(data[feature].iloc[i])

   y.append(data[feature].iloc[i+1])

 X = np.array(X).reshape(-1, 1)
 y = np.array(y)
 return X, y


# TRAIN REGRESSION MODEL


def train_regression_model(X, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, y)
  return model




# PREDICT FUTURE

def predict_future(model, current_value):

  predictions = [current_value]

  for i in range(5):
   next_value = model.predict(np.array([[predictions[-1]]]))

  predictions.append(next_value[0])

  return predictions[1:]


# WEATHER ANALYSIS

def weather_view():
    city = input('Enter any city name: ')
    current_weather = get_current_weather(city)

    # load historical data
    historical_data = pd.read_csv('D:\weather.csv')

    # prepare and train the rain prediction model
    X, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    # map wind direction to compass points

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

    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['WindGustSpeed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp']
    }

    current_df = pd.DataFrame([current_data])

    # Rain Prediction
    rain_prediction = rain_model.predict(current_df)[0]

    # Regression models
    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    X_humidity, y_humidity = prepare_regression_data(historical_data, 'Humidity')

    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_humidity, y_humidity)

    # Future predictions
    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])

    timezone = pytz.timezone('Asia/Karachi')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

    # Display results
    print(f"City: {city}, {current_weather['country']}")
    print(f"Current Temperature: {current_weather['current_temp']}°C")
    print(f"Feels Like: {current_weather['feels_like']}°C")
    print(f"Minimum Temperature: {current_weather['temp_min']}°C")
    print(f"Maximum Temperature: {current_weather['temp_max']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Weather Prediction: {current_weather['description']}")
    print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")

    print("\nFuture Temperature Predictions:")
    for time, temp in zip(future_times, future_temp):
        print(f"{time}: {round(temp, 1)}°C")

    print("\nFuture Humidity Predictions:")
    for time, humidity in zip(future_times, future_humidity):
        print(f"{time}: {round(humidity, 1)}%")

# Call the function
weather_view()
