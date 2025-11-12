import streamlit as st
import pandas as pd
import joblib
import os


# Очистка всіх кешованих даних і ресурсів
st.cache_data.clear()
st.cache_resource.clear()

# -----------------------------
# Заголовок з картинкою
# -----------------------------

st.markdown(
    "<h1 style='text-align: center; color: steelblue;'>🌦️ Прогнозування дощу в Австралії</h1>",
    unsafe_allow_html=True
)

# Центруємо картинку
col1, col2, col3 = st.columns([1, 4, 1])  # центральна колонка ширша
with col2:
    st.image("image/rain.jpg") #, use_container_width=True)
  
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Модель прогнозує ймовірність дощу завтра на основі введених погодних параметрів.</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Завантаження моделі та defaults
# -----------------------------

BASE_DIR = os.path.dirname(__file__) 

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model", "rf_weather.pkl")
    return joblib.load(model_path)

@st.cache_resource
def load_defaults():
    defaults_path = os.path.join(BASE_DIR, "data", "defaults_streamlit.pkl")
    return joblib.load(defaults_path)

model = load_model()
defaults = load_defaults()

cat_options = defaults['categorical_options']
num_stats = defaults['numeric_stats']

# -----------------------------
# Введення даних користувачем
# -----------------------------
st.header("Введіть погодні параметри на сьогодні")

input_data = {}

# Розташування
st.subheader("📍 Розташування")
input_data['Location'] = st.selectbox(
    "Місце спостереження",
    cat_options['Location']  # всі варіанти локацій з defaults
)

# Температура
st.subheader("🌡️ Температура")
col1, col2 = st.columns(2)
with col1:
    input_data['MinTemp'] = st.slider("MinTemp", num_stats['MinTemp']['min'], num_stats['MinTemp']['max'], num_stats['MinTemp']['median'])
    input_data['Temp9am'] = st.slider("Temp9am", num_stats['Temp9am']['min'], num_stats['Temp9am']['max'], num_stats['Temp9am']['median'])
with col2:
    input_data['MaxTemp'] = st.slider("MaxTemp", num_stats['MaxTemp']['min'], num_stats['MaxTemp']['max'], num_stats['MaxTemp']['median'])
    input_data['Temp3pm'] = st.slider("Temp3pm", num_stats['Temp3pm']['min'], num_stats['Temp3pm']['max'], num_stats['Temp3pm']['median'])

# Опади
st.subheader("☔ Опади")
col1, col2 = st.columns(2)
with col1:
    input_data['Rainfall'] = st.slider("Rainfall", num_stats['Rainfall']['min'], num_stats['Rainfall']['max'], num_stats['Rainfall']['median'])
    input_data['Evaporation'] = st.slider("Evaporation", num_stats['Evaporation']['min'], num_stats['Evaporation']['max'], num_stats['Evaporation']['median'])
with col2:
    input_data['Sunshine'] = st.slider("Sunshine", num_stats['Sunshine']['min'], num_stats['Sunshine']['max'], num_stats['Sunshine']['median'])
    input_data['RainToday'] = st.selectbox("RainToday", cat_options['RainToday'])

# Вітер
st.subheader("💨 Вітер")
col1, col2 = st.columns(2)
with col1:
    input_data['WindGustDir'] = st.selectbox("WindGustDir", cat_options['WindGustDir'])
    input_data['WindDir9am'] = st.selectbox("WindDir9am", cat_options['WindDir9am'])
    input_data['WindSpeed9am'] = st.slider("WindSpeed9am", num_stats['WindSpeed9am']['min'], num_stats['WindSpeed9am']['max'], num_stats['WindSpeed9am']['median'])
with col2:
    input_data['WindGustSpeed'] = st.slider("WindGustSpeed", num_stats['WindGustSpeed']['min'], num_stats['WindGustSpeed']['max'], num_stats['WindGustSpeed']['median'])
    input_data['WindDir3pm'] = st.selectbox("WindDir3pm", cat_options['WindDir3pm'])
    input_data['WindSpeed3pm'] = st.slider("WindSpeed3pm", num_stats['WindSpeed3pm']['min'], num_stats['WindSpeed3pm']['max'], num_stats['WindSpeed3pm']['median'])

# Вологість та тиск
st.subheader("💧 Вологість та тиск")
col1, col2 = st.columns(2)
with col1:
    input_data['Humidity9am'] = st.slider("Humidity9am", num_stats['Humidity9am']['min'], num_stats['Humidity9am']['max'], num_stats['Humidity9am']['median'])
    input_data['Pressure9am'] = st.slider("Pressure9am", num_stats['Pressure9am']['min'], num_stats['Pressure9am']['max'], num_stats['Pressure9am']['median'])
    input_data['Cloud9am'] = st.slider("Cloud9am", num_stats['Cloud9am']['min'], num_stats['Cloud9am']['max'], num_stats['Cloud9am']['median'])
with col2:
    input_data['Humidity3pm'] = st.slider("Humidity3pm", num_stats['Humidity3pm']['min'], num_stats['Humidity3pm']['max'], num_stats['Humidity3pm']['median'])
    input_data['Pressure3pm'] = st.slider("Pressure3pm", num_stats['Pressure3pm']['min'], num_stats['Pressure3pm']['max'], num_stats['Pressure3pm']['median'])
    input_data['Cloud3pm'] = st.slider("Cloud3pm", num_stats['Cloud3pm']['min'], num_stats['Cloud3pm']['max'], num_stats['Cloud3pm']['median'])

# -----------------------------
# Прогноз
# -----------------------------
if st.button("Прогнозувати дощ завтра"):
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    st.success(f"Прогноз: {'Буде дощ' if prediction == 'Yes' else 'Дощу не буде'}")
    st.info(f"Ймовірність дощу завтра: {round(proba * 100, 1)}%")
