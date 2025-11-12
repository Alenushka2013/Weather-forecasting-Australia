# Weather-forecasting-Australia
A Python project for predicting the probability of rain in Australia based on weather data. The project uses machine learning models and includes a Streamlit web application for interactive visualization of the predictions.



---

## 🌦️ Weather Forecasting Australia — Streamlit App

An interactive machine learning app that predicts the likelihood of rain tomorrow in Australia based on weather parameters. Built with **Streamlit** and powered by a **Random Forest Classifier**.

🔗 [Launch the app](https://weather-forecasting-australia-erxbv3refwlcnnhffynuaz.streamlit.app/)

---

### 📌 Project Goals

- Train a machine learning model to predict `RainTomorrow`
- Build a user-friendly web interface for inputting weather data
- Deploy the app publicly using Streamlit Cloud

---

### 🧠 Technologies Used

- **Python 3.11**
- **scikit-learn** — RandomForestClassifier
- **pandas**, **numpy**, **joblib**
- **Streamlit** — for interactive UI
- **VS Code** — development environment
- **GitHub + Streamlit Cloud** — version control and deployment

---

### 📁 Project Structure

```
Weather-forecasting-Australia/
├── model_training.py         # Model training script
├── streamlit_app.py          # Streamlit interface
├── model/
│   └── rf_weather.pkl        # Saved model
├── image/
│   └── rain.jpg              # UI illustration
├── weatherAUS.csv            # Input dataset
├── requirements.txt          # Dependencies
├── README.md                 # Project description
```

---

### 🚀 Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Weather-forecasting-Australia.git
   cd Weather-forecasting-Australia
   ```

2. Create and activate a virtual environment:
   ```bash
   py -3.11 -m venv .venv311
   .\.venv311\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

### 📊 Input Parameters

- Temperature, humidity, pressure, wind speed, rainfall, and other weather features
- The model predicts `RainTomorrow` as `Yes` or `No`

---
