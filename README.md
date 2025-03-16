# Automated Trading System

An advanced Python-based trading platform that leverages machine learning algorithms to predict market movements and execute trades automatically. This project integrates real-time data visualization, predictive modeling, and a user-friendly web interface using Streamlit.

## Overview
The Automated Trading System is designed to assist traders by:
- Predicting stock price movements using AI-based models.
- Providing real-time trading insights with interactive visualizations.
- Allowing users to test different trading strategies.

This system automates the trading process by analyzing historical data and making decisions based on machine learning predictions.

---

## Key Features
- **Machine Learning-Based Predictions** – Predicts market trends using trained ML models.  
- **Live Trading Dashboard** – Interactive UI to visualize stock performance.  
- **Backtesting & Strategy Evaluation** – Simulates trading strategies based on historical data.  
- **API Integration** – Fetches real-time stock data dynamically.  
- **Risk Management Tools** – Implements smart stop-loss strategies.  
- **Multi-Page Streamlit Interface** – A clean, interactive web dashboard for trading insights.  

---

## Installation
### Prerequisites
Ensure you have Python 3.12+ and `pip` installed. It is recommended to use conda or virtual environments.

### Clone the Repository
```bash
git clone https://github.com/yourusername/Automated-Trading-System.git
cd Automated-Trading-System
```

### Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
### Running the Streamlit App
```bash
streamlit run main.py
```
This will launch the Automated Trading System Dashboard in your browser.

### Training the Machine Learning Model
To train a new model using `ml_model.py`, run the following command:
```bash
python ml_model.py --ticker AAPL
```
Replace `AAPL` with the desired stock ticker to train a new model for that stock.

### Navigating the Dashboard
- **Overview** – Learn about the system.  
- **Core Features** – Explore available functionalities.  
- **Live Trading Page** – View real-time stock predictions.  
- **Trading Strategy** – Simulate investment strategies.

---

## Technology Stack
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Data Handling:** `pandas`, `numpy`
- **Backend:** `Streamlit`

---

## Machine Learning Model Architecture
The model is built using XGBoost, a powerful gradient boosting algorithm, trained on historical stock price data. The architecture follows these steps:
1. **Data Preprocessing** – Cleans and prepares stock market data.
2. **Feature Engineering** – Extracts important features like moving averages, RSI, and momentum indicators.
3. **Model Training** – Uses XGBoost to train a predictive model for stock price movement.
4. **Prediction & Evaluation** – Generates forecasts and evaluates accuracy with backtesting.

---

## Team
| Name | Role |
|------|------|
| Massimo Tassinari | Lead Model Creation |
| Pablo Viaña | Lead API Configuration |
| Maha Alkaabi | Lead Streamlit |
| Yotaro Enomoto | Lead UI & Frontend |
| Yihang Li | Lead Model Optimization |

---

## License
This project is licensed under the MIT License – feel free to use and modify it.
