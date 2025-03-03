import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Placeholder function for fetching real-time stock data from SimFin
def get_stock_data(ticker):
    return pd.DataFrame() 

# Placeholder function for predictive model
def prediction_model():
    pass

# App Layout
st.set_page_config(page_title="Trading System", layout="wide")

# Sidebar navigation
st.sidebar.title("📊 Trading System")
page = st.sidebar.radio("Navigation", ["Overview", "Go Live"])

# --------------------------------
# Overview Page
# --------------------------------
if page == "Overview":
    st.title("Automated Daily Trading System")
    st.subheader("Purpose & Objectives")
    st.write("""
        The goal of this system is to assist traders in making informed decisions based on market data and AI-powered insights. 
        By integrating **SimFin's financial data** and predictive models, the platform helps users optimize their trading strategies.
    """)

    st.subheader("Core Functionalities")
    st.write("""
        This system provides real-time stock market insights, historical trends, and market movement forecasting.  
        Core functionalities include:
        - **Live stock market data**
        - **Historical trends & analytics**
        - **Predictions if the price is higher or lower the next day**
    """)

    st.subheader("Info about the datasets")
    st.write("""
        
    """)

    st.subheader("ETL Approach")
    st.write("""
        
    """)

    st.subheader("Who are we?")
    st.write("""
        - **Massimo Tassinari**
        - **Pablo Viaña**
        - **Maha Alkwabi**
        - **Yotaro Enomoto**
        - **Yihang Li**        
    """)

# --------------------------------
# Go Live Page
# --------------------------------
elif page == "Go Live":
    st.title("Live Stock Market Data")

    # Company selector (placeholder tickers)
    stock_tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"] #Replace with function after ETL (Temparary placeeholder)
    selected_ticker = st.selectbox("Select a Stock Ticker:", stock_tickers)

    # Fetch and display historical stock data
    st.subheader(f"Historical Data for {selected_ticker}")
    stock_data = get_stock_data(selected_ticker)
    
    if stock_data.empty:
        st.warning("Stock data unavailable. Replace with actual SimFin API data.")
    else:
        fig, ax = plt.subplots()
        ax.plot(stock_data["Date"], stock_data["Price"], label=f"{selected_ticker} Stock Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

    # Higher/Lower prediction
    st.subheader("Next day prediction (Higher/Lower)")

