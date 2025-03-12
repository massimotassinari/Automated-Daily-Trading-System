import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import streamlit as st
import os

# ---------------------------
# Page Title
# ---------------------------
st.set_page_config(page_title="Automated Trading System", layout="wide")

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 20px !important;
        }
        
        section[data-testid="stSidebar"] > div {
            padding-top: 10px !important;
        }

        body { background-color: #121212; color: white; font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background-color: #1E1E1E; }
        h1, h2, h3 { color: #00ADB5; text-align: center; font-weight: bold; }

        .feature-box { 
            text-align: center; padding: 20px; border-radius: 12px; 
            background: #1E1E1E; min-width: 250px; margin: auto;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            transition: transform 0.2s ease-in-out;
        }
        .feature-box:hover {
            transform: scale(1.05);
        }

        .team-container { 
            display: flex; justify-content: center; flex-wrap: wrap; 
            gap: 50px; margin-top: 20px; 
        }
        
        .team-member { 
            text-align: center; width: 200px; padding: 15px; 
            background: #1E1E1E; border-radius: 10px; 
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        }
        .team-member img { 
            border-radius: 50%; width: 140px; height: 140px; 
            object-fit: cover; margin-bottom: 10px; 
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Automated Trading System")
page = st.sidebar.radio("Navigation", ["Home", "Live Trading"])

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "Home":
    st.markdown("# Automated Trading System")
    
    st.markdown("## Overview")
    st.info("""
        The Automated Trading System leverages AI-driven market analysis and real-time stock insights 
        to help traders make data-backed decisions. It integrates machine learning models for trend analysis, 
        stock forecasting, and live data tracking.
    """)

    st.markdown("## Core Features")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="feature-box"><b>Live Stock Data</b><br>Track real-time stock trends.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-box"><b>AI Predictions</b><br>Forecast stock movements using ML.</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-box"><b>Historical Analytics</b><br>Analyze past trends for smarter trading.</div>', unsafe_allow_html=True)

    st.markdown("## Meet the Team")

    IMAGE_DIR = "team/"  # Folder where images are stored

    team_members = [
        {"name": "Massimo Tassinari", "role": "Lead Model Creation", "image": os.path.join(IMAGE_DIR, "massimo.jpeg")},
        {"name": "Pablo Viaña", "role": "Lead API Configuration", "image": os.path.join(IMAGE_DIR, "pablo.jpeg")},
        {"name": "Maha Alkaabi", "role": "Lead Streamlit", "image": os.path.join(IMAGE_DIR, "maha.jpeg")},
        {"name": "Yotaro Enomoto", "role": "Lead Streamlit", "image": os.path.join(IMAGE_DIR, "yotaro.JPG")},
        {"name": "Yihang Li", "role": "Lead Model Creation", "image": os.path.join(IMAGE_DIR, "yihang.jpeg")}
    ]

    team_cols = st.columns(len(team_members))  # Creates equal columns for members

    for i, member in enumerate(team_members):
        with team_cols[i]:
            if os.path.exists(member["image"]):
                st.image(member["image"], width=140)  # Load images correctly
            else:
                st.warning(f"Image not found: {member['image']}")  # Debugging for missing images
            
            st.markdown(f"<p style='text-align: center;'><b>{member['name']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{member['role']}</p>", unsafe_allow_html=True)

# ---------------------------
# LIVE TRADING PAGE
# ---------------------------
elif page == "Live Trading":
    st.markdown("# Live Trading Dashboard")

    st.sidebar.markdown("### Choose Your Stock")
    stock_tickers = ["MSFT", "AAPL", "GOOGL", "TSLA", "AMZN"]
    selected_ticker = st.sidebar.selectbox("Choose Ticker", stock_tickers)

    time_period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y", "5Y"])

    def get_stock_data(ticker, period):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.date.today(), periods=365, freq="D")
        prices = np.cumsum(np.random.randn(365)) + 150
        volume = np.random.randint(5_000_000, 30_000_000, size=365)
        df = pd.DataFrame({"Date": dates, "Price": prices, "Volume": volume})

        return df.tail({"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 365 * 5}[period])

    stock_data = get_stock_data(selected_ticker, time_period)

    st.markdown(f"## {selected_ticker} Stock Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${stock_data['Price'].iloc[-1]:.2f}")
    with col2:
        st.metric("Day Change", "-0.67 (-0.20%)", delta="-0.20%")
    with col3:
        st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")

    st.markdown("### Stock Price Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(stock_data["Date"], stock_data["Price"], label="Price", color="cyan", linewidth=2)
    st.pyplot(fig)

    signal = "BUY" if np.random.rand() > 0.5 else "SELL"
    signal_color = "green" if signal == "BUY" else "red"
    st.markdown(f'<h2 style="color:{signal_color}; text-align:center;">Trading Signal: {signal}</h2>', unsafe_allow_html=True)
