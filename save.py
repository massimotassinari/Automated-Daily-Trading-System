# ---------------------------
# LIVE TRADING PAGE
# ---------------------------

##HEADER

import streamlit as st
import pandas as pd
import api_functions as api
from lightweight_charts.widgets import StreamlitChart
import functions as function
import joblib
import os
if "symbols_list" not in st.session_state:
    st.session_state.symbols_list = None
    
st.set_page_config(
    layout = 'wide',
    page_title = 'Automated Trading System'
)

st.markdown(
    """
    <style>
        footer {display: none}
        [data-testid="stHeader"] {display: none}
    </style>
    """, unsafe_allow_html = True
)
# Define the folder path
model_dir = "trained_models"

# Get list of available companies
allowed_tickers = function.get_available_companies(model_dir)


with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)



title_col, emp_col, company_col, open_col, high_col, low_col, close_col = st.columns([1,0.5,1,1,1,1,1])


## BODY
# Define Streamlit columns (Chart & Table now equal)
params_col, chart_col, data_col = st.columns([0.5, 1, 1])  # ✅ Chart & Table same width



# Ensure df_company contains allowed tickers
#if 'ticker' in df_company.columns:
#    available_tickers = df_company[df_company['ticker'].isin(allowed_tickers)]['ticker'].unique()
#else:
#    available_tickers = allowed_tickers  # Fallback in case df_company is missing tickers
#'''
with params_col:
    with st.form(key='params_form'):
        st.markdown(f'<p class="params_text">CHART DATA PARAMETERS', unsafe_allow_html=True)
        st.divider()
        
        # Dropdown for selecting ticker
        ticker = st.selectbox('Select Stock Ticker', allowed_tickers, key='ticker_selectbox')
        #######
        simfin = api.SimFinAPI(ticker)

        # Fetch company info and stock data (defaults to last two weeks)
        simfin.fetch_company_info()
        simfin.fetch_stock_data()  # Uses last two weeks by default

        df_company = simfin.get_company_dataframe()
        df_stock = simfin.get_stock_dataframe()
        
        ######   
        # Select data range (kept only this option)
        period_selection = st.selectbox("Select Data Range", 
                                        ["1 Day", "3 Days", "5 Days", "1 Week", "1 Month"], 
                                        key='period_selectbox')
        
        # Map period selection to numerical days
        period_map = {
            "1 Day": 1,
            "3 Days": 3,
            "5 Days": 5,
            "1 Week": 7,
            "1 Month": 30
        }
        period = period_map[period_selection]

        st.markdown('')
        update_chart = st.form_submit_button('Update chart')
        st.markdown('')

        if update_chart:
            # Ensure 'Date' column is in datetime format
            stock_data = df_stock.copy()
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])

            # Add 'ticker' column to df_stock by assigning the correct ticker
            stock_data['ticker'] = df_company['ticker'].iloc[0]  # Assign first available ticker

            # Merge stock data with df_company to ensure correct mapping
            stock_data = stock_data.merge(df_company[['ticker', 'name']], on='ticker', how='left')

            # Filter the selected time period
            latest_date = stock_data['Date'].max()
            stock_data = stock_data[(stock_data['ticker'] == ticker) & 
                                    (stock_data['Date'] >= latest_date - pd.DateOffset(days=period))]

            # Drop unnecessary columns
            stock_data = stock_data.drop(columns=['Dividend Paid', 'Common Shares Outstanding', 
                                                  'Adjusted Closing Price', 'ticker', 'name'], errors='ignore')

            # ✅ Ensure numeric columns are properly formatted
            numeric_cols = ['Last Closing Price', 'Opening Price', 'Highest Price', 'Lowest Price', 'Trading Volume']
            for col in numeric_cols:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

            with chart_col:
                with st.container():
                    chart = StreamlitChart(height=500, width=950)  # ✅ Fixed width (real number)
                    chart.grid(vert_enabled=True, horz_enabled=True)

                    chart.layout(background_color='#131722', font_family='Trebuchet MS', font_size=16)

                    chart.candle_style(up_color='#2962ff', down_color='#e91e63',
                                       border_up_color='#2962ffcb', border_down_color='#e91e63cb',
                                       wick_up_color='#2962ffcb', wick_down_color='#e91e63cb')

                    chart.volume_config(up_color='#2962ffcb', down_color='#e91e63cb')
                    chart.legend(visible=True, font_family='Trebuchet MS', ohlc=True, percent=True)

                    # Renaming for chart compatibility
                    hist_df = stock_data.rename(columns={
                        'Date': 'time',
                        'Last Closing Price': 'close',
                        'Opening Price': 'open',
                        'Highest Price': 'high',
                        'Lowest Price': 'low',
                        'Trading Volume': 'volume'
                    })

                    chart.set(hist_df)
                    chart.load()
                    
            with data_col:
                st.dataframe(hist_df, height=500)  # ✅ Table height matches chart


with title_col:
    st.markdown('<p class="dashboard_title">Automated <br>Trading <br>System</p>', unsafe_allow_html = True)

    st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

with company_col:
    with st.container():
        ticker = df_company.iloc[-1]['ticker']
        st.markdown(f'<p class="company_text">Ticker<br></p><p class="stock_details">{ticker}</p>', unsafe_allow_html = True)

with open_col:
    with st.container():
        opening_price = df_stock.iloc[-1]['Opening Price']
        st.markdown(f'<p class="open_text">Open<br></p><p class="stock_details">{opening_price}</p>', unsafe_allow_html = True)

with high_col:
    with st.container():
        high_price = df_stock.iloc[-1]['Highest Price']
        st.markdown(f'<p class="high_text">High<br></p><p class="stock_details">{high_price}</p>', unsafe_allow_html = True)

with low_col:
    with st.container():
        low_price = df_stock.iloc[-1]['Lowest Price']
        st.markdown(f'<p class="low_text">Low<br></p><p class="stock_details">{low_price}</p>', unsafe_allow_html = True)

with close_col:
    with st.container():
        close_price = df_stock.iloc[-1]['Last Closing Price']
        st.markdown(f'<p class="close_text">Close<br></p><p class="stock_details">{close_price}</p>', unsafe_allow_html = True)
    

# Construct the model file path dynamically
model_path = os.path.join(model_dir, f"xgb_model_final_{ticker}.pkl")

    # Load the model
xgb_model_final = joblib.load(model_path)

y = function.predict_next_day_xgboost_api(xgb_model_final, ticker)
    # Button to confirm selection
st.success(f"The prediction for tomorrows is that the stock goes: {y}")


#######

import streamlit as st
import pandas as pd
import functions as function

st.set_page_config(
    layout="wide",
    page_title="Data Retrieval Page"
)

st.title("Retrieved Data")

# ✅ Check if stored data exists
if "ticker" in st.session_state:
    st.write(f"**Selected Stock:** {st.session_state.ticker}")
else:
    st.warning("No stock selected.")

if "df_stock" in st.session_state:
    st.write("**Chart Data:**")
    st.dataframe(st.session_state.df_stock)  # Display stored DataFrame
else:
    st.warning("No df_stock data available.")

if "df_company" in st.session_state:
    st.write("**df_company:**")
    st.dataframe(st.session_state.df_company)  # Display stored DataFrame
else:
    st.warning("No df_company data available.")

if "prediction_df" in st.session_state:
    st.write("**prediction_df:**")
    st.dataframe(st.session_state.prediction_df)  # Display stored DataFrame
else:
    st.warning("No prediction_df data available.")

# ✅ Add a button to go back to the main trading page
if st.button("Go Back to Trading Page"):
    st.switch_page("pages/Live_Trading.py")  # Ensure this matches your file structure


#######

import streamlit as st
import pandas as pd
import numpy as np  # Import NumPy

st.set_page_config(
    layout="wide",
    page_title="Data Retrieval Page"
)

st.title("Retrieved Data & Trading Policy")

# ✅ Check if stored data exists
if "ticker" in st.session_state:
    st.write(f"**Selected Stock:** {st.session_state.ticker}")
else:
    st.warning("No stock selected.")

if "df_stock" in st.session_state:
    st.write("**Chart Data:**")
    st.dataframe(st.session_state.df_stock)  # Display stored DataFrame
else:
    st.warning("No df_stock data available.")

if "df_company" in st.session_state:
    st.write("**df_company:**")
    st.dataframe(st.session_state.df_company)  # Display stored DataFrame
else:
    st.warning("No df_company data available.")
######
######
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide",
    page_title="Trading Strategy Analysis"
)

st.title("Trading Strategy Performance")

# 🔹 General Explanation of How AI Trading Works
st.subheader("📌 How Does AI Trading Work?")
st.write("""
This AI trading strategy predicts whether a stock's price will go **up** or **down** the next day:
- **Prediction `1`** → AI expects the price to increase → **Buys stock**.
- **Prediction `0`** → AI expects the price to decrease → **Sells stock** (or holds cash).
The goal is to **maximize profit** by making better buy/sell decisions than a simple Buy & Hold strategy.
""")

# ✅ Ensure stored data exists
if "prediction_df" in st.session_state and "df_stock" in st.session_state:

    # Ensure `prediction_df` is a DataFrame
    if isinstance(st.session_state.prediction_df, np.ndarray):
        st.session_state.prediction_df = pd.DataFrame(st.session_state.prediction_df, columns=["value"])

    # Handle missing last 3 days
    missing_predictions = len(st.session_state.df_stock) - len(st.session_state.prediction_df)
    if missing_predictions > 0:
        trading_df = st.session_state.df_stock.iloc[:-missing_predictions].copy()
    else:
        trading_df = st.session_state.df_stock.copy()

    # Ensure predictions exist
    if "value" not in st.session_state.prediction_df.columns:
        st.warning("The 'value' column is missing from prediction_df.")
    else:
        trading_df["Prediction"] = st.session_state.prediction_df["value"].values

        if "Last Closing Price" not in trading_df.columns:
            st.warning("Closing prices missing from stock data.")
        else:
            # ✅ Initialize variables
            initial_balance = 1000
            balance = initial_balance
            shares = 0
            portfolio_values = []
            buy_sell_signals = []

            # Simulate trading
            for index, row in trading_df.iterrows():
                closing_price = row["Last Closing Price"]
                prediction = row["Prediction"]

                if prediction == 1 and balance > 0:  # Buy signal
                    shares = balance / closing_price
                    balance = 0
                    buy_sell_signals.append(("Buy", index, closing_price))

                elif prediction == 0 and shares > 0:  # Sell signal
                    balance = shares * closing_price
                    shares = 0
                    buy_sell_signals.append(("Sell", index, closing_price))

                # Store portfolio value
                portfolio_values.append(balance + (shares * closing_price))

            # Final portfolio value
            final_value = balance + (shares * trading_df.iloc[-1]["Last Closing Price"])

            # Buy & Hold Strategy
            initial_shares = initial_balance / trading_df.iloc[0]["Last Closing Price"]
            hold_final_value = initial_shares * trading_df.iloc[-1]["Last Closing Price"]

            # ✅ Display Final Results in a More Visual Way
            st.subheader("💰 Final Portfolio Performance")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="📌 AI Trading Strategy", value=f"${final_value:.2f}")
            with col2:
                st.metric(label="📌 Buy & Hold Strategy", value=f"${hold_final_value:.2f}")

            # ✅ Plot Portfolio Growth Over Time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trading_df.index, portfolio_values, label="AI Trading Strategy", color="blue")
            ax.axhline(y=hold_final_value, color="gray", linestyle="dashed", label="Buy & Hold Final Value")
            ax.set_title("Portfolio Value Over Time")
            ax.set_xlabel("Days")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            st.pyplot(fig)

            # ✅ Plot Buy & Sell Signals on Stock Price Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trading_df.index, trading_df["Last Closing Price"], label="Stock Price", color="black")

            # Add buy/sell markers
            for action, idx, price in buy_sell_signals:
                if action == "Buy":
                    ax.scatter(idx, price, color="green", marker="^", label="Buy Signal" if "Buy Signal" not in ax.get_legend_handles_labels()[1] else "")
                elif action == "Sell":
                    ax.scatter(idx, price, color="red", marker="v", label="Sell Signal" if "Sell Signal" not in ax.get_legend_handles_labels()[1] else "")

            ax.set_title("Stock Price with Buy & Sell Signals")
            ax.set_xlabel("Days")
            ax.set_ylabel("Stock Price ($)")
            ax.legend()
            st.pyplot(fig)

else:
    st.warning("No prediction_df data available.")

# ✅ Button to go back to the main trading page
if st.button("Go Back to Trading Page"):
    st.switch_page("pages/Live_Trading.py")  # Ensure this matches your file structure

###
##
#

import numpy as np
import pandas as pd
import streamlit as st
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
#selected_ticker = "BLZE" 
import api_functions as api
import joblib
import re

def prepare_company_data(df, ticker):
    """
    Prepare model data for a specific company based on its ticker.
    """
    df_company = df[df['ticker'] == ticker].copy()  # Filter for selected company
    
    # Ensure data is sorted correctly
    df_company = df_company.sort_values(by='date')

    # Creating lag features (last 3 days closing prices)
    df_company['close_t-1'] = df_company['close'].shift(1)
    df_company['close_t-2'] = df_company['close'].shift(2)
    df_company['close_t-3'] = df_company['close'].shift(3)

    # Target Variable: Next day's price movement (1 if up, 0 if down)
    df_company['target'] = (df_company['close'].shift(-1) > df_company['close']).astype(int)

    # Drop NaN values caused by shifting
    df_company.dropna(inplace=True)

    return df_company[['date', 'close_t-3', 'close_t-2', 'close_t-1', 'target']]


def transform_dataframe(df,ticker):
    
    # Step 1: Rename existing columns to match target schema
    rename_dict = {
        'Date': 'date',
        'Dividend Paid': 'dividend',
        'Common Shares Outstanding': 'shares_outstanding',
        'Last Closing Price': 'close',
        'Adjusted Closing Price': 'adj._close',
        'Highest Price': 'high',
        'Lowest Price': 'low',
        'Opening Price': 'open',
        'Trading Volume': 'volume'
    }
    df = df.rename(columns=rename_dict)

    # Step 2: Add missing columns with default values
    print(df.columns)
    if 'ticker' not in df.columns:
        df['ticker'] = ticker  # or assign a default value if known
    
    # Step 3: Reorder columns to match target structure
    target_columns = [
        'ticker','date', 'open', 'high', 'low', 'close',
        'adj._close', 'volume', 'dividend', 'shares_outstanding'
    ]
    df = df[target_columns]

    return df
def get_latest_stock_data(ticker):
    
    simfin = api.SimFinAPI(ticker)

    # Fetch company info and stock data (defaults to last two weeks)
    simfin.fetch_company_info()
    simfin.fetch_stock_data()  # Uses last two weeks by default

    # Get and display DataFrames
    #df_company = simfin.get_company_dataframe()
    #df_stock = simfin.get_stock_dataframe()

    latest_data = simfin.get_stock_dataframe()

    # Example usage:
    latest_data = transform_dataframe(latest_data,ticker)


    latest_data = latest_data.drop(columns=["dividend"], errors="ignore")

    # Apply the function
    latest_data = prepare_company_data(latest_data, ticker)
    # Check the processed data
    #print(f"✅ Data Prepared for {selected_ticker}. Shape:", latest_data.shape)
    #print("🔍 Sample Data:\n", latest_data.head())

    return latest_data
def predict_next_day_xgboost_api(model, ticker):
    """
    Fetch latest stock data from SimFin and predict whether the stock will go up or down.
    """
    # Fetch latest stock data from SimFin API
    latest_data = get_latest_stock_data(ticker)

    if latest_data is None:
        print("❌ Could not retrieve latest stock data. Aborting prediction.")
        return

    # Ensure feature alignment: Keep only the features used in training
    model_features = model.feature_names_in_
    latest_data = latest_data[model_features]  # Select only relevant columns

    # Make Prediction
    prediction = model.predict(latest_data)
    prediction_label = "📈 Up" if prediction[0] == 1 else "📉 Down"

    return prediction,prediction_label

# Function to extract available companies from model filenames
def get_available_companies(model_dir):
        companies = []
        if os.path.exists(model_dir):  # Check if folder exists
            for filename in os.listdir(model_dir):
                match = re.match(r"xgb_model_final_(.+)\.pkl", filename)  # Extract ticker from filename
                if match:
                    companies.append(match.group(1))  # Add ticker to list
        return sorted(companies)  # Return sorted company tickers



#
##
def load_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
            
            /* REMOVE STREAMLIT DEFAULT SPACING */
            .main .block-container {
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
            }

            /* HIDE FOOTER & HEADER */
            footer {display: none !important;}
            [data-testid="stHeader"] {display: none !important;}

            /* GLOBAL FONT & COLOR SETTINGS */
            html, body, [class*="css"] {
                font-family: 'Space Grotesk', sans-serif !important;
                background-color: #0f0f0f !important;
                color: #f6f6f6 !important;
            }

            /* PAGE TITLE */
            .dashboard_title {
                font-size: 100px; 
                font-family: 'Space Grotesk';
                font-weight: 700;
                line-height: 1.2;
                text-align: left;
            }

            /* NAVIGATION CONTAINER */
            .nav-container {
                display: flex;
                justify-content: center;
                background-color: #111;
                padding: 15px 30px;
                width: 100%;
                border-bottom: 1px solid #333;
            }

            /* NAVIGATION LINKS */
            .nav-text {
                font-size: 18px;
                font-weight: bold;
                color: white;
                text-decoration: none;
                cursor: pointer;
                transition: color 0.3s ease;
            }
            .nav-text:hover {
                color: #f7931a;
            }

            /* CUSTOM MARKDOWN CLASSES */
            .stock_details {
                font-size: 30px; 
                font-family: 'Space Grotesk';
                color: #f6f6f6;
                font-weight: 900;
                text-align: left;
                line-height: 1;
                padding-bottom: 10px;
            }
        </style>
        """,
       unsafe_allow_html=True
   )

load_css()

#
#
#
import streamlit as st
import os

st.set_page_config(
    layout="wide",
    page_title="Automated Trading System"
)

# Load external CSS for styling


# ---------------------------
# LAYOUT: TITLE AND NAVIGATION
# ---------------------------

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

st.markdown('<p class="dashboard_title">Automated Trading System</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])  # Centering the radio more


with col2:
    page_selection = st.radio(
        label="",
        options=["Overview", "Core Features", "Meet the Team"],
        horizontal=True,
        key="main_nav"
    )

# ---------------------------
# DYNAMIC CONTENT BASED ON SELECTION
# ---------------------------
if page_selection == "Overview":
    st.markdown("## 📄 Overview")  # Keeps the section title

    st.markdown(
    """
        - #### Welcome to the <strong>Automated Daily Trading System</strong>, an advanced Python-based platform designed to automate daily stock trading.<br>
        - #### This system leverages cutting-edge <strong>machine learning algorithms</strong> to predict market movements and provides an <strong>interactive web-based interface</strong> for traders to monitor and execute strategies in real-time.

    """,
    unsafe_allow_html=True
)


elif page_selection == "Core Features":
    st.markdown("## 📡Core Features")
    st.markdown(
        """
        - #### **Data Analytics Module:** Develops a machine learning model for market movement forecasting based on historical data from at least five major US companies.
        - #### **Web-Based Application:** A user-friendly multi-page interactive dashboard built with Streamlit to analyze stock trends and interact with predictive models.
        - #### **Live Trading Page:** Real-time stock movement visualization and execution of trades for selected stock tickers.
        - #### **Trading Strategy Module:** Implementation of machine learning-based trading strategies to optimize investment decisions.
        """
    )

elif page_selection == "Meet the Team":
    st.markdown("## 👫Meet the Team")

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

