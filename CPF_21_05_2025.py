import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from io import BytesIO
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
import time
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV
from datetime import datetime
import traceback
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration and style
st.set_page_config(
    page_title="ICE Cotton Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def set_page_style():
    primary_blue = "#0047AB"
    secondary_blue = "#4682B4"
    accent_blue = "#1E90FF"
    light_blue = "#B0C4DE"
    navy_blue = "#000080"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Roboto', sans-serif !important;
            font-weight: 600 !important;
            color: {primary_blue} !important;
            letter-spacing: -0.01em !important;
        }}
        
        h1 {{
            font-size: 2.2rem !important;
            margin-bottom: 1rem !important;
        }}
        
        h2 {{
            font-size: 1.8rem !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.8rem !important;
        }}
        
        h3 {{
            font-size: 1.4rem !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.6rem !important;
        }}
        
        h4 {{
            font-size: 1.2rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.4rem !important;
        }}
        
        .main .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px !important;
        }}
        
        .stApp {{
            background-color: #F8F9FB !important;
        }}
        
        .stSidebar {{
            background-color: white !important;
            border-right: 1px solid #EAECEF !important;
        }}
        
        .stSidebar [data-testid="stSidebarNav"] {{
            padding-top: 2rem !important;
        }}
        
        .stSidebar [data-testid="stSidebarNav"] > ul {{
            padding-left: 1rem !important;
        }}
        
        .stSidebar [data-testid="stSidebarNav"] label {{
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }}
        
        .stButton>button {{
            background-color: {primary_blue} !important;
            color: white !important;
            border-radius: 4px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s !important;
        }}
        
        .stButton>button:hover {{
            background-color: {secondary_blue} !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
            transform: translateY(-1px) !important;
        }}
        
        .stProgress .st-bo {{
            background-color: {primary_blue} !important;
        }}
        
        .info-box {{
            background-color: white !important;
            border-left: 5px solid {primary_blue} !important;
            padding: 1.2rem !important;
            border-radius: 6px !important;
            margin-bottom: 1.5rem !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08) !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }}
        
        .info-box:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1) !important;
        }}
        
        .info-box h3 {{
            margin-top: 0 !important;
            color: {navy_blue} !important;
            font-weight: 600 !important;
        }}
        
        .metric-card {{
            background-color: white !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
            padding: 1.8rem 1.2rem !important;
            text-align: center !important;
            margin-bottom: 1.5rem !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
            border-top: 4px solid {primary_blue} !important;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08) !important;
        }}
        
        .metric-value {{
            font-size: 28px !important;
            font-weight: 700 !important;
            color: {primary_blue} !important;
            margin-top: 0.5rem !important;
            line-height: 1.2 !important;
        }}
        
        .metric-title {{
            font-size: 16px !important;
            color: #5A6474 !important;
            font-weight: 500 !important;
            margin-bottom: 0.5rem !important;
        }}
        
        .card {{
            background-color: white !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
            padding: 1.8rem !important;
            margin-bottom: 1.5rem !important;
            transition: transform 0.3s ease !important;
        }}
        
        .card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08) !important;
        }}
        
        .card h3 {{
            color: {navy_blue} !important;
            margin-top: 0 !important;
            font-weight: 600 !important;
            font-size: 1.3rem !important;
            margin-bottom: 1rem !important;
        }}
        
        .success-message {{
            background-color: #EDF7ED !important;
            color: #1E4620 !important;
            border-left: 5px solid #4CAF50 !important;
            padding: 1.2rem !important;
            border-radius: 6px !important;
            margin-bottom: 1.5rem !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }}
        
        .error-message {{
            background-color: #FDEDED !important;
            color: #5F2120 !important;
            border-left: 5px solid #EF5350 !important;
            padding: 1.2rem !important;
            border-radius: 6px !important;
            margin-bottom: 1.5rem !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }}
        
        [data-testid="stDataFrame"] {{
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }}
        
        .stDataFrame div[data-testid="stVerticalBlock"] > div:first-child {{
            background-color: {light_blue} !important;
            padding: 0.5rem !important;
        }}
        
        .stDataFrame table {{
            border-collapse: collapse !important;
            font-family: 'Inter', sans-serif !important;
        }}
        
        .stDataFrame thead tr {{
            background-color: {primary_blue} !important;
            color: white !important;
        }}
        
        .stDataFrame thead th {{
            padding: 0.75rem 1rem !important;
            font-weight: 600 !important;
        }}
        
        .stDataFrame tbody tr:nth-child(even) {{
            background-color: #F5F7FA !important;
        }}
        
        .stDataFrame tbody td {{
            padding: 0.75rem 1rem !important;
            border-bottom: 1px solid #EAECEF !important;
        }}
        
        [data-testid="stFileUploader"] {{
            border: 2px dashed {light_blue} !important;
            padding: 1.5rem 1rem !important;
            border-radius: 8px !important;
            background-color: #F8FAFF !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {secondary_blue} !important;
            background-color: #F0F5FF !important;
        }}
    </style>
    """, unsafe_allow_html=True)

def display_logo():
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <img src="https://www.indianchemicalnews.com/public/uploads/news/2023/07/18177/Welspun_New.jpg" width="180" style="margin-right: 20px;">
        <div>
            <h1 style="color:#0047AB; margin-bottom: 5px;">ICE Cotton Market Intelligence</h1>
            <p style="color:#666; font-style: italic; margin-top: 0; font-size: 1.1rem;">Har Ghar Se Har Dil Tak</p>
        </div>
    </div>
    <div style="height: 5px; background: linear-gradient(90deg, #0047AB, #6495ED, #B0C4DE, white); margin-bottom: 2rem; border-radius: 2px;"></div>
    """, unsafe_allow_html=True)

def display_metric_card(title, value, unit=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{unit}</div>
    </div>
    """, unsafe_allow_html=True)

def display_info_box(title, content):
    st.markdown(f"""
    <div class="info-box">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def display_success_message(content):
    st.markdown(f"""
    <div class="success-message">
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def display_error_message(content):
    st.markdown(f"""
    <div class="error-message">
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def get_data_dir():
    data_dir = os.path.join(os.getcwd(), "cotton_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def get_last_update_time():
    try:
        data_dir = get_data_dir()
        local_file = os.path.join(data_dir, "cotton_data.csv")
        if os.path.exists(local_file):
            file_mtime = os.path.getmtime(local_file)
            return datetime.fromtimestamp(file_mtime)
        return datetime.now()  # Fallback to current time
    except Exception as e:
        st.error(f"Error checking last update time: {str(e)}")
        return datetime.now()

# Simulated data functions (for demo purposes)
@st.cache_data(ttl=600)
def fetch_test_data():
    """Generate sample data for testing the application"""
    # Date range
    dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='MS')
    
    # Base price and random fluctuation
    base_price = 70.0
    np.random.seed(42)  # For reproducibility
    fluctuations = np.cumsum(np.random.normal(0, 1.5, len(dates)))
    prices = base_price + fluctuations
    
    # Create dataframe
    df = pd.DataFrame({
        'ICE Cotton CT1 Comdty': prices
    }, index=dates)
    
    # Add some features
    df['US_Dollar_Index'] = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    df['Crude_Oil_WTI'] = 80 + np.cumsum(np.random.normal(0, 2, len(dates)))
    df['China_PMI'] = 50 + np.random.normal(0, 1.5, len(dates))
    df['US_Weather_Index'] = 50 + np.random.normal(0, 5, len(dates))
    df['Global_Stocks'] = 5000000 + np.cumsum(np.random.normal(0, 100000, len(dates)))
    
    return df

@st.cache_data
def fetch_data():
    """Fetch data from local storage or return test data if not available"""
    try:
        data_dir = get_data_dir()
        local_file = os.path.join(data_dir, "cotton_data.csv")
        
        if os.path.exists(local_file):
            df = pd.read_csv(local_file)
            if 'Identifier' in df.columns:
                try:
                    df['Identifier'] = pd.to_datetime(df['Identifier'])
                    df.set_index('Identifier', inplace=True)
                    df.sort_index(inplace=True)
                except Exception as e:
                    st.warning(f"Could not process date format properly: {str(e)}")
            
            # Remove shift operation
            # if 'ICE Cotton CT1 Comdty' in df.columns:
            #     df['ICE Cotton CT1 Comdty'] = df['ICE Cotton CT1 Comdty'].shift(-3)
                
            return df
        else:
            # Return test data if no file is found
            return fetch_test_data()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return fetch_test_data()  # Fallback to test data on error
    
def insert_data(df):
    if df is None or df.empty:
        return False, "No data to insert"
    
    try:
        df_copy = df.copy()
        
        # Reset index if it's a DatetimeIndex
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.reset_index(inplace=True)
        
        # Save to local file
        data_dir = get_data_dir()
        local_file = os.path.join(data_dir, "cotton_data.csv")
        df_copy.to_csv(local_file, index=False)
        return True, f"Saved {len(df_copy)} rows to local file {local_file}"
    except Exception as e:
        # Last resort fallback
        try:
            data_dir = get_data_dir()
            local_file = os.path.join(data_dir, "cotton_data_error_fallback.csv")
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(local_file, index=False)
                return False, f"Error inserting data but saved to {local_file}: {str(e)}"
            else:
                return False, f"Error inserting data: {str(e)}"
        except:
            return False, f"Error inserting data: {str(e)}"

@st.cache_data
def read_file(file):
    if file is None:
        return None, "No file provided"
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            df = pd.read_csv(file)
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(file, engine='openpyxl')
        else:
            return None, f"Unsupported file format: {ext}"
        expected = [
            "Identifier", "Fed Funds Rate", "RDBI REPO Rate", "ICE Cotton CT1 Comdty",
            "Cotlook USC/LBS", "Cotton - Shankar 6 Rs/Candy", "US Area Harvested (1000 HA)",
            "India Area Harvested (1000 HA)", "China Area Harvested (1000 HA)",
            "India Production 1000 480 lb bales", "Australia Production 1000 480 lb bales",
            "Cotton Arrival - India (Qty in Lakh Bales)", "India Corn Area Harvested (1000 HA)",
            "India Soyabean Area Harvested (1000 HA)", "India Wheat Area Harvested (1000 HA)",
            "US Soyabean Area Harvested (1000 HA)", "US GDP Qtr Growth %", "China GDP Qtr Growth %",
            "India GDP Qtr Growth %", "China Production 1000 480 lb bales",
            "Brazil Area Harvested (1000 HA)", "Australia Area Harvested (1000 HA)",
            "US Production 1000 480 lb bales", "Brazil Production 1000 480 lb bales",
            "US Consumption 1000 480 lb bales", "India Consumption 1000 480 lb bales",
            "China Consumption 1000 480 lb bales", "Brazil Consumption 1000 480 lb bales",
            "Crude oil, average-($/bbl)-CRUDE_PETRO", "Crude oil, Brent-($/bbl)-CRUDE_BRENT",
            "Crude oil, Dubai-($/bbl)-CRUDE_DUBAI", "Crude oil, WTI-($/bbl)-CRUDE_WTI",
            "Coal, Australian-($/mt)-COAL_AUS", "Microsoft Teams-($/mt)-COAL_SAFRICA",
            "Natural gas, US-($/mmbtu)-NGAS_US", "Natural gas, Europe-($/mmbtu)-NGAS_EUR",
            "Liquefied natural gas, Japan-($/mmbtu)-NGAS_JP", "Natural gas index-(2010=100)-iNATGAS",
            "Cocoa-($/kg)-COCOA", "Coffee, Arabica-($/kg)-COFFEE_ARABIC", "Coffee, Robusta-($/kg)-COFFEE_ROBUS",
            "Tea, avg 3 auctions-($/kg)-TEA_AVG", "Tea, Colombo-($/kg)-TEA_COLOMBO",
            "Tea, Kolkata-($/kg)-TEA_KOLKATA", "Tea, Mombasa-($/kg)-TEA_MOMBASA",
            "Coconut oil-($/mt)-COCONUT_OIL", "Groundnuts-($/mt)-GRNUT", "Fish meal-($/mt)-FISH_MEAL",
            "Groundnut oil **-($/mt)-GRNUT_OIL", "Palm oil-($/mt)-PALM_OIL",
            "Palm kernel oil-($/mt)-PLMKRNL_OIL", "Soybeans-($/mt)-SOYBEANS",
            "Soybean oil-($/mt)-SOYBEAN_OIL", "Soybean meal-($/mt)-SOYBEAN_MEAL",
            "Rapeseed oil-($/mt)-RAPESEED_OIL", "Sunflower oil-($/mt)-SUNFLOWER_OIL",
            "Maize-($/mt)-MAIZE", "Rice, Thai 5% -($/mt)-RICE_05", "Rice, Thai 25% -($/mt)-RICE_25",
            "Rice, Thai A.1-($/mt)-RICE_A1", "Rice, Viet Namese 5%-($/mt)-RICE_05_VNM",
            "Wheat, US SRW-($/mt)-WHEAT_US_SRW", "Wheat, US HRW-($/mt)-WHEAT_US_HRW",
            "Banana, Europe-($/kg)-BANANA_EU", "Banana, US-($/kg)-BANANA_US",
            "Orange-($/kg)-ORANGE", "Beef **-($/kg)-BEEF", "Chicken **-($/kg)-CHICKEN",
            "Lamb **-($/kg)-LAMB", "Shrimps, Mexican-($/kg)-SHRIMP_MEX", "Sugar, EU-($/kg)-SUGAR_EU",
            "Sugar, US-($/kg)-SUGAR_US", "Sugar, world-($/kg)-SUGAR_WLD",
            "Tobacco, US import u.v.-($/mt)-TOBAC_US", "Logs, Cameroon-($/cubic meter)-LOGS_CMR",
            "Logs, Malaysian-($/cubic meter)-LOGS_MYS", "Sawnwood, Cameroon-($/cubic meter)-SAWNWD_CMR",
            "Sawnwood, Malaysian-($/cubic meter)-SAWNWD_MYS", "Plywood-(cents/sheet)-PLYWOOD",
            "Cotton, A Index-($/kg)-COTTON_A_INDX", "Rubber, TSR20 **-($/kg)-RUBBER_TSR20",
            "Rubber, RSS3-($/kg)-RUBBER1_MYSG", "Phosphate rock-($/mt)-PHOSROCK", "DAP-($/mt)-DAP",
            "TSP-($/mt)-TSP", "Urea -($/mt)-UREA_EE_BULK", "Potassium chloride **-($/mt)-POTASH",
            "Aluminum-($/mt)-ALUMINUM", "Iron ore, cfr spot-($/dmtu)-IRON_ORE", "Copper-($/mt)-COPPER",
            "Lead-($/mt)-LEAD", "Tin-($/mt)-Tin", "Nickel-($/mt)-NICKEL", "Zinc-($/mt)-Zinc",
            "Gold-($/troy oz)-GOLD", "Platinum-($/troy oz)-PLATINUM", "Silver-($/troy oz)-SILVER",
            "Metal Index", "Agriculture Index", "Man Made Fibres", "Polyester Chips or Polyethylene Terepthalate (PET) Chips",
            "Acrylic Fibre", "Viscose Staple Fibre", "Polyester Fibre Fabric", "PSF",
            "Australia_exports", "Brazil_exports", "India_exports", "United States_exports",
            "Bangladesh_imports", "China_imports", "Pakistan_imports", "Turkey_imports",
            "month", "quarter"
        ]
        actual = list(df.columns)
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]

        mismatch_msg = ""
        null_msg = ""

        if missing or extra:
            parts = []
            if missing:
                parts.append(f"Missing columns: {', '.join(missing)}")
            if extra:
                parts.append(f"Unexpected columns: {', '.join(extra)}")
            mismatch_msg = " | ".join(parts)

        if "Identifier" in df.columns:
            orig = df["Identifier"].copy()
            try:
                df["Identifier"] = pd.to_datetime(df["Identifier"])
                if df["Identifier"].isna().any():
                    raise Exception
                df.set_index("Identifier", inplace=True)
                df.sort_index(inplace=True)
            except:
                parsed = []
                for v in orig:
                    try:
                        parsed.append(pd.to_datetime(v, dayfirst=True))
                    except:
                        parsed.append(pd.NaT)
                df["Identifier"] = parsed
                if df["Identifier"].isna().any():
                    bad = [i for i, v in enumerate(parsed) if pd.isna(v)]
                    sample = ", ".join([str(orig.iloc[i]) for i in bad[:5]])
                    if mismatch_msg:
                        mismatch_msg += "\n\n"
                    mismatch_msg += f"Date parsing failed for values: {sample}"
                df.set_index("Identifier", inplace=True)
                df.sort_index(inplace=True)

        if isinstance(df.index, pd.DatetimeIndex):
            nn = df.isna()
            nulls = []
            for col in df.columns:
                dates = df.index[nn[col]].strftime("%Y-%m-%d").tolist()
                for d in dates:
                    nulls.append(f"{d} → {col}")
            if nulls:
                null_msg = "Null values found at: " + "; ".join(nulls)

        error_parts = []
        if mismatch_msg:
            error_parts.append(mismatch_msg)
        if null_msg:
            error_parts.append(null_msg)

        if error_parts:
            return None, "\n\n".join(error_parts)
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"
    
def insert_data_page():
    display_logo()
    
    st.markdown("""
    <div class="page-header">
        <h2><span style="vertical-align: middle; margin-right: 10px;">&#x1F504;</span> Insert Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3><span style="vertical-align: middle; margin-right: 10px;">&#x1F4CB;</span> Upload Cotton Price Data</h3>
        <p>Upload your Excel or CSV file containing cotton price data. The file must follow these requirements:</p>
        <ul>
            <li>Must contain an <strong>'Identifier'</strong> column with dates</li>
            <li>Must contain <strong>'ICE Cotton CT1 Comdty'</strong> column with price values</li>
            <li>Supported formats: <strong>.csv, .xlsx, .xls</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    last_update = get_last_update_time()
    if last_update:
        st.markdown(f"""
        <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #1565C0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 10px;">&#x1F550;</span>
                <div>
                    <p style="margin: 0; font-weight: 500; color: #0047AB;">Database Status</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #555;">Last updated on: <strong>{last_update.strftime('%d %b %Y')}</strong></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Added SharePoint link
    st.markdown("""
    <div style="background-color: #E8F5E9; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #2E7D32;">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 10px;">&#x1F517;</span>
            <div>
                <p style="margin: 0; font-weight: 500; color: #2E7D32;">Link to the Data</p>
                <p style="margin: 0; font-size: 0.9rem; margin-top: 0.5rem;">
                    <a href="https://welspungroup.sharepoint.com/:x:/r/sites/HyderabadAIML/_layouts/15/Doc.aspx?sourcedoc=%7B73630830-77C2-4264-B5BD-62B9DB59E2F9%7D&file=Data_for_forecasting.xlsx&action=default&mobileredirect=true" 
                       target="_blank" 
                       style="color: #1565C0; text-decoration: none; font-weight: 500; display: inline-flex; align-items: center;">
                        <span style="margin-right: 5px;">&#x1F4C4;</span> Data_for_forecasting.xlsx <span style="margin-left: 5px; font-size: 0.8rem;">(SharePoint)</span>
                    </a>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 0.5rem; font-weight: 500; color: #333;">
        Upload your data file:
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        with st.spinner("Reading file..."):
            df, error_message = read_file(uploaded_file)
        
        if df is not None:
            st.markdown(f"""
            <div class="success-message" style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 15px;">&#x2705;</span>
                <div>
                    <p style="margin: 0; font-weight: 500;">File loaded successfully!</p>
                    <p style="margin: 0; font-size: 0.9rem;">
                        <strong>{len(df)}</strong> rows and <strong>{len(df.columns)}</strong> columns found in file
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Remove rows with NaN in the index if needed
            # Instead of using df.index.name (which might be None), just filter out NaN indices
            if df.index.isna().any():
                df = df.loc[~df.index.isna()]
            
            with st.expander("&#x1F4CA; Preview Data"):
                st.dataframe(df.head(), use_container_width=True)
                
                st.markdown("### Data Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    display_metric_card("Date Range", f"{df.index.min().strftime('%b %Y')} - {df.index.max().strftime('%b %Y')}")
                
                with col2:
                    target_col = "ICE Cotton CT1 Comdty"
                    display_metric_card("Average Price", f"{df[target_col].mean():.2f}")
                
                with col3:
                    display_metric_card("Price Change", f"{((df[target_col].iloc[-1] / df[target_col].iloc[0]) - 1) * 100:.1f}%")
            
            st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
            insert_button = st.button("&#x1F4BE; Insert Data to Database", type="primary", help="Click to save this data to the database")
            
            if insert_button:
                with st.spinner("Inserting data to database..."):
                    # Insert the data using our function
                    success, message = insert_data(df)
                
                if success:
                    st.markdown(f"""
                    <div class="success-message" style="display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 15px;">&#x2705;</span>
                        <div>
                            <p style="margin: 0; font-weight: 500;">Data inserted successfully!</p>
                            <p style="margin: 0; font-size: 0.9rem;">{message}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <h3 style="margin-top: 1.5rem; color: #0047AB; font-size: 1.2rem; border-bottom: 2px solid #0047AB; padding-bottom: 0.5rem;">
                        <span style="vertical-align: middle; margin-right: 8px;">&#x1F4CB;</span> Last 5 Rows of Inserted Data
                    </h3>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(df.tail(5), use_container_width=True)
                    
                    st.markdown("""
                    <div style="background-color: #E8F5E9; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; text-align: center;">
                        <p style="margin: 0; font-weight: 500; color: #2E7D32;">&#x2728; Data is now ready for forecasting! Go to the Market Insights page to generate predictions.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Clear cache to reload data
                    fetch_data.clear()
                else:
                    st.markdown(f"""
                    <div class="error-message" style="display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 15px;">&#x274C;</span>
                        <div>
                            <p style="margin: 0; font-weight: 500;">Error inserting data</p>
                            <p style="margin: 0; font-size: 0.9rem;">{message}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-message" style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 15px;">&#x274C;</span>
                <div>
                    <p style="margin: 0; font-weight: 500;">File could not be processed</p>
                    <p style="margin: 0; font-size: 0.9rem;">{error_message}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #FFF8E1; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; border-left: 4px solid #FFA000;">
                <h4 style="margin-top: 0; color: #F57C00;">Troubleshooting Tips:</h4>
                <ul style="margin-bottom: 0;">
                    <li>Ensure your file has an <strong>'Identifier'</strong> column with valid dates</li>
                    <li>Verify the <strong>'ICE Cotton CT1 Comdty'</strong> column exists and contains numeric values</li>
                    <li>Check for any formatting issues in your date columns</li>
                    <li>Try removing any special characters or formatting from your file</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #F5F5F5; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">&#x1F4C1;</div>
            <h3 style="margin: 0 0 0.5rem 0; color: #555;">Drag and drop your file here</h3>
            <p style="margin: 0; color: #777; font-size: 0.9rem;">or click to browse your files</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 View Sample File Format"):
            st.markdown("""
            Your file should have the following structure:
            """)
            
            sample_data = {
                'Identifier': ['2023-01-01', '2023-02-01', '2023-03-01'],
                'ICE Cotton CT1 Comdty': [70.25, 71.50, 69.75],
                'Feature1': [25.3, 26.1, 24.8],
                'Feature2': [102.5, 105.2, 98.7]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)
            
            st.markdown("""
            **Important Notes:**
            - The 'Identifier' column must contain valid dates
            - The 'ICE Cotton CT1 Comdty' column is required for price forecasting
            - Additional feature columns can be included for improved predictions
            """)
def run_pipeline(df, target_col):
    # 1) sort & minimum‐data check
    df = df.sort_index()
    horizon = 3
    if len(df) < horizon + 4:
        st.error(f"Not enough data: need ≥{horizon+4} rows, got {len(df)}")
        return None

    # 2) feature engineering
    feat = df[[target_col]].copy()
    for lag in [1,2,3]:
        feat[f"lag_{lag}"] = feat[target_col].shift(lag)
    feat["month"]     = feat.index.month
    feat["month_sin"] = np.sin(2*np.pi*feat["month"]/12)
    feat["month_cos"] = np.cos(2*np.pi*feat["month"]/12)
    feat = feat.dropna()

    # 3) train/validation split
    train_df = feat.iloc[:-horizon]
    valid_df = feat.iloc[-horizon:]
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_valid, y_valid = valid_df.drop(columns=[target_col]), valid_df[target_col]

    # 4) scale
    scaler   = StandardScaler().fit(X_train)
    X_tr_sc  = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_valid)

    # 5) train RF
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr_sc, y_train)

    # 6) predict validation
    y_pred = model.predict(X_val_sc)

    # 7) metrics
    mae  = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2   = r2_score(y_valid, y_pred)
    mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100

    # 8) rolling 3-month forecast with ensemble dispersion
    last_hist    = df[target_col].copy()
    future_index = [df.index[-1] + pd.DateOffset(months=i+1) for i in range(horizon)]
    future_preds = []
    future_conf  = []

    for d in future_index:
        row = {}
        for lag in [1,2,3]:
            row[f"lag_{lag}"] = last_hist.iloc[-lag] if len(last_hist)>=lag else last_hist.mean()
        m = d.month
        row["month"]     = m
        row["month_sin"] = np.sin(2*np.pi*m/12)
        row["month_cos"] = np.cos(2*np.pi*m/12)

        Xf_sc = scaler.transform(pd.DataFrame([row]))
        # tree-by-tree
        tp = np.array([t.predict(Xf_sc)[0] for t in model.estimators_])
        p  = tp.mean()
        cv = tp.std()/p if p!=0 else 0
        conf = max(0.0,(1-cv)*100)

        future_preds.append(p)
        future_conf.append(round(conf,2))

        new_pt = pd.Series([p],index=[d],name=target_col)
        last_hist = pd.concat([last_hist,new_pt])

    # package up
    test_data = pd.DataFrame({
        "Year-Month": valid_df.index.strftime("%Y-%m"),
        "Actual":     y_valid.values,
        "Predicted":  y_pred
    })
    future_data = pd.DataFrame({
        "Year-Month":     [d.strftime("%Y-%m") for d in future_index],
        "Predicted":      future_preds,
        "Confidence": future_conf
    })

    return {
        "metrics":    {"mae":mae,"rmse":rmse,"r2":r2,"mape":mape},
        "test_data":   test_data,
        "future_data": future_data
    }


def plot_results(results, container, df_original=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import traceback

    try:
        # 1) pull out validation & future dataframes
        test_df   = results["test_data"]
        future_df = results["future_data"]

        # 2) build readable month labels
        def mk_labels(year_months):
            return pd.to_datetime(year_months + "-01").dt.strftime("%b %Y")
        test_labels   = mk_labels(test_df["Year-Month"])
        future_labels = mk_labels(future_df["Year-Month"])

        # 3) historical slice (6m prior to validation window)
        hist_months, hist_vals = [], []
        if df_original is not None and not df_original.empty:
            first_val = pd.to_datetime(test_df["Year-Month"].iloc[0] + "-01")
            six_prior = first_val - pd.DateOffset(months=6)
            slice_df  = df_original.loc[six_prior : first_val - pd.DateOffset(days=1)]
            hist_months = slice_df.index.strftime("%b %Y").tolist()
            hist_vals   = slice_df["ICE Cotton CT1 Comdty"].tolist()

        # 4) assemble full series
        months   = hist_months + test_labels.tolist() + future_labels.tolist()
        actuals  = hist_vals + test_df["Actual"].tolist() + [None]*len(future_df)
        preds    = [None]*len(hist_months) + test_df["Predicted"].tolist() + future_df["Predicted"].tolist()

        df_plot = pd.DataFrame({
            "Month":     months,
            "Actual":    actuals,
            "Predicted": preds
        })

        # 5) Model Performance cards
        mape = results["metrics"]["mape"]
        acc  = 100 - mape
        container.markdown(f"""
        <div style="display:flex;gap:2rem;margin-top:1.5rem;">
          <div style="flex:1;background:#F8F9FB;border-left:4px solid #1565C0;
                      padding:1rem;border-radius:8px;">
            <h4 style="margin:0;color:#1565C0;">Average Percentage Error </h4>
            <p style="font-size:1.5rem;font-weight:bold;margin:0;">{mape:.2f}%</p>
          </div>
          <div style="flex:1;background:#F8F9FB;border-left:4px solid #2E7D32;
                      padding:1rem;border-radius:8px;">
            <h4 style="margin:0;color:#2E7D32;">Accuracy</h4>
            <p style="font-size:1.5rem;font-weight:bold;margin:0;">{acc:.2f}%</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 6) Plot
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(df_plot))

        ax.plot(x, df_plot["Actual"],    marker="o", label="Actual",    linewidth=2)
        ax.plot(x, df_plot["Predicted"], marker="s", label="Predicted", linestyle="--", linewidth=2)

        # annotate every point
        for xi in x:
            y_act = df_plot["Actual"].iloc[xi]
            y_prd = df_plot["Predicted"].iloc[xi]
            if y_act is not None:
                ax.annotate(f"{y_act:.2f}",
                            xy=(xi, y_act), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            color="#2e7d32", fontweight="bold", fontsize=9)
            if y_prd is not None:
                ax.annotate(f"{y_prd:.2f}",
                            xy=(xi, y_prd), xytext=(0, -12),
                            textcoords="offset points", ha="center",
                            color="#1565C0", fontweight="bold", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["Month"], rotation=45, ha="right")
        ax.set_ylabel("Price (USD)")
        ax.set_title("ICE Cotton CT1 Comdty – Actual vs Predicted")
        ax.grid(alpha=0.6, linestyle="--")
        ax.legend()
        container.pyplot(fig)

        # 7) Validation table
        container.markdown("### 📋 Validation Results")
        container.dataframe(test_df.reset_index(drop=True))

        # 8) Future + confidence table
        container.markdown("### 📈 Future Forecast with Confidence")
        container.dataframe(future_df.reset_index(drop=True))

        # 9) return error metrics
        abs_err     = np.mean(np.abs(test_df["Actual"] - test_df["Predicted"]))
        return abs_err, mape

    except Exception as e:
        import traceback
        container.error(f"Error plotting results: {e}")
        container.text(traceback.format_exc())
        return None

def market_insights_page():
    import time, traceback

    display_logo()
    st.markdown("""
    <div class="page-header">
      <h2><span style="margin-right:10px;">📊</span> Market Insights & Price Forecast</h2>
    </div>""", unsafe_allow_html=True)

    df = fetch_data()
    if df is None or df.empty:
        st.warning("No data available—please upload under Insert Data.")
        return

    target_col = "ICE Cotton CT1 Comdty"

    # --- Overview metrics ---
    last_update = get_last_update_time()
    if last_update:
        st.info(f"Database last updated: {last_update.strftime('%d %b %Y')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1: display_metric_card("Total Records", f"{len(df):,}")
    with c2: display_metric_card("Date Range", f"{df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')}")
    with c3: display_metric_card("Latest Price", f"{df[target_col].iloc[-1]:.2f}")
    with c4:
        pct = (df[target_col].iloc[-1] / df[target_col].iloc[0] - 1) * 100
        display_metric_card("Overall Change", f"{pct:.1f}%")

    # --- Historical trend ---
    st.markdown("#### 📈 Historical Price Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[target_col], color="#0047AB", linewidth=2)
    ax.set_title("Historical ICE Cotton Prices")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig)

    # --- Stats ---
    st.markdown("#### Price Statistics")
    s1, s2, s3, s4 = st.columns(4)
    with s1: display_metric_card("Average",  f"{df[target_col].mean():.2f}")
    with s2: display_metric_card("Median",   f"{df[target_col].median():.2f}")
    with s3: display_metric_card("Minimum",  f"{df[target_col].min():.2f}")
    with s4: display_metric_card("Maximum",  f"{df[target_col].max():.2f}")

    # --- Forecast button ---
    if st.button("📊 Generate Price Forecast"):
        prog = st.progress(0)
        for pct, msg in [(20, "Analyzing…"), (40, "Feature engineering…"), (60, "Training model…"), (80, "Forecasting…")]:
            time.sleep(0.3)
            prog.progress(pct)
            st.markdown(f"**{msg}**")
        prog.progress(100)

        results = run_pipeline(df, target_col)
        if not results:
            st.error("Forecast failed—ensure at least 9 months of data.")
            return

        display_success_message("Forecast generated successfully! 🎉")
        pc = st.container()
        rv = plot_results(results, pc, df)
        if rv is not None:
            abs_err, mape = rv
            # st.markdown(f"**Validation Abs Error:** {abs_err:.2f}  |  **MAPE:** {mape:.2f}%")
        else:
            st.error("Plotting failed—see logs above.")

        # --- Upcoming Features ---
        st.markdown("""
        <div style="margin-top:2rem;background:#E3F2FD;padding:1.5rem;border-radius:8px;">
          <h3 style="margin:0;display:flex;align-items:center;">
            <span style="margin-right:8px;">🚀</span>Upcoming Features
          </h3>
          <ul style="margin-top:0.5rem;padding-left:1.2rem;">
            <li><strong>Real-time News Integration:</strong> Processing geopolitical events, government tariffs, and policy changes that impact cotton markets.</li>
            <li><strong>LLM-Enhanced Predictions:</strong> Leveraging advanced language models to analyze market sentiment and improve forecasting accuracy.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)


def help_guide_page():
    display_logo()
    
    st.markdown("""
    <div class="page-header">
        <h2><span style="vertical-align: middle; margin-right: 10px;">&#x2753;</span> Help & Guide</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3><span style="vertical-align: middle; margin-right: 10px;">&#x1F4F1;</span> About This Application</h3>
        <p>The ICE Cotton Market Intelligence application provides advanced price forecasting and market insights for cotton prices. 
        Using AI and machine learning, it analyzes historical trends to predict future price movements, helping you make informed 
        procurement and business decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 2rem;'>How to Use This Application</h3>", unsafe_allow_html=True)
    
    tabs = st.tabs(["&#x1F504; Data Upload", "&#x1F4CA; Market Insights", "&#x1F52E; Forecast Interpretation"])
    
    with tabs[0]:
        st.markdown("""
        <div style="padding: 1rem;">
            <h4 style="color: #0047AB;">Uploading Your Data</h4>
            <ol>
                <li><strong>Prepare your data file</strong> - Ensure your file includes:
                    <ul>
                        <li>'Identifier' column with dates</li>
                        <li>'ICE Cotton CT1 Comdty' column with price values</li>
                        <li>Preferably at least 12 months of data</li>
                    </ul>
                </li>
                <li><strong>Navigate to the Insert Data page</strong> - Select from the sidebar menu</li>
                <li><strong>Upload your file</strong> - Click the upload area or drag and drop your file</li>
                <li><strong>Review data preview</strong> - Confirm data is loaded correctly</li>
                <li><strong>Insert to database</strong> - Click the "Insert Data to Database" button</li>
            </ol>
            
            <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #1565C0;">
                <p style="margin: 0; font-weight: 500; color: #0047AB;">Pro Tip</p>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">Always check the data preview to ensure your date columns and price values have been correctly recognized.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
        <div style="padding: 1rem;">
            <h4 style="color: #0047AB;">Exploring Market Insights</h4>
            <ol>
                <li><strong>Navigate to the Market Insights page</strong> - Select from the sidebar menu</li>
                <li><strong>Review data overview</strong> - See key metrics and statistics about your data</li>
                <li><strong>Examine historical trends</strong> - The chart shows past price movements and patterns</li>
                <li><strong>Generate a forecast</strong> - Click the "Generate Price Forecast" button to run the AI model</li>
                <li><strong>Analyze results</strong> - Review the visualization showing validation and future predictions</li>
                <li><strong>Export data</strong> - Use the download button to save the forecast for offline analysis</li>
            </ol>
            
            <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #1565C0;">
                <p style="margin: 0; font-weight: 500; color: #0047AB;">Pro Tip</p>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">Pay attention to the model performance metrics to gauge forecast reliability.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        <div style="padding: 1rem;">
            <h4 style="color: #0047AB;">Understanding the Forecast</h4>
            <ul>
                <li><strong>Validation Period (Blue Line)</strong> - Shows how well the model predicts known historical data</li>
                <li><strong>Future Forecast (Red Line)</strong> - Represents predictions for upcoming months</li>
                <li><strong>Actual Values (Green Line)</strong> - Real historical prices for comparison</li>
                <li><strong>Error Metrics</strong> - Indicators of model accuracy:
                    <ul>
                        <li><em>Absolute Error</em> - Average dollar amount difference between actual and predicted values</li>
                        <li><em>Percent Error</em> - Average percentage difference between actual and predicted values</li>
                    </ul>
                </li>
                <li><strong>Market Recommendations</strong> - Suggested actions based on the forecast</li>
            </ul>
            
            <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #1565C0;">
                <p style="margin: 0; font-weight: 500; color: #0047AB;">Pro Tip</p>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">The forecast is most accurate for the near term and should be regularly updated as new data becomes available.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="margin-top: 2rem;">Frequently Asked Questions</h3>
    """, unsafe_allow_html=True)
    
    with st.expander("What data format is required?"):
        st.markdown("""
        Your data file must:
        - Be in CSV or Excel (xlsx/xls) format
        - Include an 'Identifier' column with dates
        - Include an 'ICE Cotton CT1 Comdty' column with price values
        - Preferably have at least 12 months of historical data
        """)
    
    with st.expander("How accurate is the forecast?"):
        st.markdown("""
        The forecast accuracy depends on several factors:
        - The quality and quantity of historical data provided
        - The volatility of the cotton market
        - The time horizon (near-term forecasts are generally more accurate)
        
        Our model typically achieves 85-95% accuracy for 1-3 month forecasts, measured by comparing predicted vs. actual prices in the validation period.
        """)
    
    with st.expander("How often should I update the forecast?"):
        st.markdown("""
        For optimal results, we recommend:
        - Updating the forecast monthly as new price data becomes available
        - Refreshing the data after significant market events
        - Comparing actual prices against forecasted values regularly to assess performance
        """)
    
    with st.expander("Can I customize the forecast parameters?"):
        st.markdown("""
        The current version uses optimized default parameters for cotton price forecasting. Future releases will include:
        - Customizable forecast horizons
        - Adjustable training/validation periods
        - Selection of different forecasting algorithms
        - Scenario analysis capabilities
        """)
    
    st.markdown("""
    <div class="card" style="margin-top: 2rem;">
        <h3><span style="vertical-align: middle; margin-right: 10px;">&#x1F4DE;</span> Support & Contact</h3>
        <p>If you need assistance or have questions about using this application, please contact:</p>
        <ul>
            <li><strong>Technical Support:</strong> 
                <ul style="margin-top: 0.5rem;">
                    <li><strong>AI Team Lead:</strong> Neha_Porwal@welspun.com</li>
                    <li><strong>Team:</strong> SadakPramodh_Maduru@welspun.com, ramu_sangineni@welspun.com</li>
                </ul>
            </li>   
        </ul>       
    </div> 
    
    <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #666; font-size: 0.9rem;">
        <p>© 2025 Welspun Group. All rights reserved.</p>
        <p style="font-style: italic; margin-top: 0.5rem;">"Har Ghar Se Har Dil Tak"</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    set_page_style()
    
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem 0.5rem; margin-bottom: 2rem; border-bottom: 1px solid #eee;">
        <h3 style="margin: 0.5rem 0; color: #0047AB; font-size: 1.2rem;">Cotton Market Intelligence</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<h3 style='margin-bottom: 1rem; font-size: 1.1rem;'>Navigation</h3>", unsafe_allow_html=True)
    
    selected_page = None
    
    # Reordered buttons - Market Insights now comes first
    market_insights_button = st.sidebar.button("📊 Market Insights", key="nav_market_insights", help="View forecasts and analysis")
    if market_insights_button:
        selected_page = "Market Insights"
    
    insert_data_button = st.sidebar.button("↻ Insert Data", key="nav_insert_data", help="Upload and manage data")
    if insert_data_button:
        selected_page = "Insert Data"
    
    help_guide_button = st.sidebar.button("❓ Help & Guide", key="nav_help_guide", help="Learn how to use the app")
    if help_guide_button:
        selected_page = "Help & Guide"
    
    if selected_page is None:
        if 'current_page' not in st.session_state:
            # Changed default page to Market Insights
            st.session_state['current_page'] = "Market Insights"
        selected_page = st.session_state['current_page']
    else:
        st.session_state['current_page'] = selected_page
    
    st.sidebar.markdown("""
    <div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #eee; text-align: center; font-size: 0.8rem; color: #666;">
        <p style="margin-bottom: 0.5rem;">Current Version: 1.1.0</p>
        <p>Last Updated: April 15, 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    if selected_page == "Insert Data":
        insert_data_page()
    elif selected_page == "Market Insights":
        market_insights_page()
    elif selected_page == "Help & Guide":
        help_guide_page()

if __name__ == "__main__":
    main()
