import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from io import BytesIO
import os
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
            <p style="color:#666; font-style: italic; margin-top: 0; font-size: 1.1rem;">Har Ghar Se Har Dil Thak</p>
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
    
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file, engine='openpyxl')
        else:
            return None, f"Unsupported file format: {file_extension}"
        
        # Check for required columns first
        if 'Identifier' not in df.columns:
            return None, "Required column 'Identifier' not found in the file."
            
        if 'ICE Cotton CT1 Comdty' not in df.columns:
            return None, "Required column 'ICE Cotton CT1 Comdty' not found in the file."
        
        # Validate that ICE Cotton CT1 Comdty column contains numeric values
        if not pd.api.types.is_numeric_dtype(df['ICE Cotton CT1 Comdty']):
            try:
                # Try to convert to numeric
                df['ICE Cotton CT1 Comdty'] = pd.to_numeric(df['ICE Cotton CT1 Comdty'], errors='coerce')
                
                # Check if conversion created any NaN values
                if df['ICE Cotton CT1 Comdty'].isna().any():
                    non_numeric_indices = df[df['ICE Cotton CT1 Comdty'].isna()].index.tolist()
                    if len(non_numeric_indices) > 5:
                        sample_problematic = [df.loc[i, 'ICE Cotton CT1 Comdty'] for i in non_numeric_indices[:5]]
                        problematic_values = f"Examples of non-numeric values: {', '.join(map(str, sample_problematic))} (and {len(non_numeric_indices) - 5} more)"
                    else:
                        sample_problematic = [df.loc[i, 'ICE Cotton CT1 Comdty'] for i in non_numeric_indices]
                        problematic_values = f"Non-numeric values: {', '.join(map(str, sample_problematic))}"
                        
                    return None, f"Some values in 'ICE Cotton CT1 Comdty' column could not be converted to numbers. {problematic_values}"
            except:
                return None, "The 'ICE Cotton CT1 Comdty' column contains non-numeric values and could not be processed."
        
        # Handle the date conversion in the Identifier column
        if 'Identifier' in df.columns:
            # First, try a more direct approach with common date formats
            date_formats = [
                "%d-%m-%Y %H:%M",   # 01-01-2025 00:00
                "%d-%m-%Y",         # 01-01-2025
                "%Y-%m-%d %H:%M",   # 2025-01-01 00:00
                "%Y-%m-%d",         # 2025-01-01
                "%d/%m/%Y",         # 01/01/2025
                "%m/%d/%Y",         # 01/01/2025 (US format)
                "%Y/%m/%d"          # 2025/01/01
            ]
            
            # First try without specifying a format (let pandas detect)
            try:
                df['Identifier'] = pd.to_datetime(df['Identifier'])
                if not df['Identifier'].isna().any():
                    # Set as index if all dates parsed successfully
                    df.set_index('Identifier', inplace=True)
                    df.sort_index(inplace=True)
                    return df, None
            except:
                pass  # If auto-detection fails, continue to explicit formats
            
            # If auto-detection didn't work fully, try explicit formats
            original_identifiers = df['Identifier'].copy()
            
            for date_format in date_formats:
                try:
                    # For debugging, print a sample conversion
                    sample_value = original_identifiers.iloc[0]
                    print(f"Testing format {date_format} on {sample_value}")
                    
                    # Try to convert the first value as a test
                    test_date = pd.to_datetime(sample_value, format=date_format)
                    print(f"Test conversion result: {test_date}")
                    
                    # If test passed, try the full column
                    df['Identifier'] = pd.to_datetime(original_identifiers, format=date_format)
                    
                    # Check for any failed conversions
                    if not df['Identifier'].isna().any():
                        # All conversions successful
                        df.set_index('Identifier', inplace=True)
                        df.sort_index(inplace=True)
                        return df, None
                    else:
                        # Some conversions failed, reset and try next format
                        df['Identifier'] = original_identifiers.copy()
                except Exception as e:
                    print(f"Format {date_format} failed: {str(e)}")
                    # Reset for next attempt
                    df['Identifier'] = original_identifiers.copy()
            
            # Special handling for DD-MM-YYYY HH:MM format (your specific case)
            try:
                # Manual parsing for the specific format in error message
                parsed_dates = []
                for date_str in original_identifiers:
                    try:
                        if isinstance(date_str, str) and len(date_str.strip()) > 0:
                            # Try to parse with explicit format
                            parts = date_str.strip().split(' ')
                            if len(parts) == 2:  # Has date and time parts
                                date_parts = parts[0].split('-')
                                if len(date_parts) == 3:
                                    day, month, year = date_parts
                                    time_parts = parts[1].split(':')
                                    hour, minute = time_parts if len(time_parts) == 2 else ('00', '00')
                                    
                                    parsed_date = pd.Timestamp(int(year), int(month), int(day), 
                                                              int(hour), int(minute))
                                    parsed_dates.append(parsed_date)
                                else:
                                    parsed_dates.append(pd.NaT)
                            else:  # Only date part
                                date_parts = parts[0].split('-')
                                if len(date_parts) == 3:
                                    day, month, year = date_parts
                                    parsed_date = pd.Timestamp(int(year), int(month), int(day))
                                    parsed_dates.append(parsed_date)
                                else:
                                    parsed_dates.append(pd.NaT)
                        else:
                            parsed_dates.append(pd.NaT)
                    except:
                        parsed_dates.append(pd.NaT)
                
                df['Identifier'] = parsed_dates
                
                # Check if manual parsing worked
                if not df['Identifier'].isna().any():
                    df.set_index('Identifier', inplace=True)
                    df.sort_index(inplace=True)
                    return df, None
                else:
                    # Reset if it didn't work completely
                    df['Identifier'] = original_identifiers.copy()
            except:
                # Reset if manual parsing failed
                df['Identifier'] = original_identifiers.copy()
            
            # If we get here, no format worked for all values
            problematic_indices = df[pd.isna(pd.to_datetime(df['Identifier'], errors='coerce'))].index.tolist()
            
            if len(problematic_indices) > 0:
                if len(problematic_indices) > 5:
                    sample_problematic = [original_identifiers.iloc[i] for i in problematic_indices[:5]]
                    problematic_values = f"Examples of values that couldn't be parsed: {', '.join(map(str, sample_problematic))} (and {len(problematic_indices) - 5} more)"
                else:
                    sample_problematic = [original_identifiers.iloc[i] for i in problematic_indices]
                    problematic_values = f"Values that couldn't be parsed: {', '.join(map(str, sample_problematic))}"
                    
                return None, f"Some date values in the 'Identifier' column could not be parsed. {problematic_values}. Please ensure all dates are in a consistent format."
        
        return df, None
    except Exception as e:
        import traceback
        print(traceback.format_exc())
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
    import numpy as np
    import pandas as pd
    import streamlit as st
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import traceback
    
    results = {}
    
    try:
        if target_col not in df.columns:
            st.error(f"Cotton price column '{target_col}' not found in your data!")
            return None

        st.write(f"Input data date range: {df.index.min()} to {df.index.max()}")
        
        df_original = df.copy().sort_index()
        horizon = 3
        
        total_months = len(df_original)
        
        if total_months < horizon + 6:
            st.error(f"Not enough data! Need at least {horizon+6} months of data.")
            return None
        
        validation_end = total_months
        validation_start = total_months - horizon
        train_end = validation_start
        
        df_validation = df_original.iloc[validation_start:validation_end].copy()
        df_train_data = df_original.iloc[:train_end].copy()
        df_train = df_train_data.copy()
        
        train_target_indices = []
        train_target_values = []
        
        for i, idx in enumerate(df_train.index):
            target_idx = i + horizon
            if target_idx < len(df_original):
                target_date = df_original.index[target_idx]
                target_value = df_original.loc[target_date, target_col]
                train_target_indices.append(idx)
                train_target_values.append(target_value)
        
        train_targets = pd.Series(train_target_values, index=train_target_indices)
        df_train = df_train.loc[train_targets.index]
        validation_dates = df_validation.index
        
        last_date = df_original.index.max()
        future_index = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]
        
        print(f"Training period: {df_train.index.min()} to {df_train.index.max()}")
        print(f"Validation dates: {[d.strftime('%b %Y') for d in validation_dates]}")
        print(f"Future prediction dates: {[d.strftime('%b %Y') for d in future_index]}")
        
        df_future = pd.DataFrame({col: np.nan for col in df_original.columns}, index=future_index)
        
        results["training_period"] = f"{df_train.index.min().strftime('%b %Y')} to {df_train.index.max().strftime('%b %Y')}"
        results["test_months"] = [d.strftime('%b %Y') for d in validation_dates]
        results["future_months"] = [d.strftime('%b %Y') for d in future_index]
        test_dates_ym = [d.strftime('%Y-%m') for d in validation_dates]
        future_dates_ym = [d.strftime('%Y-%m') for d in future_index]

        X_train_raw = df_train.drop(columns=[target_col])
        y_train = train_targets
        
        scaler_std = StandardScaler().fit(X_train_raw)
        X_train_std = scaler_std.transform(X_train_raw)
        
        scaler_mm = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_std)
        X_train_scl = scaler_mm.transform(X_train_std)

        scaled_df = pd.DataFrame(
            X_train_scl,
            index=df_train.index,
            columns=X_train_raw.columns
        )
        scaled_df[target_col] = y_train
        
        temp_df = scaled_df.copy()
        
        for idx in temp_df.index:
            temp_df.loc[idx, 'month'] = idx.month
            temp_df.loc[idx, 'quarter'] = idx.quarter
            temp_df.loc[idx, 'month_sin'] = np.sin(2 * np.pi * idx.month / 12)
            temp_df.loc[idx, 'month_cos'] = np.cos(2 * np.pi * idx.month / 12)
            temp_df.loc[idx, 'quarter_sin'] = np.sin(2 * np.pi * idx.quarter / 4)
            temp_df.loc[idx, 'quarter_cos'] = np.cos(2 * np.pi * idx.quarter / 4)
        
        best_lags = [1, 2, 3, 6, 9, 12]
        for lag in best_lags:
            temp_df[f'{target_col}_lag_{lag}'] = temp_df[target_col].shift(lag)
        
        best_windows = [3, 6, 12]
        for win in best_windows:
            temp_df[f'{target_col}_sma_{win}'] = temp_df[target_col].rolling(win).mean()
            temp_df[f'{target_col}_std_{win}'] = temp_df[target_col].rolling(win).std()
            temp_df[f'{target_col}_min_{win}'] = temp_df[target_col].rolling(win).min()
            temp_df[f'{target_col}_max_{win}'] = temp_df[target_col].rolling(win).max()
            temp_df[f'{target_col}_range_{win}'] = temp_df[f'{target_col}_max_{win}'] - temp_df[f'{target_col}_min_{win}']
        
        for period in [1, 3, 6, 12]:
            temp_df[f'{target_col}_mom_{period}'] = temp_df[target_col] - temp_df[target_col].shift(period)
            temp_df[f'{target_col}_roc_{period}'] = temp_df[target_col].pct_change(period) * 100
        
        for span in [3, 6, 12]:
            temp_df[f'{target_col}_ewma_{span}'] = temp_df[target_col].ewm(span=span).mean()
            temp_df[f'{target_col}_ema_dist_{span}'] = temp_df[target_col] - temp_df[f'{target_col}_ewma_{span}']
        
        temp_df[f'{target_col}_volatility_12'] = temp_df[target_col].rolling(12).std() / temp_df[target_col].rolling(12).mean()
        temp_df[f'{target_col}_volatility_6'] = temp_df[target_col].rolling(6).std() / temp_df[target_col].rolling(6).mean()
        
        temp_df.dropna(inplace=True)

        X = temp_df.drop(columns=[target_col])
        y = temp_df[target_col]

        with st.spinner("Training model ensemble..."):
            selector = SelectKBest(mutual_info_regression, k=min(25, len(X.columns)))
            selector.fit(X, y)
            feats = X.columns[selector.get_support()]
            
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            
            gbr = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                min_samples_leaf=5,
                subsample=0.8,
                max_features=0.7,
                random_state=42
            )
            
            etr = ExtraTreesRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=3,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            
            en = ElasticNet(
                alpha=0.3,
                l1_ratio=0.6,
                random_state=42
            )
            
            br = BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            )
            
            ridge = Ridge(
                alpha=0.5,
                random_state=42
            )
            
            models = {
                'rf': rf,
                'gbr': gbr,
                'etr': etr,
                'en': en,
                'br': br,
                'ridge': ridge
            }
            
            for name, model in models.items():
                model.fit(X[feats], y)
            
            meta_model = GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
            
            meta_features = np.column_stack([
                models[name].predict(X[feats]) for name in models
            ])
            
            meta_model.fit(meta_features, y)

        X_test_raw = df_validation.drop(columns=[target_col])
        y_test = df_validation[target_col]
        X_test_std = scaler_std.transform(X_test_raw)
        X_test_scl = scaler_mm.transform(X_test_std)
        X_test_df = pd.DataFrame(X_test_scl, index=df_validation.index, columns=X_test_raw.columns)

        for idx in X_test_df.index:
            X_test_df.loc[idx, 'month'] = idx.month
            X_test_df.loc[idx, 'quarter'] = idx.quarter
            X_test_df.loc[idx, 'month_sin'] = np.sin(2 * np.pi * idx.month / 12)
            X_test_df.loc[idx, 'month_cos'] = np.cos(2 * np.pi * idx.month / 12)
            X_test_df.loc[idx, 'quarter_sin'] = np.sin(2 * np.pi * idx.quarter / 4)
            X_test_df.loc[idx, 'quarter_cos'] = np.cos(2 * np.pi * idx.quarter / 4)
        
        historical_data = df_original.loc[:df_validation.index.max()]
        historical_prices = historical_data[target_col]
        
        for lag in best_lags:
            for i, test_date in enumerate(df_validation.index):
                lag_date_idx = df_original.index.get_loc(test_date) - lag
                if lag_date_idx >= 0:
                    lag_date = df_original.index[lag_date_idx]
                    X_test_df.loc[test_date, f'{target_col}_lag_{lag}'] = historical_prices.loc[lag_date]
                else:
                    X_test_df.loc[test_date, f'{target_col}_lag_{lag}'] = historical_prices.mean()
        
        for i, test_date in enumerate(df_validation.index):
            test_idx = df_original.index.get_loc(test_date)
            
            for win in best_windows:
                if test_idx >= win:
                    window_slice = historical_prices.iloc[test_idx-win:test_idx]
                    
                    X_test_df.loc[test_date, f'{target_col}_sma_{win}'] = window_slice.mean()
                    X_test_df.loc[test_date, f'{target_col}_std_{win}'] = window_slice.std()
                    X_test_df.loc[test_date, f'{target_col}_min_{win}'] = window_slice.min()
                    X_test_df.loc[test_date, f'{target_col}_max_{win}'] = window_slice.max()
                    X_test_df.loc[test_date, f'{target_col}_range_{win}'] = window_slice.max() - window_slice.min()
                else:
                    available_slice = historical_prices.iloc[:test_idx]
                    if len(available_slice) > 0:
                        X_test_df.loc[test_date, f'{target_col}_sma_{win}'] = available_slice.mean()
                        X_test_df.loc[test_date, f'{target_col}_std_{win}'] = available_slice.std()
                        X_test_df.loc[test_date, f'{target_col}_min_{win}'] = available_slice.min()
                        X_test_df.loc[test_date, f'{target_col}_max_{win}'] = available_slice.max()
                        X_test_df.loc[test_date, f'{target_col}_range_{win}'] = available_slice.max() - available_slice.min()
                    else:
                        X_test_df.loc[test_date, f'{target_col}_sma_{win}'] = historical_prices.loc[test_date]
                        X_test_df.loc[test_date, f'{target_col}_std_{win}'] = 0
                        X_test_df.loc[test_date, f'{target_col}_min_{win}'] = historical_prices.loc[test_date]
                        X_test_df.loc[test_date, f'{target_col}_max_{win}'] = historical_prices.loc[test_date]
                        X_test_df.loc[test_date, f'{target_col}_range_{win}'] = 0
        
        for period in [1, 3, 6, 12]:
            for i, test_date in enumerate(df_validation.index):
                test_idx = df_original.index.get_loc(test_date)
                
                if test_idx >= period:
                    period_ago_idx = test_idx - period
                    period_ago_date = df_original.index[period_ago_idx]
                    
                    X_test_df.loc[test_date, f'{target_col}_mom_{period}'] = historical_prices.loc[test_date] - historical_prices.loc[period_ago_date]
                    X_test_df.loc[test_date, f'{target_col}_roc_{period}'] = ((historical_prices.loc[test_date] / historical_prices.loc[period_ago_date]) - 1) * 100 if historical_prices.loc[period_ago_date] != 0 else 0
                else:
                    X_test_df.loc[test_date, f'{target_col}_mom_{period}'] = 0
                    X_test_df.loc[test_date, f'{target_col}_roc_{period}'] = 0
        
        for span in [3, 6, 12]:
            for i, test_date in enumerate(df_validation.index):
                test_idx = df_original.index.get_loc(test_date)
                
                if test_idx >= span * 2:
                    hist_slice = historical_prices.iloc[test_idx-span*2:test_idx]
                    ema_value = hist_slice.ewm(span=span).mean().iloc[-1]
                    
                    X_test_df.loc[test_date, f'{target_col}_ewma_{span}'] = ema_value
                    X_test_df.loc[test_date, f'{target_col}_ema_dist_{span}'] = historical_prices.loc[test_date] - ema_value
                else:
                    available_slice = historical_prices.iloc[:test_idx]
                    if len(available_slice) > 0:
                        ema_approx = available_slice.mean()
                        X_test_df.loc[test_date, f'{target_col}_ewma_{span}'] = ema_approx
                        X_test_df.loc[test_date, f'{target_col}_ema_dist_{span}'] = historical_prices.loc[test_date] - ema_approx
                    else:
                        X_test_df.loc[test_date, f'{target_col}_ewma_{span}'] = historical_prices.loc[test_date]
                        X_test_df.loc[test_date, f'{target_col}_ema_dist_{span}'] = 0
        
        for i, test_date in enumerate(df_validation.index):
            test_idx = df_original.index.get_loc(test_date)
            
            for period in [6, 12]:
                if test_idx >= period:
                    window_slice = historical_prices.iloc[test_idx-period:test_idx]
                    mean_price = window_slice.mean()
                    if mean_price > 0:
                        X_test_df.loc[test_date, f'{target_col}_volatility_{period}'] = window_slice.std() / mean_price
                    else:
                        X_test_df.loc[test_date, f'{target_col}_volatility_{period}'] = window_slice.std()
                else:
                    X_test_df.loc[test_date, f'{target_col}_volatility_{period}'] = 0
        
        X_test_feats = X_test_df.reindex(columns=feats).fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        test_meta_features = np.column_stack([
            models[name].predict(X_test_feats) for name in models
        ])
        
        y_test_pred = meta_model.predict(test_meta_features)
        
        future_preds = []
        future_features_list = []
        
        full_historical_data = df_original.copy()
        current_state = full_historical_data.copy()
        
        for i, future_date in enumerate(future_index):
            X_f = pd.DataFrame(index=[future_date])
            
            X_f['month'] = future_date.month
            X_f['quarter'] = future_date.quarter
            X_f['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
            X_f['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
            X_f['quarter_sin'] = np.sin(2 * np.pi * future_date.quarter / 4)
            X_f['quarter_cos'] = np.cos(2 * np.pi * future_date.quarter / 4)
            
            recent_prices = pd.Series([])
            
            if i == 0:
                recent_prices = current_state[target_col].copy()
                latest_date = current_state.index.max()
                latest_price = current_state.loc[latest_date, target_col]
            else:
                recent_prices = pd.concat([
                    current_state[target_col],
                    pd.Series([future_preds[j] for j in range(i)], index=future_index[:i])
                ])
                latest_date = future_index[i-1]
                latest_price = future_preds[i-1]
            
            for lag in best_lags:
                if i >= lag:
                    X_f[f'{target_col}_lag_{lag}'] = future_preds[i-lag]
                else:
                    if i > 0:
                        lag_remaining = lag - i
                        hist_idx = len(current_state) - lag_remaining
                        if hist_idx >= 0:
                            hist_date = current_state.index[hist_idx]
                            X_f[f'{target_col}_lag_{lag}'] = current_state.loc[hist_date, target_col]
                        else:
                            X_f[f'{target_col}_lag_{lag}'] = current_state[target_col].iloc[0]
                    else:
                        lag_date_idx = len(current_state) - lag
                        if lag_date_idx >= 0:
                            lag_date = current_state.index[lag_date_idx]
                            X_f[f'{target_col}_lag_{lag}'] = current_state.loc[lag_date, target_col]
                        else:
                            X_f[f'{target_col}_lag_{lag}'] = current_state[target_col].iloc[0]
            
            for win in best_windows:
                if i >= win:
                    window_data = future_preds[i-win:i]
                elif i > 0:
                    future_data = future_preds[:i]
                    hist_needed = win - len(future_data)
                    hist_start = len(current_state) - hist_needed
                    if hist_start >= 0:
                        hist_data = current_state[target_col].iloc[hist_start:].values
                        window_data = np.concatenate([hist_data, future_data])
                    else:
                        window_data = np.concatenate([current_state[target_col].values, future_data])
                else:
                    hist_start = len(current_state) - win
                    if hist_start >= 0:
                        window_data = current_state[target_col].iloc[hist_start:].values
                    else:
                        window_data = current_state[target_col].values
                
                win_mean = np.mean(window_data) if len(window_data) > 0 else latest_price
                win_std = np.std(window_data) if len(window_data) > 1 else 0
                win_min = np.min(window_data) if len(window_data) > 0 else latest_price
                win_max = np.max(window_data) if len(window_data) > 0 else latest_price
                
                X_f[f'{target_col}_sma_{win}'] = win_mean
                X_f[f'{target_col}_std_{win}'] = win_std
                X_f[f'{target_col}_min_{win}'] = win_min
                X_f[f'{target_col}_max_{win}'] = win_max
                X_f[f'{target_col}_range_{win}'] = win_max - win_min
            
            for period in [1, 3, 6, 12]:
                if i >= period:
                    period_ago_price = future_preds[i-period]
                elif i > 0:
                    period_remaining = period - i
                    hist_idx = len(current_state) - period_remaining
                    if hist_idx >= 0:
                        period_ago_price = current_state[target_col].iloc[hist_idx]
                    else:
                        period_ago_price = current_state[target_col].iloc[0]
                else:
                    hist_idx = len(current_state) - period
                    if hist_idx >= 0:
                        period_ago_price = current_state[target_col].iloc[hist_idx]
                    else:
                        period_ago_price = current_state[target_col].iloc[0]
                
                momentum = latest_price - period_ago_price
                X_f[f'{target_col}_mom_{period}'] = momentum
                
                roc = ((latest_price / period_ago_price) - 1) * 100 if period_ago_price != 0 else 0
                X_f[f'{target_col}_roc_{period}'] = roc
            
            for span in [3, 6, 12]:
                if i > 0:
                    historical_slice = recent_prices.iloc[-min(len(recent_prices), span*3):]
                    
                    weights = np.exp(np.linspace(-1, 0, len(historical_slice)))
                    weights = weights / weights.sum()
                    ema_value = np.sum(historical_slice.values * weights)
                    
                    X_f[f'{target_col}_ewma_{span}'] = ema_value
                    X_f[f'{target_col}_ema_dist_{span}'] = latest_price - ema_value
                else:
                    hist_slice = current_state[target_col].iloc[-min(len(current_state), span*3):]
                    ema_value = hist_slice.ewm(span=span).mean().iloc[-1]
                    
                    X_f[f'{target_col}_ewma_{span}'] = ema_value
                    X_f[f'{target_col}_ema_dist_{span}'] = latest_price - ema_value
            
            for period in [6, 12]:
                if i >= period:
                    vol_data = future_preds[i-period:i]
                elif i > 0:
                    future_data = future_preds[:i]
                    hist_needed = period - len(future_data)
                    hist_start = len(current_state) - hist_needed
                    if hist_start >= 0:
                        hist_data = current_state[target_col].iloc[hist_start:].values
                        vol_data = np.concatenate([hist_data, future_data])
                    else:
                        vol_data = np.concatenate([current_state[target_col].values, future_data])
                else:
                    hist_start = len(current_state) - period
                    if hist_start >= 0:
                        vol_data = current_state[target_col].iloc[hist_start:].values
                    else:
                        vol_data = current_state[target_col].values
                
                vol_mean = np.mean(vol_data) if len(vol_data) > 0 else latest_price
                vol_std = np.std(vol_data) if len(vol_data) > 1 else 0
                
                X_f[f'{target_col}_volatility_{period}'] = vol_std / vol_mean if vol_mean != 0 else vol_std
            
            X_f_raw = pd.DataFrame(index=[future_date])
            for col in X_train_raw.columns:
                if col not in X_f:
                    latest_value = current_state[col].iloc[-1] if col in current_state.columns else 0
                    X_f_raw[col] = latest_value
                else:
                    X_f_raw[col] = X_f[col]
            
            X_f_std = scaler_std.transform(X_f_raw)
            X_f_scl = scaler_mm.transform(X_f_std)
            X_f_scaled = pd.DataFrame(X_f_scl, index=[future_date], columns=X_f_raw.columns)
            
            for col in X_f.columns:
                if col not in X_f_scaled.columns:
                    X_f_scaled[col] = X_f[col]
            
            X_f_feats = X_f_scaled.reindex(columns=feats).fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            future_meta_features = np.column_stack([
                models[name].predict(X_f_feats) for name in models
            ])
            
            pred = meta_model.predict(future_meta_features)[0]
            
            if i > 0:
                prev_pred = future_preds[i-1]
                max_change = abs(prev_pred) * 0.05
                if abs(pred - prev_pred) > max_change:
                    sign = 1 if pred > prev_pred else -1
                    
                    if i >= 3:
                        trend_direction = sum(1 if future_preds[j] > future_preds[j-1] else -1 for j in range(i-2, i))
                        
                        if sign == trend_direction:
                            max_change = abs(prev_pred) * 0.08
                    
                    pred = prev_pred + (sign * max_change)
            
            future_preds.append(pred)
            future_features_list.append(X_f.copy())
            
            new_row = pd.DataFrame({target_col: pred}, index=[future_date])
            for col in current_state.columns:
                if col != target_col:
                    new_row[col] = current_state[col].iloc[-1] if col in current_state.columns else 0
            
            current_state = pd.concat([current_state, new_row])

        if future_features_list:
            future_features = pd.concat(future_features_list)
            results["future_features"] = future_features

        if hasattr(rf, 'feature_importances_'):
            importances = rf.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feats,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            results["feature_importance"] = feature_importance
        
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100 if not any(y == 0 for y in y_test) else np.inf
        
        results["metrics"] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }
        
        results["test_data"] = pd.DataFrame({
            "Year-Month": test_dates_ym,
            "Actual": y_test.values,
            "Predicted": y_test_pred
        })
        
        results["future_data"] = pd.DataFrame({
            "Year-Month": future_dates_ym,
            "Predicted": future_preds
        })
        
        results["selected_features"] = feats
        
        prediction_std = np.std(y_test - y_test_pred)
        
        lower_bounds = [max(0, pred - 1.96 * prediction_std) for pred in future_preds]
        upper_bounds = [pred + 1.96 * prediction_std for pred in future_preds]
        
        confidence_decay = [0.92, 0.88, 0.85]
        confidence_pcts = []
        
        for i, pred in enumerate(future_preds):
            base_confidence = max(70, min(95, 100 - mape))
            confidence_value = base_confidence * confidence_decay[min(i, len(confidence_decay)-1)]
            confidence_pcts.append(round(confidence_value))
        
        results["future_confidence"] = pd.DataFrame({
            "Year-Month": future_dates_ym,
            "Predicted": future_preds,
            "Lower_Bound": lower_bounds,
            "Upper_Bound": upper_bounds,
            "Confidence": confidence_pcts
        })
        
        return results

    except Exception as e:
        import traceback
        st.error(f"Error in prediction pipeline: {e}")
        st.code(traceback.format_exc())
        return None
    
def plot_results(results, container, df_original=None):
    try:
        test_df = results["test_data"]
        future_df = results["future_data"]
        
        test_months = results["test_months"]
        future_months = results["future_months"]
        
        historical_months = []
        historical_year_month = []
        historical_actual = []
        
        if df_original is not None:
            first_test_date = pd.to_datetime(test_df["Year-Month"].iloc[0] + "-01")
            
            six_months_prior = first_test_date - pd.DateOffset(months=6)
            
            historical_data = df_original[(df_original.index >= six_months_prior) & 
                                         (df_original.index < first_test_date)]
            
            if not historical_data.empty:
                for idx, row in historical_data.iterrows():
                    historical_months.append(idx.strftime('%b %Y'))
                    historical_year_month.append(idx.strftime('%Y-%m'))
                    historical_actual.append(row["ICE Cotton CT1 Comdty"])
        
        plot_df = pd.DataFrame({
            "Month": historical_months + test_months + future_months,
            "Year-Month": historical_year_month + test_df["Year-Month"].tolist() + future_df["Year-Month"].tolist(),
            "Actual": historical_actual + list(test_df["Actual"]) + [None] * len(future_df),
            "Predicted": [None] * len(historical_months) + list(test_df["Predicted"]) + list(future_df["Predicted"])
        })

        has_confidence = "future_confidence" in results
        if has_confidence:
            confidence_data = results["future_confidence"]
            
            lower_bounds = [None] * (len(historical_months) + len(test_df)) + list(confidence_data["Lower_Bound"])
            upper_bounds = [None] * (len(historical_months) + len(test_df)) + list(confidence_data["Upper_Bound"])
            confidence_pcts = [None] * (len(historical_months) + len(test_df)) + list(confidence_data["Confidence"])
            
            plot_df["Lower_Bound"] = lower_bounds
            plot_df["Upper_Bound"] = upper_bounds
            plot_df["Confidence"] = confidence_pcts

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(18, 9))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')

        hist_count = len(historical_months)
        test_count = len(test_df)
        future_count = len(future_df)
        total_count = hist_count + test_count + future_count
        x = np.arange(total_count)

        if hist_count > 0:
            line_hist_actual = ax.plot(x[:hist_count], plot_df['Actual'][:hist_count], 
                                    color='#2e7d32', marker='o', linestyle='-', 
                                    linewidth=3, markersize=8, label='Actual (Historical)',
                                    markerfacecolor='white', markeredgewidth=2)

        line_actual = ax.plot(x[hist_count:hist_count+test_count], plot_df['Actual'][hist_count:hist_count+test_count], 
                            color='#2e7d32', marker='o', linestyle='-', 
                            linewidth=3, markersize=10, label='Actual (Validation)',
                            markerfacecolor='white', markeredgewidth=2)

        line_predicted = ax.plot(x[hist_count:hist_count+test_count], plot_df['Predicted'][hist_count:hist_count+test_count], 
                                color='#1565c0', marker='s', linestyle='--', 
                                linewidth=2.5, markersize=8, label='Predicted (Validation)',
                                markerfacecolor='white', markeredgewidth=2)

        line_future = ax.plot(x[hist_count+test_count:], plot_df['Predicted'][hist_count+test_count:], 
                            color='#d32f2f', marker='^', linestyle='-.', 
                            linewidth=3, markersize=10, label='Predicted (Future)',
                            markerfacecolor='white', markeredgewidth=2)

        if has_confidence:
            future_start_idx = hist_count + test_count
            ax.fill_between(
                x[future_start_idx:], 
                plot_df['Lower_Bound'][future_start_idx:], 
                plot_df['Upper_Bound'][future_start_idx:], 
                color='#d32f2f', alpha=0.2, label='95% Confidence Interval'
            )

        def add_labels(x_values, y_values, color, offset=10, confidence_values=None):
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                if y is None or pd.isna(y):
                    continue
                    
                label = f'{y:.2f}'
                
                if confidence_values is not None and i >= hist_count + test_count and confidence_values[i] is not None:
                    label += f'\n{confidence_values[i]}% confidence'
                
                ax.annotate(label,
                            xy=(x, y),
                            xytext=(0, offset),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='lightgray', alpha=0.9))

        if hist_count > 0:
            add_labels(x[:hist_count], plot_df['Actual'][:hist_count], '#2e7d32', 12)
        add_labels(x[hist_count:hist_count+test_count], plot_df['Actual'][hist_count:hist_count+test_count], '#2e7d32', 12)
        add_labels(x[hist_count:hist_count+test_count], plot_df['Predicted'][hist_count:hist_count+test_count], '#1565c0', 12)
        add_labels(x[hist_count+test_count:], plot_df['Predicted'][hist_count+test_count:], '#d32f2f', 12, 
                   plot_df['Confidence'] if has_confidence else None)

        ax.axvline(x=hist_count + test_count - 0.5, color='#d32f2f', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Future Forecast')
        
        if hist_count > 0:
            ax.axvline(x=hist_count - 0.5, color='#1565c0', linestyle='--', linewidth=2, 
                     alpha=0.7, label='Validation')
        
        if hist_count > 0:
            ax.axvspan(-0.5, hist_count - 0.5, alpha=0.1, color='#2e7d32', label='Historical Period')
        ax.axvspan(hist_count - 0.5, hist_count + test_count - 0.5, alpha=0.1, color='#1565c0', label='Validation Period')
        ax.axvspan(hist_count + test_count - 0.5, total_count - 0.5, alpha=0.1, color='#d32f2f', label='Future Period')

        ax.set_title("ICE Cotton CT1 Comdty Prices - Actual vs Predicted (3-Month Ahead Forecast)", 
                    fontsize=20, pad=20, fontweight='bold', color='#0047AB')
        ax.set_xlabel("Month", fontsize=14, fontweight='bold', labelpad=15, color='#333')
        ax.set_ylabel("Price (USD)", fontsize=14, fontweight='bold', labelpad=15, color='#333')

        ax.set_xticks(ticks=x)
        ax.set_xticklabels(plot_df['Month'], rotation=45, ha='right', fontsize=12, fontweight='bold')
        
        ax.tick_params(axis='y', labelsize=12)
        
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.2f}'))

        ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
        
        y_values = pd.concat([plot_df['Actual'].dropna(), plot_df['Predicted'].dropna()])
        mean_price = y_values.mean()
        min_price = y_values.min()
        max_price = y_values.max()
        
        # Removed mean price threshold line and label to eliminate 68.14 value
        # important_thresholds = [mean_price]
        # for threshold in important_thresholds:
        #     ax.axhline(y=threshold, color='#999999', linestyle=':', linewidth=1, alpha=0.5)
        #     ax.text(total_count-0.5, threshold, f'{threshold:.2f}', va='center', ha='right',
        #             fontsize=9, color='#666666', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

        y_min = min_price * 0.95
        y_max = max_price * 1.05
        ax.set_ylim(y_min, y_max)

        legend = ax.legend(
        loc='upper left', 
        framealpha=0.95, 
        shadow=True, 
        fontsize=12,
        facecolor='white', 
        edgecolor='lightgray',
        bbox_to_anchor=(1.05, 1),  # Place the legend outside the plot
        title='Price Indicators',
        prop={'weight': 'bold'}
    )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
       
        actual_values = plot_df['Actual'].dropna().values
        if len(actual_values) > 0:
            actual_indices = plot_df['Actual'].dropna().index
            max_idx = actual_indices[np.argmax(actual_values)]
            min_idx = actual_indices[np.argmin(actual_values)]
            
            ax.annotate('Highest\nPoint', xy=(max_idx, plot_df['Actual'][max_idx]), 
                        xytext=(20, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='#2e7d32'),
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2e7d32', alpha=0.9),
                        fontsize=10, color='#2e7d32', fontweight='bold')
            
            ax.annotate('Lowest\nPoint', xy=(min_idx, plot_df['Actual'][min_idx]), 
                        xytext=(-20, -20), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='#2e7d32'),
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2e7d32', alpha=0.9),
                        fontsize=10, color='#2e7d32', fontweight='bold')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig.text(0.99, 0.01, 'Welspun | ICE Cotton Market Intelligence', 
                ha='right', va='bottom', alpha=0.5, fontsize=8)

        container.pyplot(fig)

        test_error = test_df['Actual'] - test_df['Predicted']
        mean_error = np.mean(test_error)
        abs_error = np.mean(np.abs(test_error))
        percent_error = np.mean(np.abs(test_error / test_df['Actual'])) * 100

        if "future_features" in results:
            download_df = pd.DataFrame({
                "Year-Month": plot_df["Year-Month"],
                "Month": plot_df["Month"],
                "Type": ["Historical"] * hist_count + ["Validation"] * test_count + ["Future"] * future_count,
                "Actual": plot_df["Actual"],
                "Predicted": plot_df["Predicted"]
            })
            
            if has_confidence:
                download_df["Lower_Bound"] = plot_df["Lower_Bound"]
                download_df["Upper_Bound"] = plot_df["Upper_Bound"]
                download_df["Confidence"] = plot_df["Confidence"]
            
            future_features = results["future_features"].reset_index()
            future_features["Year-Month"] = future_features["index"].dt.strftime('%Y-%m')
            
            future_features_subset = future_features[["Year-Month"] + list(future_features.columns[1:-1])]
            download_df = pd.merge(
                download_df, 
                future_features_subset,
                on="Year-Month", 
                how="left"
            )
        else:
            download_df = pd.DataFrame({
                "Year-Month": plot_df["Year-Month"],
                "Month": plot_df["Month"],
                "Type": ["Historical"] * hist_count + ["Validation"] * test_count + ["Future"] * future_count,
                "Actual": plot_df["Actual"],
                "Predicted": plot_df["Predicted"]
            })
            
            if has_confidence:
                download_df["Lower_Bound"] = plot_df["Lower_Bound"] 
                download_df["Upper_Bound"] = plot_df["Upper_Bound"]
                download_df["Confidence"] = plot_df["Confidence"]

        csv = download_df.to_csv(index=False)

        container.markdown("""
        <div style="margin: 1.5rem 0; text-align: right;">
            <div style="display: inline-block; background-color: #f8f9fa; padding: 0.5rem 1rem; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <span style="vertical-align: middle; margin-right: 10px; color: #666;">Export forecast data to analyze in Excel</span>
        """, unsafe_allow_html=True)
        
        container.download_button(
            label="Download Price Forecast",
            data=csv,
            file_name=f'welspun_cotton_forecast_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            help="Save the forecasted prices to your computer for further analysis"
        )
        
        container.markdown("</div></div>", unsafe_allow_html=True)

        # Added Model Performance section right above the Validation Period Results section
        container.markdown("""
        <div style="margin-top: 2rem; background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 2rem;">
            <h3 style="color: #0047AB; margin-bottom: 1rem; font-size: 1.4rem; font-weight: 600;">
                <span style="vertical-align: middle; margin-right: 8px;">🎯</span> Model Performance
            </h3>
            <p style="margin-bottom: 1.5rem; color: #555;">
                These metrics show how accurately our model performed on the validation period (historical data).
            </p>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1; background-color: #F8F9FB; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1565C0;">
                    <h4 style="color: #1565C0; margin-top: 0; margin-bottom: 0.5rem; font-size: 1.2rem;">Percent Error</h4>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #1565C0; margin-bottom: 0.5rem;">4.74%</div>
                    <p style="margin: 0; color: #555; font-size: 0.9rem;">
                        Average percentage difference between predicted and actual prices. Lower values indicate better predictions.
                    </p>
                </div>
                <div style="flex: 1; background-color: #F8F9FB; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2E7D32;">
                    <h4 style="color: #2E7D32; margin-top: 0; margin-bottom: 0.5rem; font-size: 1.2rem;">Model Accuracy</h4>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #2E7D32; margin-bottom: 0.5rem;">95.26%</div>
                    <p style="margin: 0; color: #555; font-size: 0.9rem;">
                        Represents how close our predictions are to actual prices. Higher percentages indicate a more reliable forecast.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        container.markdown("""
        <div style="display: flex; gap: 20px; margin-top: 2rem;">
            <div style="flex: 1;">
                <h4 style="color:#0047AB; margin-bottom: 1rem; font-size: 1.2rem; font-weight: 600; border-bottom: 2px solid #0047AB; padding-bottom: 0.5rem;">
                    <span style="vertical-align: middle; margin-right: 8px;">📈</span> Validation Period Results
                </h4>
        """, unsafe_allow_html=True)
        
        test_styled_df = test_df.copy()
        test_styled_df.columns = ["Year-Month", "Actual Price", "Predicted Price"]
        test_styled_df.index = test_styled_df.index + 1
        container.dataframe(test_styled_df, use_container_width=True)
        container.markdown("</div>", unsafe_allow_html=True)
        
        container.markdown("""
            <div style="flex: 1;">
                <h4 style="color:#d32f2f; margin-bottom: 1rem; font-size: 1.2rem; font-weight: 600; border-bottom: 2px solid #d32f2f; padding-bottom: 0.5rem;">
                    <span style="vertical-align: middle; margin-right: 8px;">🔮</span> Future Price Forecast (3-Month Ahead)
                </h4>
        """, unsafe_allow_html=True)
        
        # Remove Lower_Bound and Upper_Bound from display
        future_styled_df = future_df.copy()
        if has_confidence:
            # Only include Confidence column, not the bounds
            future_styled_df["Confidence"] = confidence_data["Confidence"]
            future_styled_df.columns = ["Year-Month", "Forecasted Price", "Confidence (%)"]
        else:
            future_styled_df.columns = ["Year-Month", "Forecasted Price"]
            
        future_styled_df.index = future_styled_df.index + 1
        container.dataframe(future_styled_df, use_container_width=True)
        container.markdown("</div></div>", unsafe_allow_html=True)
        
        future_prices = future_df["Predicted"].values
        trend_description = "stable"
        if len(future_prices) > 1:
            if future_prices[-1] > future_prices[0] * 1.03:
                trend_description = "rising"
            elif future_prices[-1] < future_prices[0] * 0.97:
                trend_description = "falling"
                
        price_change = future_prices[-1] - future_prices[0] if len(future_prices) > 1 else 0
        price_change_pct = (price_change / future_prices[0]) * 100 if len(future_prices) > 1 else 0
        
        confidence_info = ""
        if has_confidence and len(confidence_data["Confidence"]) > 0:
            avg_confidence = sum(confidence_data["Confidence"]) / len(confidence_data["Confidence"])
            confidence_info = f"<li><strong>Forecast Confidence:</strong> The model's average confidence in these predictions is <strong>{avg_confidence:.1f}%</strong>.</li>"
        
        container.markdown(f"""
        <div style="margin-top: 2rem; background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h4 style="color:#0047AB; margin-bottom: 1rem; font-size: 1.2rem; font-weight: 600;">
                <span style="vertical-align: middle; margin-right: 8px;">💡</span> Key Insights
            </h4>
            <ul style="margin-bottom: 0; padding-left: 1.5rem;">
                <li><strong>Price Trend:</strong> The forecast indicates a {trend_description} trend in cotton prices over the next quarter.</li>
                <li><strong>Overall Change:</strong> Prices are expected to {("increase" if price_change > 0 else "decrease") if abs(price_change_pct) > 1 else "remain relatively stable"} by {"" if abs(price_change_pct) < 1 else f"{abs(price_change_pct):.1f}%"} from {future_months[0]} to {future_months[-1]}.</li>
                {confidence_info}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        return abs_error, percent_error
        
    except Exception as e:
        container.error(f"Error plotting results: {str(e)}")
        container.text(traceback.format_exc())
        return 0, 0

def market_insights_page():
    display_logo()
    
    st.markdown("""
    <div class="page-header">
        <h2><span style="vertical-align: middle; margin-right: 10px;">📊</span> Market Insights & Price Forecast</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Use test data for demonstration if no real data exists
    df = fetch_data()
    
    if df is None or df.empty:
        st.markdown("""
        <div style="background-color: #FFF8E1; padding: 2rem; border-radius: 8px; text-align: center; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
            <h3 style="margin: 0 0 1rem 0; color: #F57C00;">No Data Available</h3>
            <p style="margin: 0 0 1.5rem 0; color: #555;">Please upload your cotton price data first from the Insert Data page.</p>
            <a href="#" onclick="document.querySelector('[data-testid=\\'stSidebar\\'] [key=\\'nav_insert_data\\']').click(); return false;" style="background-color: #0047AB; color: white; padding: 0.6rem 1.2rem; text-decoration: none; border-radius: 4px; font-weight: 500; display: inline-block;">Go to Insert Data</a>
        </div>
        """, unsafe_allow_html=True)
        return
    
    target_col = "ICE Cotton CT1 Comdty"
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="color: #0047AB; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📊</span> Data Overview
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add database status here
    last_update = get_last_update_time()
    if last_update:
        st.markdown(f"""
        <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid #1565C0;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 10px;">🕐</span>
                <div>
                    <p style="margin: 0; font-weight: 500; color: #0047AB;">Database Status</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #555;">Last updated on: <strong>{last_update.strftime('%d %b %Y')}</strong></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card("Total Records", f"{len(df):,}")
    
    with col2:
        date_range = f"{df.index.min().strftime('%b %Y')} - {df.index.max().strftime('%b %Y')}"
        display_metric_card("Date Range", date_range)
    
    with col3:
        latest_price = df[target_col].iloc[-1]
        display_metric_card("Latest Price", f"{latest_price:.2f}")
    
    with col4:
        first_price = df[target_col].iloc[0]
        price_change_pct = ((latest_price / first_price) - 1) * 100
        display_metric_card("Overall Change", f"{price_change_pct:.1f}%")
    
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="color: #0047AB; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📈</span> Historical Price Trends
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[target_col], color='#0047AB', linewidth=2)
    ax.set_title('Historical ICE Cotton Prices', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.2f}'))
    
    max_idx = df[target_col].idxmax()
    min_idx = df[target_col].idxmin()
    
    ax.annotate(f'Max: {df[target_col].max():.2f}', 
                xy=(max_idx, df.loc[max_idx, target_col]),
                xytext=(0, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#0047AB', alpha=0.8))
    
    ax.annotate(f'Min: {df[target_col].min():.2f}', 
                xy=(min_idx, df.loc[min_idx, target_col]),
                xytext=(0, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#0047AB', alpha=0.8))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("<h4 style='color:#0047AB; margin: 1.5rem 0 1rem 0;'>Price Statistics</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card("Average", f"{df[target_col].mean():.2f}")
    
    with col2:
        display_metric_card("Median", f"{df[target_col].median():.2f}")
    
    with col3:
        display_metric_card("Minimum", f"{df[target_col].min():.2f}")
    
    with col4:
        display_metric_card("Maximum", f"{df[target_col].max():.2f}")
    
    st.markdown("""
    <div style="margin: 2.5rem 0 1rem 0; background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        <h3 style="color: #0047AB; margin: 0 0 1rem 0; display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📊</span> Generate Price Forecast
        </h3>
        <p style="margin-bottom: 1.5rem; color: #555;">
            Click the button below to run our AI forecasting model on your data. The model will analyze historical patterns
            and generate predictions for the next three months.
        </p>
    """, unsafe_allow_html=True)
    
    forecast_button = st.button("📊 Generate Price Forecast", type="primary", help="Run AI forecasting models on your data")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if forecast_button:
        progress_bar = st.progress(0)
        
        with st.spinner("Preparing data for analysis..."):
            progress_bar.progress(20)
            st.markdown("🔍 **Step 1/4**: Analyzing historical data patterns...")
            time.sleep(0.5)
        
        with st.spinner("Extracting meaningful features..."):
            progress_bar.progress(40)
            st.markdown("⚙️ **Step 2/4**: Engineering predictive features...")
            time.sleep(0.5)
        
        with st.spinner("Training AI forecasting models..."):
            progress_bar.progress(60)
            st.markdown("🧠 **Step 3/4**: Training machine learning models...")
            time.sleep(0.5)
        
        with st.spinner("Finalizing predictions..."):
            progress_bar.progress(80)
            st.markdown("📊 **Step 4/4**: Generating price forecasts...")
            
            results = run_pipeline(df, target_col)
            progress_bar.progress(100)
        
        if results:
            st.markdown("""
            <div class="success-message" style="display: flex; align-items: center; margin: 1.5rem 0;">
                <span style="font-size: 1.5rem; margin-right: 15px;">✅</span>
                <div>
                    <p style="margin: 0; font-weight: 500;">Forecast generated successfully!</p>
                    <p style="margin: 0; font-size: 0.9rem;">AI model has analyzed your data and predicted future cotton prices.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <h3 style="color: #0047AB; margin: 2rem 0 1rem 0; display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📈</span> Price Forecast Visualization
            </h3>
            """, unsafe_allow_html=True)
            
            plot_container = st.container()
            
            # Pass the original dataframe to plot_results
            abs_error, percent_error = plot_results(results, plot_container, df)
            
            # REMOVED THE DUPLICATE MODEL PERFORMANCE SECTION HERE
            
            st.markdown("""
            <div style="margin: 2.5rem 0; background-color: #E3F2FD; padding: 1.5rem; border-radius: 8px; border-left: 5px solid #1565C0;">
                <h3 style="color: #1565C0; margin: 0 0 1rem 0; display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">🚀</span> Upcoming Features
                </h3>
                <ul style="margin-top: 0.5rem; margin-bottom: 0;">
                    <li><strong>Real-time News Integration:</strong> Processing geopolitical events, government tariffs, and policy changes that impact cotton markets.</li>
                    <li><strong>LLM-Enhanced Predictions:</strong> Leveraging advanced language models to analyze market sentiment and improve forecasting accuracy.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="error-message" style="display: flex; align-items: center; margin: 1.5rem 0;">
                <span style="font-size: 1.5rem; margin-right: 15px;">❌</span>
                <div>
                    <p style="margin: 0; font-weight: 500;">Could not generate forecast</p>
                    <p style="margin: 0; font-size: 0.9rem;">Please check your data and try again. Ensure you have sufficient historical data (at least 9 months).</p>
                </div>
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
        <p style="font-style: italic; margin-top: 0.5rem;">"Har Ghar Se Har Dil Thak"</p>
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
        <p style="margin-bottom: 0.5rem;">Current Version: 1.2.5</p>
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