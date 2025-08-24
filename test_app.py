import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("VMD Analyzer Test")
st.write("Testing basic functionality...")

# Test data loading
if st.button("Test Data Loading"):
    data = yf.download('AAPL', period='5d')
    st.write(f"Downloaded {len(data)} rows of AAPL data")
    st.dataframe(data.head())
    
    # Test VMD function (simplified)
    prices = data['Close'].values
    normalized = (prices - np.mean(prices)) / np.std(prices)
    st.line_chart(normalized)
    st.success("Basic functionality working!")

st.write("If you can see this, Streamlit is working correctly!")