# -*- coding: utf-8 -*-
"""
Streamlit VMD Analyzer with Predictive Stability Analysis
Interactive web application for VMD analysis of stock prices
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="VMD Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7):
    """VMD implementation"""
    if len(signal) % 2:
        signal = signal[:-1]
    
    T = len(signal)
    t = np.arange(1, T+1)/T
    freqs = t - 0.5 - 1/T
    
    f_hat = np.fft.fftshift(np.fft.fft(signal))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T//2] = 0
    
    u_hat_plus = np.zeros((K, len(freqs)), dtype=complex)
    omega = np.zeros((K,))
    
    if init == 1:
        omega = np.linspace(0, 0.5, K+1)[1:K+1]
    
    lambda_hat = np.zeros(len(freqs), dtype=complex)
    
    n = 0
    sum_uk = 0
    uDiff = tol + 1
    
    while uDiff > tol and n < 500:
        for k in range(K):
            u_hat_plus[k] = (f_hat_plus - sum_uk + u_hat_plus[k] + lambda_hat/2) / (1 + alpha * (freqs - omega[k])**2)
            
            if not DC:
                u_hat_plus[k][T//2] = 0
            
            omega[k] = np.sum(freqs * np.abs(u_hat_plus[k])**2) / np.sum(np.abs(u_hat_plus[k])**2)
        
        lambda_hat = lambda_hat + tau * (f_hat_plus - np.sum(u_hat_plus, axis=0))
        
        sum_uk_new = np.sum(u_hat_plus, axis=0)
        uDiff = (1/T) * np.sum(np.abs(sum_uk_new - sum_uk)**2)
        sum_uk = sum_uk_new
        
        n += 1
    
    u_hat = np.zeros((T, K), dtype=complex)
    u = np.zeros((K, T))
    
    for k in range(K):
        u_hat[T//2:T, k] = u_hat_plus[k, T//2:T]
        u_hat[1:T//2, k] = np.conj(u_hat_plus[k, T//2-1:0:-1])
        u_hat[0, k] = np.conj(u_hat[-1, k])
        u[k] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    
    return u, u_hat, omega


class PredictiveStabilityAnalyzer:
    """Analyzer for VMD predictive stability and utility"""
    
    def __init__(self, modes, frequencies, original_signal):
        self.modes = modes
        self.frequencies = frequencies
        self.original = original_signal
        self.K = len(modes)
        
        # Ensure all arrays have consistent length
        if len(modes) > 0:
            self.min_length = min(len(original_signal), len(modes[0]))
        else:
            self.min_length = len(original_signal)
        
        self.original = self.original[:self.min_length]
        self.modes = [mode[:self.min_length] for mode in self.modes]
        
    def calculate_trend_consistency(self):
        """Analiza stałości trendu w każdym modzie"""
        consistencies = []
        
        for i, mode in enumerate(self.modes):
            directions = np.diff(mode)
            direction_changes = np.sum(np.diff(np.sign(directions)) != 0)
            consistency = 1.0 - (direction_changes / len(directions)) if len(directions) > 0 else 0.0
            consistencies.append(consistency)
            
        return consistencies
    
    def calculate_mode_stability(self, window_size=5):
        """Analiza stabilności lokalnej każdego modu"""
        stabilities = []
        
        for mode in self.modes:
            local_vars = []
            for i in range(len(mode) - window_size + 1):
                window = mode[i:i + window_size]
                local_var = np.var(window)
                local_vars.append(local_var)
            
            stability = 1.0 / (1.0 + np.mean(local_vars)) if local_vars else 0.0
            stabilities.append(stability)
            
        return stabilities
    
    def calculate_predictive_power(self):
        """Moc predykcyjna - korelacja z przyszłymi wartościami"""
        powers = []
        
        for mode in self.modes:
            if len(mode) < 4:
                powers.append(0.0)
                continue
                
            future_correlations = []
            for lag in range(1, min(4, len(mode)//2)):
                if len(mode) > lag:
                    current = mode[:-lag]
                    future = mode[lag:]
                    if len(current) > 1 and len(future) > 1:
                        try:
                            corr = abs(np.corrcoef(current, future)[0, 1])
                            if not np.isnan(corr):
                                future_correlations.append(corr)
                        except:
                            pass
            
            power = np.mean(future_correlations) if future_correlations else 0.0
            powers.append(power)
            
        return powers
    
    def calculate_mode_separability(self):
        """Separowalność modów - jak różne są od siebie"""
        separabilities = []
        
        for i, mode_i in enumerate(self.modes):
            cross_correlations = []
            for j, mode_j in enumerate(self.modes):
                if i != j and len(mode_i) > 1 and len(mode_j) > 1:
                    try:
                        corr = abs(np.corrcoef(mode_i, mode_j)[0, 1])
                        if not np.isnan(corr):
                            cross_correlations.append(corr)
                    except:
                        pass
            
            separability = 1.0 - np.mean(cross_correlations) if cross_correlations else 1.0
            separabilities.append(separability)
            
        return separabilities
    
    def calculate_reconstruction_quality_by_modes(self):
        """Jakość rekonstrukcji kumulatywnej"""
        qualities = []
        cumulative = np.zeros(self.min_length)
        
        for i, mode in enumerate(self.modes):
            cumulative += mode
            
            # Ensure same length for comparison
            comparison_length = min(len(self.original), len(cumulative))
            original_comp = self.original[:comparison_length]
            cumulative_comp = cumulative[:comparison_length]
            
            # MSE między oryginalnym sygnałem a rekonstrukcją
            mse = np.mean((original_comp - cumulative_comp)**2)
            
            # Korelacja
            if comparison_length > 1:
                try:
                    correlation = np.corrcoef(original_comp, cumulative_comp)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                except:
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Jakość jako kombinacja MSE i korelacji
            quality = correlation - mse * 0.1
            qualities.append(quality)
            
        return qualities
    
    def calculate_overall_predictive_utility(self):
        """Ogólna użyteczność predykcyjna każdego modu"""
        trend_consistency = self.calculate_trend_consistency()
        stability = self.calculate_mode_stability()
        predictive_power = self.calculate_predictive_power()
        separability = self.calculate_mode_separability()
        
        utilities = []
        for i in range(self.K):
            utility = (
                trend_consistency[i] * 0.3 +
                stability[i] * 0.25 +
                predictive_power[i] * 0.3 +
                separability[i] * 0.15
            )
            utilities.append(utility)
            
        return utilities, {
            'trend_consistency': trend_consistency,
            'stability': stability,
            'predictive_power': predictive_power,
            'separability': separability
        }


@st.cache_data
def load_stock_data(symbol, period):
    """Load stock data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1d")
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None


def create_candlestick_chart(data, title="Stock Price", vmd_reconstruction=None):
    """Create interactive candlestick chart with Plotly"""
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="OHLC",
        increasing_line_color='#00CC96',
        decreasing_line_color='#EF553B'
    ))
    
    # Add moving averages
    ma20 = data['Close'].rolling(window=20).mean()
    ma50 = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma20,
        mode='lines',
        name='MA20',
        line=dict(color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma50,
        mode='lines',
        name='MA50',
        line=dict(color='purple', width=2)
    ))
    
    # Add VMD reconstruction if available
    if vmd_reconstruction is not None:
        fig.add_trace(go.Scatter(
            x=vmd_reconstruction['dates'],
            y=vmd_reconstruction['reconstruction'],
            mode='lines',
            name='VMD Reconstruction',
            line=dict(color='red', width=3)
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template='plotly_white',
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_vmd_modes_chart(modes, frequencies, original_signal, dates):
    """Create VMD modes visualization - Individual modes only, full width"""
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    
    # Individual modes
    for i, (mode, freq) in enumerate(zip(modes, frequencies)):
        fig.add_trace(go.Scatter(
            x=dates,
            y=mode,
            mode='lines',
            name=f'Mode {i+1} (f={freq:.3f})',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Add original signal
    fig.add_trace(go.Scatter(
        x=dates,
        y=original_signal,
        mode='lines',
        name='Original',
        line=dict(color='black', width=3, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="Individual VMD Modes",
        yaxis_title="Normalized Amplitude",
        xaxis_title="Date", 
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def create_mode_frequencies_chart(frequencies):
    """Create mode frequencies bar chart"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Mode {i+1}' for i in range(len(frequencies))],
        y=frequencies,
        name='Frequencies',
        marker_color=colors[:len(frequencies)],
        text=[f'{f:.3f}' for f in frequencies],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="VMD Mode Frequencies",
        yaxis_title="Frequency (Hz)",
        xaxis_title="Modes",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_reconstruction_error_chart(modes, original_signal):
    """Create reconstruction error chart"""
    reconstruction_error = []
    cumulative = np.zeros_like(modes[0])
    
    for i, mode in enumerate(modes):
        cumulative += mode
        error = np.mean((original_signal - cumulative)**2)
        reconstruction_error.append(error)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f'Modes 1-{i+1}' for i in range(len(modes))],
        y=reconstruction_error,
        mode='lines+markers',
        name='MSE Error',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Cumulative Reconstruction Error",
        yaxis_title="Mean Squared Error",
        xaxis_title="Cumulative Modes",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_predictive_analysis_chart(utilities, metrics, reconstruction_quality):
    """Create predictive analysis visualization"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Overall Predictive Utility", "Detailed Stability Metrics", "Trend Consistency",
            "Reconstruction Quality", "Mode Comparison", "Recommendations"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"secondary_y": False}, {"type": "bar"}, {"type": "table"}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    mode_names = [f'Mode {i+1}' for i in range(len(utilities))]
    
    # Overall utility (top left)
    fig.add_trace(go.Bar(
        x=mode_names,
        y=utilities,
        name='Utility',
        marker_color=colors[:len(utilities)],
        text=[f'{u:.3f}' for u in utilities],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # Detailed metrics (top middle)
    fig.add_trace(go.Bar(
        x=mode_names,
        y=metrics['stability'],
        name='Stability',
        marker_color='lightblue',
        offsetgroup=0,
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=mode_names,
        y=metrics['predictive_power'],
        name='Pred. Power',
        marker_color='lightgreen',
        offsetgroup=1,
        showlegend=False
    ), row=1, col=2)
    
    # Trend consistency (top right)
    fig.add_trace(go.Bar(
        x=mode_names,
        y=metrics['trend_consistency'],
        name='Consistency',
        marker_color=colors[:len(utilities)],
        text=[f'{c:.3f}' for c in metrics['trend_consistency']],
        textposition='outside',
        showlegend=False
    ), row=1, col=3)
    
    # Reconstruction quality (bottom left)
    cumulative_names = [f'M1-{i+1}' for i in range(len(reconstruction_quality))]
    fig.add_trace(go.Scatter(
        x=cumulative_names,
        y=reconstruction_quality,
        mode='lines+markers',
        name='Quality',
        line=dict(color='purple', width=3),
        marker=dict(size=10),
        showlegend=False
    ), row=2, col=1)
    
    # Mode comparison radar (bottom middle)
    categories = ['Trend\nConsistency', 'Stability', 'Predictive\nPower', 'Separability']
    
    for i, mode_name in enumerate(mode_names[:4]):  # Limit to first 4 modes for readability
        fig.add_trace(go.Bar(
            x=categories,
            y=[
                metrics['trend_consistency'][i],
                metrics['stability'][i],
                metrics['predictive_power'][i],
                metrics['separability'][i]
            ],
            name=mode_name,
            marker_color=colors[i],
            showlegend=False
        ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',
        title_text="Predictive Stability Analysis"
    )
    
    return fig


def main():
    # Title and description
    st.title("📈 VMD Stock Analyzer with Predictive Stability")
    st.markdown("""
    **Advanced Variational Mode Decomposition (VMD) Analysis** for stock price prediction and stability assessment.
    
    🎯 **Features:**
    - Interactive candlestick charts with moving averages
    - VMD decomposition with customizable parameters  
    - Predictive stability analysis with multiple metrics
    - Comprehensive visualization dashboard
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Parameters")
        
        # Stock selection with auto-load on enter
        symbol = st.text_input(
            "Stock Symbol", 
            value="AAPL", 
            help="Enter stock ticker (e.g., AAPL, TSLA, MSFT) and press Enter",
            key="symbol_input"
        )
        
        period = st.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2,
            help="Historical data period",
            key="period_select"
        )
        
        # VMD parameters
        st.subheader("🔧 VMD Parameters")
        K = st.slider("Number of Modes", min_value=2, max_value=8, value=4, 
                     help="Number of VMD modes to extract", key="K_slider")
        alpha = st.slider("Alpha Parameter", min_value=500, max_value=5000, value=2000, step=100, 
                         help="Bandwidth constraint (higher = narrower frequency bands)", key="alpha_slider")
        
        # Analysis options
        st.subheader("🎛️ Analysis Options")
        show_individual_modes = st.checkbox("Show Individual Modes", value=True, key="show_modes")
        show_predictive_analysis = st.checkbox("Show Predictive Analysis", value=True, key="show_pred")
        range_selection = st.checkbox("Enable Range Selection", value=False, 
                                     help="Select specific date range for analysis", key="range_sel")
        
        # Time animation controls
        st.subheader("🎬 Time Animation")
        enable_time_slider = st.checkbox("Enable Time Slider", value=False, 
                                        help="Animate data changes over time", key="time_slider_enable")
    
    # Auto-load data when symbol or period changes
    if ('data' not in st.session_state or 
        st.session_state.get('last_symbol') != symbol.upper() or 
        st.session_state.get('last_period') != period):
        
        if symbol.strip():  # Only load if symbol is not empty
            with st.spinner(f"Loading {symbol} data..."):
                data = load_stock_data(symbol.upper(), period)
                
                if data is not None and not data.empty:
                    st.session_state.data = data
                    st.session_state.symbol = symbol.upper()
                    st.session_state.last_symbol = symbol.upper()
                    st.session_state.last_period = period
                    st.success(f"✅ Loaded {len(data)} candles for {symbol.upper()}")
                    # Clear previous VMD results when loading new data
                    if 'vmd_results' in st.session_state:
                        del st.session_state.vmd_results
                else:
                    st.error("❌ Failed to load data. Please check the symbol and try again.")
                    return
    
    if 'data' not in st.session_state:
        st.info("👆 Please load stock data to begin analysis")
        return
    
    data = st.session_state.data
    
    # Candlestick chart
    st.subheader("📈 Stock Price Chart")
    
    # Time slider for animation
    time_end_idx = len(data) - 1
    if enable_time_slider:
        time_slider_value = st.slider(
            "Data End Point", 
            min_value=20,  # Minimum 20 data points for meaningful analysis
            max_value=time_end_idx,
            value=time_end_idx,
            step=1,
            help="Slide to animate data changes over time",
            key="time_slider"
        )
        
        # Show current date (smaller text)
        current_date = data.index[time_slider_value]
        st.caption(f"📅 {current_date.strftime('%Y-%m-%d')} ({time_slider_value + 1} candles)")
        
        # Trim data based on slider
        data = data.iloc[:time_slider_value + 1]
    
    # Check if VMD reconstruction exists and force refresh if needed
    vmd_reconstruction = None
    
    # Force VMD analysis if we have data but no results (initial load issue)
    if 'vmd_results' not in st.session_state and len(data) >= 20:
        with st.spinner("Running initial VMD decomposition..."):
            # Use full data for initial analysis
            analysis_data_temp = data
            
            # Normalize prices
            prices = analysis_data_temp['Close'].values
            normalized_prices = (prices - np.mean(prices)) / np.std(prices)
            
            # Perform VMD
            modes, _, frequencies = vmd(normalized_prices, K=K, alpha=alpha)
            
            # Ensure consistent lengths
            min_length = min(len(normalized_prices), len(modes[0]))
            normalized_prices = normalized_prices[:min_length]
            modes = [mode[:min_length] for mode in modes]
            dates = analysis_data_temp.index[:min_length]
            
            # Store results in session state
            st.session_state.vmd_results = {
                'modes': modes,
                'frequencies': frequencies,
                'normalized_prices': normalized_prices,
                'dates': dates,
                'original_data': analysis_data_temp.iloc[:min_length]
            }
            time_slider_val = st.session_state.get('time_slider', len(data) - 1) if enable_time_slider else len(data)
            st.session_state.last_vmd_params = (K, alpha, len(data), time_slider_val)
    
    if 'vmd_results' in st.session_state:
        results = st.session_state.vmd_results
        # Calculate cumulative reconstruction and scale back to price levels
        cumulative_reconstruction = np.sum(results['modes'], axis=0)
        original_data = results['original_data']
        mean_price = original_data['Close'].mean()
        std_price = original_data['Close'].std()
        scaled_reconstruction = cumulative_reconstruction * std_price + mean_price
        
        # Always use exact length matching - no trimming needed as VMD is recalculated
        result_dates = results['dates']
        # Ensure reconstruction matches exactly the current data length
        current_len = len(data)
        if len(scaled_reconstruction) != current_len:
            # This should not happen with proper VMD recalculation, but safety check
            min_len = min(len(scaled_reconstruction), current_len, len(result_dates))
            scaled_reconstruction = scaled_reconstruction[:min_len]
            result_dates = result_dates[:min_len]
        
        vmd_reconstruction = {
            'dates': result_dates,
            'reconstruction': scaled_reconstruction
        }
    
    candlestick_fig = create_candlestick_chart(
        data, 
        f"{st.session_state.symbol} Stock Price", 
        vmd_reconstruction=vmd_reconstruction
    )
    
    # Add range selection if enabled
    if range_selection:
        st.info("💡 Use the range selector below the chart to select a specific period for VMD analysis")
    
    chart_container = st.plotly_chart(candlestick_fig, use_container_width=True, key="candlestick_chart")
    
    # Range selection
    start_date = data.index[0].date()
    end_date = data.index[-1].date()
    
    if range_selection:
        col1, col2 = st.columns(2)
        with col1:
            selected_start = st.date_input("Analysis Start Date", value=start_date, 
                                          min_value=start_date, max_value=end_date)
        with col2:
            selected_end = st.date_input("Analysis End Date", value=end_date,
                                        min_value=start_date, max_value=end_date)
        
        # Filter data based on selection
        mask = (data.index.date >= selected_start) & (data.index.date <= selected_end)
        analysis_data = data.loc[mask]
    else:
        analysis_data = data
    
    if len(analysis_data) < 10:
        st.warning("⚠️ Selected range too small. Please select at least 10 data points.")
        return
    
    # Auto-run VMD Analysis when parameters change or data is loaded
    # Always use the exact current data length with explicit slider position
    current_data_len = len(analysis_data)
    time_slider_pos = st.session_state.get('time_slider', time_end_idx) if enable_time_slider else None
    current_vmd_params = (K, alpha, current_data_len, symbol.upper(), period, time_slider_pos)
    
    # Force VMD recalculation when slider moves
    should_run_vmd = ('vmd_results' not in st.session_state or 
                      st.session_state.get('last_vmd_params') != current_vmd_params or
                      len(st.session_state.get('vmd_results', {}).get('modes', [[]])[0]) != current_data_len)
    
    if should_run_vmd:
        with st.spinner("Running VMD decomposition..."):
            # Normalize prices for exact current data length
            prices = analysis_data['Close'].values
            normalized_prices = (prices - np.mean(prices)) / np.std(prices)
            
            # Perform VMD
            modes, _, frequencies = vmd(normalized_prices, K=K, alpha=alpha)
            
            # Force exact length match - no truncation issues
            target_length = current_data_len
            if len(modes[0]) != target_length:
                # Pad or truncate modes to exact target length
                adjusted_modes = []
                for mode in modes:
                    if len(mode) > target_length:
                        adjusted_modes.append(mode[:target_length])
                    elif len(mode) < target_length:
                        # Pad with zeros if needed (shouldn't happen but safety)
                        padded_mode = np.zeros(target_length)
                        padded_mode[:len(mode)] = mode
                        adjusted_modes.append(padded_mode)
                    else:
                        adjusted_modes.append(mode)
                modes = adjusted_modes
            
            normalized_prices = normalized_prices[:target_length]
            dates = analysis_data.index[:target_length]
            
            # Store results in session state
            st.session_state.vmd_results = {
                'modes': modes,
                'frequencies': frequencies,
                'normalized_prices': normalized_prices,
                'dates': dates,
                'original_data': analysis_data.iloc[:target_length]
            }
            st.session_state.last_vmd_params = current_vmd_params
            
        st.success(f"✅ VMD analysis complete! Extracted {K} modes for {current_data_len} data points")
    
    # Display VMD results
    if 'vmd_results' in st.session_state:
        results = st.session_state.vmd_results
        
        # VMD modes visualization
        if show_individual_modes:
            st.subheader("🌊 VMD Modes Analysis")
            
            # Individual modes chart (full width)
            vmd_fig = create_vmd_modes_chart(
                results['modes'], 
                results['frequencies'], 
                results['normalized_prices'],
                results['dates']
            )
            st.plotly_chart(vmd_fig, use_container_width=True)
            
            # Additional charts in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                freq_fig = create_mode_frequencies_chart(results['frequencies'])
                st.plotly_chart(freq_fig, use_container_width=True)
                
            with col2:
                error_fig = create_reconstruction_error_chart(
                    results['modes'], 
                    results['normalized_prices']
                )
                st.plotly_chart(error_fig, use_container_width=True)
        
        # Predictive stability analysis
        if show_predictive_analysis:
            with st.spinner("Calculating predictive stability metrics..."):
                analyzer = PredictiveStabilityAnalyzer(
                    results['modes'],
                    results['frequencies'],
                    results['normalized_prices']
                )
                
                utilities, metrics = analyzer.calculate_overall_predictive_utility()
                reconstruction_quality = analyzer.calculate_reconstruction_quality_by_modes()
            
            st.subheader("🎯 Predictive Stability Analysis")
            
            # Create metrics summary
            col1, col2, col3 = st.columns(3)
            
            best_utility_idx = np.argmax(utilities)
            best_trend_idx = np.argmax(metrics['trend_consistency'])
            best_stability_idx = np.argmax(metrics['stability'])
            
            with col1:
                st.metric(
                    "Best Overall Utility", 
                    f"Mode {best_utility_idx + 1}",
                    f"Score: {utilities[best_utility_idx]:.3f}"
                )
            
            with col2:
                st.metric(
                    "Most Consistent Trend", 
                    f"Mode {best_trend_idx + 1}",
                    f"Score: {metrics['trend_consistency'][best_trend_idx]:.3f}"
                )
            
            with col3:
                st.metric(
                    "Most Stable", 
                    f"Mode {best_stability_idx + 1}",
                    f"Score: {metrics['stability'][best_stability_idx]:.3f}"
                )
            
            # Detailed predictive analysis chart
            pred_fig = create_predictive_analysis_chart(utilities, metrics, reconstruction_quality)
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Trading Recommendations")
            
            if utilities[0] > 0.6:
                st.success("✅ **Mode 1 (Primary Trend)** shows excellent predictive utility - suitable for trend-following strategies")
            elif utilities[0] > 0.4:
                st.info("ℹ️ **Mode 1 (Primary Trend)** shows moderate predictive utility - use with caution")
            else:
                st.warning("⚠️ **Mode 1 (Primary Trend)** shows poor predictive utility - consider other indicators")
            
            if len(utilities) > 1:
                if utilities[1] > 0.5:
                    st.success("✅ **Mode 2 (Secondary Trend)** shows good utility for medium-term pattern recognition")
                
            # Mode stability ranking
            stability_ranking = sorted(enumerate(utilities), key=lambda x: x[1], reverse=True)
            
            st.subheader("📊 Mode Ranking by Predictive Utility")
            ranking_data = []
            for rank, (mode_idx, utility) in enumerate(stability_ranking):
                ranking_data.append({
                    'Rank': rank + 1,
                    'Mode': f'Mode {mode_idx + 1}',
                    'Utility Score': f'{utility:.3f}',
                    'Frequency': f'{results["frequencies"][mode_idx]:.3f} Hz',
                    'Trend Consistency': f'{metrics["trend_consistency"][mode_idx]:.3f}',
                    'Stability': f'{metrics["stability"][mode_idx]:.3f}'
                })
            
            st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)
            
            # Display basic stock info below analysis
            st.subheader("📊 Stock Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = data['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                price_change_pct = (price_change / data['Close'].iloc[0]) * 100
                st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:+.2f}%")
            
            with col3:
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatility (Annual)", f"{volatility:.1f}%")
            
            with col4:
                volume_avg = data['Volume'].mean()
                st.metric("Avg Volume", f"{volume_avg/1e6:.1f}M")
    
    # Additional information
    with st.expander("ℹ️ About VMD Analysis"):
        st.markdown("""
        **Variational Mode Decomposition (VMD)** is an advanced signal processing technique that decomposes a time series into intrinsic mode functions.
        
        **Predictive Stability Metrics:**
        - **Trend Consistency**: How stable the directional changes are
        - **Mode Stability**: Local variance analysis for predictability  
        - **Predictive Power**: Correlation with future values
        - **Separability**: How unique each mode is from others
        - **Reconstruction Quality**: How well modes recreate the original signal
        
        **Interpretation:**
        - **Mode 1**: Usually captures the main trend (low frequency)
        - **Higher Modes**: Capture oscillations and noise (higher frequencies)
        - **High Utility Scores** (>0.6): Suitable for trading strategies
        - **Low Utility Scores** (<0.3): Likely noise, avoid for predictions
        """)


if __name__ == "__main__":
    main()