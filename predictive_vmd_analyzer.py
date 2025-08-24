# -*- coding: utf-8 -*-
"""
Predictive Stability VMD Analyzer
Advanced analysis of VMD modes for predictive stability and utility assessment
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import ttk, messagebox
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

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
            # Oblicz kierunki zmian (up/down)
            directions = np.diff(mode)
            direction_changes = np.sum(np.diff(np.sign(directions)) != 0)
            
            # Normalizuj przez długość sygnału
            consistency = 1.0 - (direction_changes / len(directions))
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
            
            # Stabilność = odwrotność zmienności lokalnej
            stability = 1.0 / (1.0 + np.mean(local_vars))
            stabilities.append(stability)
            
        return stabilities
    
    def calculate_predictive_power(self):
        """Moc predykcyjna - korelacja z przyszłymi wartościami"""
        powers = []
        
        for mode in self.modes:
            if len(mode) < 4:
                powers.append(0.0)
                continue
                
            # Korelacja z przesunięciem czasowym
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
            
            # Separowalność = 1 - średnia korelacja z innymi modami
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
            original_truncated = self.original[:len(cumulative)]
            cumulative_truncated = cumulative[:len(self.original)]
            
            # Use shorter length
            comparison_length = min(len(original_truncated), len(cumulative_truncated))
            original_comp = original_truncated[:comparison_length]
            cumulative_comp = cumulative_truncated[:comparison_length]
            
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
            quality = correlation - mse * 0.1  # Waga MSE
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
            # Kombinacja różnych metryk
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


class PredictiveVMDAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Predictive Stability VMD Analyzer")
        self.master.geometry("1800x1200")
        
        # Data storage
        self.data = None
        self.selected_range = None
        self.vmd_modes = None
        self.K = 4
        
        # Colors for modes
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Charts frame with navigation
        self.setup_charts_frame(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
        # Initialize plots
        self.setup_plots()
        
    def setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1 - Basic controls
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(row1, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
        self.symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(row1, textvariable=self.symbol_var, width=8).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(row1, text="Period:").pack(side=tk.LEFT, padx=(0, 5))
        self.period_var = tk.StringVar(value="3mo")
        period_combo = ttk.Combobox(row1, textvariable=self.period_var, width=8)
        period_combo['values'] = ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        period_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(row1, text="VMD Modes:").pack(side=tk.LEFT, padx=(0, 5))
        self.modes_var = tk.IntVar(value=4)
        ttk.Spinbox(row1, from_=2, to=8, textvariable=self.modes_var, width=5).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row1, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=(0, 5))
        
        # Row 2 - Analysis controls
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X)
        
        ttk.Label(row2, text="VMD Alpha:").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_var = tk.IntVar(value=2000)
        ttk.Scale(row2, from_=500, to=5000, orient=tk.HORIZONTAL, variable=self.alpha_var, length=150).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(row2, text="Clear Selection", command=self.clear_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="Analyze VMD", command=self.analyze_vmd).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="Predictive Analysis", command=self.analyze_predictive_stability).pack(side=tk.LEFT, padx=(0, 5))
        
    def setup_charts_frame(self, parent):
        """Setup charts with navigation toolbar"""
        charts_frame = ttk.Frame(parent)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with more space
        self.fig = plt.Figure(figsize=(18, 12), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        
        # Navigation toolbar
        toolbar_frame = ttk.Frame(charts_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready. Load stock data to begin.")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        
    def setup_plots(self):
        """Setup the plot layout"""
        self.fig.clear()
        
        # Create complex grid layout
        gs = self.fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 0.8], width_ratios=[2, 1, 1], hspace=0.35, wspace=0.3)
        
        # Main candlestick chart (top row, full width)
        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_main.set_title("Stock Price - Click and drag to select range for VMD analysis", fontsize=12, fontweight='bold')
        
        # VMD modes plot (row 2, left)
        self.ax_modes = self.fig.add_subplot(gs[1, 0])
        self.ax_modes.set_title("Individual VMD Modes", fontsize=10)
        
        # Cumulative plot (row 2, middle)  
        self.ax_cumulative = self.fig.add_subplot(gs[1, 1])
        self.ax_cumulative.set_title("Cumulative Reconstruction", fontsize=10)
        
        # Predictive utility (row 2, right)
        self.ax_utility = self.fig.add_subplot(gs[1, 2])
        self.ax_utility.set_title("Predictive Utility", fontsize=10)
        
        # Stability metrics (row 3, left)
        self.ax_stability = self.fig.add_subplot(gs[2, 0])
        self.ax_stability.set_title("Mode Stability Analysis", fontsize=10)
        
        # Trend consistency (row 3, middle)
        self.ax_trend = self.fig.add_subplot(gs[2, 1])
        self.ax_trend.set_title("Trend Consistency", fontsize=10)
        
        # Reconstruction quality (row 3, right)
        self.ax_quality = self.fig.add_subplot(gs[2, 2])
        self.ax_quality.set_title("Reconstruction Quality", fontsize=10)
        
        # Statistics panel (bottom, full width)
        self.ax_stats = self.fig.add_subplot(gs[3, :])
        self.ax_stats.axis('off')
        
        self.canvas.draw()
    
    def load_data(self):
        """Load stock data with progress indication"""
        try:
            symbol = self.symbol_var.get().upper()
            period = self.period_var.get()
            
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(20)
            self.status_var.set(f"Loading {symbol} data...")
            self.master.update()
            
            # Download data
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period, interval="1d")
            self.progress_var.set(60)
            self.master.update()
            
            if self.data.empty:
                messagebox.showerror("Error", f"No data found for symbol {symbol}")
                return
                
            self.progress_var.set(80)
            self.plot_candlestick()
            self.setup_range_selector()
            
            self.progress_var.set(100)
            self.status_var.set(f"Loaded {len(self.data)} candles for {symbol}")
            
            # Hide progress bar after completion
            self.master.after(1000, lambda: self.progress_bar.pack_forget())
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
            self.progress_bar.pack_forget()
    
    def plot_candlestick(self):
        """Plot enhanced candlestick chart"""
        self.ax_main.clear()
        
        dates = self.data.index
        opens = self.data['Open'].values
        highs = self.data['High'].values
        lows = self.data['Low'].values
        closes = self.data['Close'].values
        volumes = self.data['Volume'].values
        
        # Plot candlesticks
        for i, (date, o, h, l, c, v) in enumerate(zip(dates, opens, highs, lows, closes, volumes)):
            color = '#2ECC71' if c >= o else '#E74C3C'
            alpha = min(1.0, v / np.max(volumes) * 0.5 + 0.3)
            
            # High-low line (wick)
            self.ax_main.plot([i, i], [l, h], color='black', linewidth=0.8, alpha=0.8)
            
            # Body rectangle
            height = abs(c - o)
            bottom = min(c, o)
            
            if height > 0:
                rect = Rectangle((i-0.4, bottom), 0.8, height, 
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha)
                self.ax_main.add_patch(rect)
            else:
                self.ax_main.plot([i-0.4, i+0.4], [c, c], color=color, linewidth=2)
        
        # Add moving averages
        ma20 = self.data['Close'].rolling(window=20).mean()
        ma50 = self.data['Close'].rolling(window=50).mean()
        
        self.ax_main.plot(range(len(ma20)), ma20, color='orange', linewidth=1.5, alpha=0.7, label='MA20')
        self.ax_main.plot(range(len(ma50)), ma50, color='purple', linewidth=1.5, alpha=0.7, label='MA50')
        
        self.ax_main.set_title(f"{self.symbol_var.get()} - Click and drag to select range for VMD analysis", 
                              fontsize=12, fontweight='bold')
        self.ax_main.set_ylabel("Price ($)", fontsize=10)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='upper left', fontsize=8)
        
        # Format x-axis
        step = max(1, len(dates) // 15)
        x_ticks = range(0, len(dates), step)
        x_labels = [dates[i].strftime('%m/%d') for i in x_ticks]
        self.ax_main.set_xticks(x_ticks)
        self.ax_main.set_xticklabels(x_labels, rotation=45, fontsize=8)
        
        self.canvas.draw()
    
    def setup_range_selector(self):
        """Setup range selection"""
        self.selector = RectangleSelector(
            self.ax_main,
            self.on_range_select,
            useblit=True,
            button=[1],
            minspanx=3,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2)
        )
        
    def on_range_select(self, eclick, erelease):
        """Range selection handler"""
        x_start = int(max(0, min(eclick.xdata, erelease.xdata)))
        x_end = int(min(len(self.data)-1, max(eclick.xdata, erelease.xdata)))
        
        if x_end - x_start < 5:
            self.status_var.set("Selection too small. Please select at least 5 candles.")
            return
            
        self.selected_range = (x_start, x_end)
        
        # Show selection info
        selected_data = self.data.iloc[x_start:x_end+1]
        start_date = selected_data.index[0].strftime('%Y-%m-%d')
        end_date = selected_data.index[-1].strftime('%Y-%m-%d')
        price_change = ((selected_data['Close'].iloc[-1] - selected_data['Close'].iloc[0]) / 
                       selected_data['Close'].iloc[0] * 100)
        
        self.status_var.set(f"Selected: {start_date} to {end_date} ({x_end - x_start + 1} candles, {price_change:+.2f}%)")
        
    def clear_selection(self):
        """Clear selection and analysis results"""
        if hasattr(self, 'selector'):
            self.selector.set_visible(False)
            
        self.selected_range = None
        
        # Clear all analysis plots
        for ax in [self.ax_modes, self.ax_cumulative, self.ax_utility, self.ax_stability, self.ax_trend, self.ax_quality]:
            ax.clear()
            
        self.ax_modes.set_title("Individual VMD Modes", fontsize=10)
        self.ax_cumulative.set_title("Cumulative Reconstruction", fontsize=10)
        self.ax_utility.set_title("Predictive Utility", fontsize=10)
        self.ax_stability.set_title("Mode Stability Analysis", fontsize=10)
        self.ax_trend.set_title("Trend Consistency", fontsize=10)
        self.ax_quality.set_title("Reconstruction Quality", fontsize=10)
        
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        self.status_var.set("Selection cleared")
        self.canvas.draw()
    
    def analyze_vmd(self):
        """VMD analysis"""
        if self.data is None or self.selected_range is None:
            messagebox.showwarning("Warning", "Please load data and select a range first")
            return
        
        try:
            self.K = self.modes_var.get()
            alpha = self.alpha_var.get()
            x_start, x_end = self.selected_range
            
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(30)
            self.status_var.set("Running VMD decomposition...")
            self.master.update()
            
            # Extract and normalize data
            selected_data = self.data.iloc[x_start:x_end+1]
            prices = selected_data['Close'].values
            normalized_prices = (prices - np.mean(prices)) / np.std(prices)
            
            # VMD decomposition
            modes, _, frequencies = vmd(normalized_prices, K=self.K, alpha=alpha)
            
            self.progress_var.set(80)
            self.plot_vmd_results(modes, frequencies, selected_data, normalized_prices)
            
            self.progress_var.set(100)
            self.status_var.set(f"VMD analysis complete - {self.K} modes extracted")
            
            # Store for predictive analysis
            self.vmd_modes = modes
            self.vmd_frequencies = frequencies
            self.normalized_prices = normalized_prices
            
            self.master.after(1000, lambda: self.progress_bar.pack_forget())
            
        except Exception as e:
            messagebox.showerror("Error", f"VMD analysis failed: {str(e)}")
            self.status_var.set("VMD analysis failed")
            self.progress_bar.pack_forget()
    
    def plot_vmd_results(self, modes, frequencies, selected_data, normalized_prices):
        """Plot basic VMD results"""
        # Ensure consistent dimensions
        mode_length = len(modes[0])
        original_length = len(normalized_prices)
        plot_length = min(mode_length, original_length)
        
        x_range = range(plot_length)
        normalized_prices_plot = normalized_prices[:plot_length]
        modes_plot = [mode[:plot_length] for mode in modes]
        
        # Individual modes
        self.ax_modes.clear()
        for i, (mode, freq) in enumerate(zip(modes_plot, frequencies)):
            color = self.colors[i % len(self.colors)]
            self.ax_modes.plot(x_range, mode, color=color, linewidth=1.5, 
                             label=f'M{i+1} f={freq:.3f}', alpha=0.8)
        
        self.ax_modes.plot(x_range, normalized_prices_plot, color='black', linewidth=2, 
                          linestyle='--', alpha=0.7, label='Original')
        self.ax_modes.set_title("Individual VMD Modes", fontsize=10, fontweight='bold')
        self.ax_modes.grid(True, alpha=0.3)
        
        # Cumulative reconstruction
        self.ax_cumulative.clear()
        cumulative = np.zeros_like(modes_plot[0])
        
        for i, mode in enumerate(modes_plot):
            cumulative += mode
            color = self.colors[i % len(self.colors)]
            self.ax_cumulative.plot(x_range, cumulative, color=color, linewidth=2, 
                                  alpha=0.8, label=f'M1-{i+1}')
        
        self.ax_cumulative.plot(x_range, normalized_prices_plot, color='black', linewidth=2,
                               linestyle='--', alpha=0.7, label='Original')
        self.ax_cumulative.set_title("Cumulative VMD Reconstruction", fontsize=10, fontweight='bold')
        self.ax_cumulative.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def analyze_predictive_stability(self):
        """Analyze predictive stability and utility"""
        if not hasattr(self, 'vmd_modes') or self.vmd_modes is None:
            messagebox.showwarning("Warning", "Please run VMD analysis first")
            return
        
        try:
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(20)
            self.status_var.set("Analyzing predictive stability...")
            self.master.update()
            
            # Create analyzer
            analyzer = PredictiveStabilityAnalyzer(self.vmd_modes, self.vmd_frequencies, self.normalized_prices)
            
            self.progress_var.set(40)
            self.status_var.set("Calculating stability metrics...")
            self.master.update()
            
            # Calculate all metrics
            utilities, metrics = analyzer.calculate_overall_predictive_utility()
            reconstruction_quality = analyzer.calculate_reconstruction_quality_by_modes()
            
            self.progress_var.set(80)
            self.status_var.set("Creating visualizations...")
            self.master.update()
            
            # Plot results
            self.plot_predictive_analysis(utilities, metrics, reconstruction_quality)
            
            self.progress_var.set(100)
            self.status_var.set("Predictive stability analysis complete")
            
            self.master.after(1000, lambda: self.progress_bar.pack_forget())
            
        except Exception as e:
            messagebox.showerror("Error", f"Predictive analysis failed: {str(e)}")
            self.status_var.set("Predictive analysis failed")
            self.progress_bar.pack_forget()
    
    def plot_predictive_analysis(self, utilities, metrics, reconstruction_quality):
        """Plot comprehensive predictive analysis"""
        mode_names = [f'Mode {i+1}' for i in range(self.K)]
        
        # Overall predictive utility (bar chart)
        self.ax_utility.clear()
        bars = self.ax_utility.bar(mode_names, utilities, 
                                  color=[self.colors[i % len(self.colors)] for i in range(self.K)],
                                  alpha=0.7)
        self.ax_utility.set_title("Overall Predictive Utility", fontsize=10, fontweight='bold')
        self.ax_utility.set_ylabel("Utility Score", fontsize=8)
        self.ax_utility.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, utilities):
            height = bar.get_height()
            self.ax_utility.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Stability metrics comparison
        self.ax_stability.clear()
        x_pos = np.arange(self.K)
        width = 0.2
        
        self.ax_stability.bar(x_pos - width, metrics['stability'], width, 
                             label='Stability', alpha=0.7, color='blue')
        self.ax_stability.bar(x_pos, metrics['predictive_power'], width, 
                             label='Pred. Power', alpha=0.7, color='green')
        self.ax_stability.bar(x_pos + width, metrics['separability'], width, 
                             label='Separability', alpha=0.7, color='orange')
        
        self.ax_stability.set_title("Detailed Stability Metrics", fontsize=10, fontweight='bold')
        self.ax_stability.set_ylabel("Score", fontsize=8)
        self.ax_stability.set_xticks(x_pos)
        self.ax_stability.set_xticklabels([f'M{i+1}' for i in range(self.K)])
        self.ax_stability.grid(True, alpha=0.3, axis='y')
        
        # Trend consistency
        self.ax_trend.clear()
        trend_bars = self.ax_trend.bar(mode_names, metrics['trend_consistency'],
                                      color=[self.colors[i % len(self.colors)] for i in range(self.K)],
                                      alpha=0.7)
        self.ax_trend.set_title("Trend Consistency", fontsize=10, fontweight='bold')
        self.ax_trend.set_ylabel("Consistency Score", fontsize=8)
        self.ax_trend.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(trend_bars, metrics['trend_consistency']):
            height = bar.get_height()
            self.ax_trend.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Reconstruction quality progression
        self.ax_quality.clear()
        cumulative_modes = [f'M1-{i+1}' for i in range(self.K)]
        quality_line = self.ax_quality.plot(cumulative_modes, reconstruction_quality, 
                                           marker='o', linewidth=2, markersize=6)
        self.ax_quality.set_title("Cumulative Reconstruction Quality", fontsize=10, fontweight='bold')
        self.ax_quality.set_ylabel("Quality Score", fontsize=8)
        self.ax_quality.grid(True, alpha=0.3)
        
        # Add value labels
        for i, value in enumerate(reconstruction_quality):
            self.ax_quality.text(i, value + 0.02, f'{value:.3f}', 
                               ha='center', va='bottom', fontsize=8)
        
        # Statistics panel
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Create comprehensive statistics text
        best_utility_idx = np.argmax(utilities)
        best_trend_idx = np.argmax(metrics['trend_consistency'])
        best_stability_idx = np.argmax(metrics['stability'])
        
        stats_text = []
        stats_text.append(f"🎯 PREDICTIVE STABILITY ANALYSIS")
        stats_text.append(f"Best Overall Utility: Mode {best_utility_idx + 1} (Score: {utilities[best_utility_idx]:.3f})")
        stats_text.append(f"Most Consistent Trend: Mode {best_trend_idx + 1} (Score: {metrics['trend_consistency'][best_trend_idx]:.3f})")
        stats_text.append(f"Most Stable: Mode {best_stability_idx + 1} (Score: {metrics['stability'][best_stability_idx]:.3f})")
        
        # Mode recommendations
        if utilities[0] > 0.6:
            stats_text.append(f"✅ Mode 1 (Trend) shows good predictive utility - suitable for trend following")
        if len(utilities) > 1 and utilities[1] > 0.5:
            stats_text.append(f"✅ Mode 2 shows moderate utility - good for medium-term patterns")
        
        full_text = "\n".join(stats_text)
        self.ax_stats.text(0.5, 0.5, full_text, ha='center', va='center', 
                          transform=self.ax_stats.transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        self.canvas.draw()


def main():
    """Main function"""
    root = tk.Tk()
    app = PredictiveVMDAnalyzer(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1800 // 2)
    y = (root.winfo_screenheight() // 2) - (1200 // 2)
    root.geometry(f"1800x1200+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()