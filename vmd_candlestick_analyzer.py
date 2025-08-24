# -*- coding: utf-8 -*-
"""
Interactive VMD Stock Analyzer with Candlestick Charts
Enhanced UI with zoomable, pannable charts and range selection
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

def vmd(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7):
    """VMD implementation - copied from vmd_interactive_slider.py"""
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


class VMDCandlestickAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive VMD Stock Analyzer")
        self.master.geometry("1400x900")
        
        # Data storage
        self.data = None
        self.selected_range = None
        self.vmd_modes = None
        self.K = 4  # Number of VMD modes
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Symbol input
        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.symbol_var = tk.StringVar(value="AAPL")
        ttk.Entry(control_frame, textvariable=self.symbol_var, width=10).grid(row=0, column=1, padx=(0, 10))
        
        # Period selection
        ttk.Label(control_frame, text="Period:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.period_var = tk.StringVar(value="3mo")
        period_combo = ttk.Combobox(control_frame, textvariable=self.period_var, width=8)
        period_combo['values'] = ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        period_combo.grid(row=0, column=3, padx=(0, 10))
        
        # VMD modes
        ttk.Label(control_frame, text="VMD Modes:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.modes_var = tk.IntVar(value=4)
        ttk.Spinbox(control_frame, from_=2, to=8, textvariable=self.modes_var, width=5).grid(row=0, column=5, padx=(0, 10))
        
        # Buttons
        ttk.Button(control_frame, text="Load Data", command=self.load_data).grid(row=0, column=6, padx=(0, 5))
        ttk.Button(control_frame, text="Clear Selection", command=self.clear_selection).grid(row=0, column=7, padx=(0, 5))
        ttk.Button(control_frame, text="Analyze VMD", command=self.analyze_vmd).grid(row=0, column=8)
        
        # Charts frame
        charts_frame = ttk.Frame(main_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load stock data to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Initialize plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup the plot layout"""
        self.fig.clear()
        
        # Main candlestick chart (top 60%)
        self.ax_main = self.fig.add_subplot(3, 1, 1)
        self.ax_main.set_title("Stock Price - Select range for VMD analysis")
        
        # VMD modes subplot (bottom 40%)
        self.ax_vmd = self.fig.add_subplot(3, 1, (2, 3))
        self.ax_vmd.set_title("VMD Modes - Cumulative Overlay")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def load_data(self):
        """Load stock data"""
        try:
            symbol = self.symbol_var.get().upper()
            period = self.period_var.get()
            
            self.status_var.set(f"Loading {symbol} data...")
            self.master.update()
            
            # Download data
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period, interval="1d")
            
            if self.data.empty:
                messagebox.showerror("Error", f"No data found for symbol {symbol}")
                return
                
            self.status_var.set(f"Loaded {len(self.data)} candles for {symbol}")
            self.plot_candlestick()
            self.setup_range_selector()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def plot_candlestick(self):
        """Plot candlestick chart"""
        self.ax_main.clear()
        
        # Create OHLC data for candlestick
        dates = self.data.index
        opens = self.data['Open'].values
        highs = self.data['High'].values
        lows = self.data['Low'].values
        closes = self.data['Close'].values
        
        # Plot candlesticks manually
        for i, (date, o, h, l, c) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = 'green' if c >= o else 'red'
            
            # High-low line
            self.ax_main.plot([i, i], [l, h], color='black', linewidth=1)
            
            # Body rectangle
            height = abs(c - o)
            bottom = min(c, o)
            self.ax_main.bar(i, height, bottom=bottom, color=color, alpha=0.7, width=0.8)
        
        # Overlay close price line
        self.ax_main.plot(range(len(closes)), closes, color='blue', linewidth=1, alpha=0.5, label='Close Price')
        
        self.ax_main.set_title(f"{self.symbol_var.get()} - Select range for VMD analysis")
        self.ax_main.set_ylabel("Price")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend()
        
        # Format x-axis with dates (every 10th date)
        step = max(1, len(dates) // 10)
        x_ticks = range(0, len(dates), step)
        x_labels = [dates[i].strftime('%Y-%m-%d') for i in x_ticks]
        self.ax_main.set_xticks(x_ticks)
        self.ax_main.set_xticklabels(x_labels, rotation=45)
        
        self.canvas.draw()
    
    def setup_range_selector(self):
        """Setup interactive range selection"""
        self.selector = RectangleSelector(
            self.ax_main,
            self.on_range_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5,  # Minimum selection width
            spancoords='pixels',
            interactive=True
        )
        
    def on_range_select(self, eclick, erelease):
        """Handle range selection"""
        x_start = int(max(0, min(eclick.xdata, erelease.xdata)))
        x_end = int(min(len(self.data)-1, max(eclick.xdata, erelease.xdata)))
        
        self.selected_range = (x_start, x_end)
        
        # Highlight selected range
        if hasattr(self, 'selection_highlight'):
            self.selection_highlight.remove()
            
        y_min, y_max = self.ax_main.get_ylim()
        self.selection_highlight = self.ax_main.axvspan(
            x_start, x_end, alpha=0.3, color='yellow', label='Selected Range'
        )
        
        self.status_var.set(f"Selected range: {x_start} to {x_end} ({x_end - x_start + 1} candles)")
        self.canvas.draw()
    
    def clear_selection(self):
        """Clear the current selection"""
        if hasattr(self, 'selection_highlight'):
            self.selection_highlight.remove()
            
        self.selected_range = None
        self.ax_vmd.clear()
        self.ax_vmd.set_title("VMD Modes - Cumulative Overlay")
        
        self.status_var.set("Selection cleared")
        self.canvas.draw()
    
    def analyze_vmd(self):
        """Perform VMD analysis on selected range"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        if self.selected_range is None:
            messagebox.showwarning("Warning", "Please select a range first")
            return
        
        try:
            self.K = self.modes_var.get()
            x_start, x_end = self.selected_range
            
            # Extract selected data
            selected_data = self.data.iloc[x_start:x_end+1]
            prices = selected_data['Close'].values
            
            self.status_var.set("Running VMD analysis...")
            self.master.update()
            
            # Normalize prices for VMD
            normalized_prices = (prices - np.mean(prices)) / np.std(prices)
            
            # Perform VMD decomposition
            modes, _, frequencies = vmd(normalized_prices, K=self.K, alpha=2000)
            
            self.plot_vmd_results(modes, frequencies, x_start, x_end)
            self.status_var.set(f"VMD analysis complete - {self.K} modes extracted")
            
        except Exception as e:
            messagebox.showerror("Error", f"VMD analysis failed: {str(e)}")
            self.status_var.set("VMD analysis failed")
    
    def plot_vmd_results(self, modes, frequencies, x_start, x_end):
        """Plot VMD results with cumulative overlay"""
        self.ax_vmd.clear()
        
        x_range = range(x_end - x_start + 1)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot individual modes
        cumulative = np.zeros_like(modes[0])
        
        for i, mode in enumerate(modes):
            cumulative += mode
            
            # Plot individual mode (thin line)
            self.ax_vmd.plot(x_range, mode, 
                           color=colors[i % len(colors)], 
                           linewidth=1, 
                           alpha=0.6,
                           label=f'Mode {i+1} (f={frequencies[i]:.3f})')
            
            # Plot cumulative (thick line)
            self.ax_vmd.plot(x_range, cumulative, 
                           color=colors[i % len(colors)], 
                           linewidth=3, 
                           alpha=0.8,
                           label=f'Cumulative 1-{i+1}')
        
        # Add original normalized signal for comparison
        selected_data = self.data.iloc[x_start:x_end+1]
        prices = selected_data['Close'].values
        normalized_prices = (prices - np.mean(prices)) / np.std(prices)
        
        self.ax_vmd.plot(x_range, normalized_prices, 
                        color='black', 
                        linewidth=2, 
                        alpha=0.7, 
                        linestyle='--',
                        label='Original (normalized)')
        
        self.ax_vmd.set_title(f"VMD Decomposition - {self.K} Modes with Cumulative Overlay")
        self.ax_vmd.set_xlabel("Time Index")
        self.ax_vmd.set_ylabel("Normalized Amplitude")
        self.ax_vmd.grid(True, alpha=0.3)
        self.ax_vmd.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add frequency information
        freq_text = "Frequencies: " + ", ".join([f"f{i+1}={frequencies[i]:.3f}" for i in range(len(frequencies))])
        self.ax_vmd.text(0.02, 0.98, freq_text, 
                        transform=self.ax_vmd.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        verticalalignment='top')
        
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = VMDCandlestickAnalyzer(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1400 // 2)
    y = (root.winfo_screenheight() // 2) - (900 // 2)
    root.geometry(f"1400x900+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()