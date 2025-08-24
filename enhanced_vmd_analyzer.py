# -*- coding: utf-8 -*-
"""
Enhanced Interactive VMD Stock Analyzer
Advanced UI with matplotlib navigation toolbar, proper candlesticks, and zoom/pan
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


class EnhancedVMDAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced Interactive VMD Stock Analyzer")
        self.master.geometry("1600x1000")
        
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
        ttk.Button(row2, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=(0, 5))
        
    def setup_charts_frame(self, parent):
        """Setup charts with navigation toolbar"""
        charts_frame = ttk.Frame(parent)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(16, 10), dpi=100)
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
        
        # Create grid layout: 3 rows, overlayed signals chart takes full width
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.3], hspace=0.3)
        
        # Main candlestick chart (top row, full width)
        self.ax_main = self.fig.add_subplot(gs[0])
        self.ax_main.set_title("Stock Price - Click and drag to select range for VMD analysis", fontsize=12, fontweight='bold')
        
        # VMD modes plot (middle, full width)
        self.ax_modes = self.fig.add_subplot(gs[1])
        self.ax_modes.set_title("Individual VMD Modes", fontsize=10)
        
        # Statistics panel (bottom, full width)
        self.ax_stats = self.fig.add_subplot(gs[2])
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
        
        # Plot candlesticks with better styling
        for i, (date, o, h, l, c, v) in enumerate(zip(dates, opens, highs, lows, closes, volumes)):
            color = '#2ECC71' if c >= o else '#E74C3C'  # Green for bullish, red for bearish
            alpha = min(1.0, v / np.max(volumes) * 0.5 + 0.3)  # Volume-based transparency
            
            # High-low line (wick)
            self.ax_main.plot([i, i], [l, h], color='black', linewidth=0.8, alpha=0.8)
            
            # Body rectangle
            height = abs(c - o)
            bottom = min(c, o)
            
            if height > 0:  # Avoid zero-height rectangles
                rect = Rectangle((i-0.4, bottom), 0.8, height, 
                               facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha)
                self.ax_main.add_patch(rect)
            else:
                # Doji - draw a line
                self.ax_main.plot([i-0.4, i+0.4], [c, c], color=color, linewidth=2)
        
        # Add moving averages for context
        ma20 = self.data['Close'].rolling(window=20).mean()
        ma50 = self.data['Close'].rolling(window=50).mean()
        
        self.ax_main.plot(range(len(ma20)), ma20, color='orange', linewidth=1.5, alpha=0.7, label='MA20')
        self.ax_main.plot(range(len(ma50)), ma50, color='purple', linewidth=1.5, alpha=0.7, label='MA50')
        
        # Add cumulative reconstruction if VMD analysis was performed
        if hasattr(self, 'cumulative_reconstruction') and self.selected_range:
            x_start, x_end = self.selected_range
            x_range = range(x_start, x_end + 1)
            
            # Scale reconstruction back to price levels
            selected_closes = closes[x_start:x_end + 1]
            mean_price = np.mean(selected_closes)
            std_price = np.std(selected_closes)
            scaled_reconstruction = self.cumulative_reconstruction * std_price + mean_price
            
            self.ax_main.plot(x_range, scaled_reconstruction, color='red', linewidth=2, 
                             alpha=0.8, label='VMD Reconstruction')
        
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
        """Setup enhanced range selection"""
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
        """Enhanced range selection handler"""
        x_start = int(max(0, min(eclick.xdata, erelease.xdata)))
        x_end = int(min(len(self.data)-1, max(eclick.xdata, erelease.xdata)))
        
        if x_end - x_start < 5:  # Minimum selection size
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
        
        # Clear cumulative reconstruction data
        if hasattr(self, 'cumulative_reconstruction'):
            delattr(self, 'cumulative_reconstruction')
        
        # Clear analysis plots
        self.ax_modes.clear()
        self.ax_modes.set_title("Individual VMD Modes", fontsize=10)
        self.ax_modes.text(0.5, 0.5, "Select a range and click 'Analyze VMD'", 
                          ha='center', va='center', transform=self.ax_modes.transAxes, fontsize=10)
        
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Redraw candlestick without reconstruction
        if self.data is not None:
            self.plot_candlestick()
        
        self.status_var.set("Selection cleared")
        self.canvas.draw()
    
    def analyze_vmd(self):
        """Enhanced VMD analysis with detailed visualization"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        if self.selected_range is None:
            messagebox.showwarning("Warning", "Please select a range first")
            return
        
        try:
            self.K = self.modes_var.get()
            alpha = self.alpha_var.get()
            x_start, x_end = self.selected_range
            
            # Show progress
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(10)
            self.status_var.set("Extracting selected data...")
            self.master.update()
            
            # Extract selected data
            selected_data = self.data.iloc[x_start:x_end+1]
            prices = selected_data['Close'].values
            
            self.progress_var.set(30)
            self.status_var.set("Normalizing data...")
            self.master.update()
            
            # Normalize prices
            normalized_prices = (prices - np.mean(prices)) / np.std(prices)
            
            self.progress_var.set(50)
            self.status_var.set("Running VMD decomposition...")
            self.master.update()
            
            # VMD decomposition
            modes, _, frequencies = vmd(normalized_prices, K=self.K, alpha=alpha)
            
            self.progress_var.set(80)
            self.status_var.set("Generating visualizations...")
            self.master.update()
            
            # Plot results
            self.plot_vmd_analysis(modes, frequencies, selected_data, normalized_prices)
            
            self.progress_var.set(100)
            self.status_var.set(f"VMD analysis complete - {self.K} modes extracted")
            
            # Hide progress bar
            self.master.after(1000, lambda: self.progress_bar.pack_forget())
            
        except Exception as e:
            messagebox.showerror("Error", f"VMD analysis failed: {str(e)}")
            self.status_var.set("VMD analysis failed")
            self.progress_bar.pack_forget()
    
    def plot_vmd_analysis(self, modes, frequencies, selected_data, normalized_prices):
        """Plot comprehensive VMD analysis results"""
        # Clear previous plots
        self.ax_modes.clear()
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Ensure all arrays have the same length
        mode_length = len(modes[0])
        original_length = len(normalized_prices)
        
        # Use the shorter length for consistency
        plot_length = min(mode_length, original_length)
        x_range = range(plot_length)
        dates = selected_data.index[:plot_length]
        
        # Truncate arrays to same length
        normalized_prices_plot = normalized_prices[:plot_length]
        modes_plot = [mode[:plot_length] for mode in modes]
        
        # Store cumulative reconstruction for main chart
        self.cumulative_reconstruction = np.sum(modes_plot, axis=0)
        
        # Individual modes plot (now takes full width)
        for i, (mode, freq) in enumerate(zip(modes_plot, frequencies)):
            color = self.colors[i % len(self.colors)]
            self.ax_modes.plot(x_range, mode, color=color, linewidth=1.5, 
                             label=f'Mode {i+1} (f={freq:.3f})', alpha=0.8)
        
        self.ax_modes.plot(x_range, normalized_prices_plot, color='black', linewidth=2, 
                          linestyle='--', alpha=0.7, label='Original')
        self.ax_modes.set_title("Individual VMD Modes", fontsize=10, fontweight='bold')
        self.ax_modes.set_ylabel("Amplitude", fontsize=8)
        self.ax_modes.set_xlabel("Time", fontsize=8)
        self.ax_modes.grid(True, alpha=0.3)
        self.ax_modes.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Statistics and information panel
        self.plot_statistics(modes_plot, frequencies, selected_data, normalized_prices_plot)
        
        # Format x-axis
        if len(dates) > 10:
            step = len(dates) // 8
            x_ticks = range(0, len(dates), step)
            x_labels = [dates[i].strftime('%m/%d') for i in x_ticks]
            self.ax_modes.set_xticks(x_ticks)
            self.ax_modes.set_xticklabels(x_labels, rotation=45, fontsize=7)
        
        # Update main chart to show reconstruction
        self.plot_candlestick()
        
        self.canvas.draw()
        
    def plot_statistics(self, modes, frequencies, selected_data, normalized_prices):
        """Plot statistics and information panel"""
        stats_text = []
        
        # Basic statistics
        price_change = ((selected_data['Close'].iloc[-1] - selected_data['Close'].iloc[0]) / 
                       selected_data['Close'].iloc[0] * 100)
        volatility = np.std(selected_data['Close'].pct_change().dropna()) * np.sqrt(252) * 100
        
        stats_text.append(f"Period: {selected_data.index[0].strftime('%Y-%m-%d')} to {selected_data.index[-1].strftime('%Y-%m-%d')}")
        stats_text.append(f"Price Change: {price_change:+.2f}%  |  Volatility: {volatility:.1f}%")
        
        # VMD mode analysis
        mode_energies = [np.sum(mode**2) for mode in modes]
        total_energy = sum(mode_energies)
        energy_percentages = [energy/total_energy * 100 for energy in mode_energies]
        
        mode_info = "  |  ".join([f"Mode {i+1}: {freq:.3f} Hz ({pct:.1f}%)" 
                                 for i, (freq, pct) in enumerate(zip(frequencies, energy_percentages))])
        stats_text.append(mode_info)
        
        # Reconstruction quality
        reconstruction = np.sum(modes, axis=0)
        mse = np.mean((normalized_prices - reconstruction)**2)
        correlation = np.corrcoef(normalized_prices, reconstruction)[0, 1]
        
        stats_text.append(f"Reconstruction - MSE: {mse:.4f}  |  Correlation: {correlation:.4f}")
        
        # Display statistics
        full_text = "\n".join(stats_text)
        self.ax_stats.text(0.5, 0.5, full_text, ha='center', va='center', 
                          transform=self.ax_stats.transAxes, fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
    def export_results(self):
        """Export analysis results"""
        if self.selected_range is None or not hasattr(self, 'vmd_modes'):
            messagebox.showwarning("Warning", "No analysis results to export")
            return
            
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")


def main():
    """Main function"""
    root = tk.Tk()
    app = EnhancedVMDAnalyzer(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1600 // 2)
    y = (root.winfo_screenheight() // 2) - (1000 // 2)
    root.geometry(f"1600x1000+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()