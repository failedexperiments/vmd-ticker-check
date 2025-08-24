# 🚀 Streamlit VMD Stock Analyzer

**Modern web application** for VMD stock analysis with predictive stability assessment.

## ✨ Key Features

### 📊 **Interactive Charts**
- **Candlestick charts** with Plotly (zoom, pan, hover)
- **Moving averages** MA20, MA50
- **Date range selection** for analysis
- **Real-time data updates**

### 🔬 **Advanced VMD Analysis**
- **Mode decomposition** with configurable parameters
- **Frequency visualization** for each mode
- **Cumulative signal reconstruction**
- **Real-time reconstruction error**

### 🎯 **Predictive Stability**
- **Trend Consistency** - stability of change directions
- **Mode Stability** - local variance analysis
- **Predictive Power** - correlation with future values
- **Separability** - uniqueness of each mode
- **Overall Utility Score** - general usefulness assessment

### 💡 **Intelligent Recommendations**
- **Automatic assessment** of each mode
- **Mode ranking** by usefulness
- **Strategic trading recommendations**
- **Color-coded quality indicators**

## 🚀 Quick Start

### 1. Installation
```bash
# Option A: Using batch file (Windows)
install_streamlit.bat

# Option B: Manual installation
pip install streamlit plotly yfinance scipy
```

### 2. Launch
```bash
streamlit run streamlit_vmd_analyzer.py
```
or alternatively
```bash
python -m streamlit run streamlit_vmd_analyzer.py
```

### 3. Open Browser
The application will automatically open at: `http://localhost:8501`

## 🎛️ How to Use

### **Step 1: Parameter Configuration**
**Sidebar → Analysis Parameters:**
- **Symbol**: Enter stock ticker (AAPL, TSLA, MSFT...)
- **Period**: Select historical period (1mo - 5y)
- **VMD Modes**: Number of modes to extract (2-8)
- **Alpha**: Bandwidth parameter (500-5000)

### **Step 2: Data Loading**
- Click **"📥 Load Data"**
- Review basic stock metrics
- Analyze candlestick chart with MA

### **Step 3: Range Selection (Optional)**
- Enable **"Enable Range Selection"**
- Choose start and end dates
- Focus analysis on specific period

### **Step 4: VMD Analysis**
- Click **"🔬 Run VMD Analysis"**
- Review mode decomposition
- Check frequencies and reconstruction quality

### **Step 5: Stability Assessment**
- Enable **"Show Predictive Analysis"**
- Analyze stability metrics
- Read automatic recommendations

## 📈 Results Interpretation

### **Mode Assessment:**
- **🟢 Score > 0.6**: Excellent utility - recommended for trading
- **🟡 Score 0.3-0.6**: Moderate utility - use with caution  
- **🔴 Score < 0.3**: Low utility - likely noise

### **VMD Modes:**
- **Mode 1**: Main trend (low frequency)
- **Mode 2-3**: Medium-term patterns
- **Higher modes**: Noise and short-term fluctuations

### **Strategic Recommendations:**
- **Trend Following**: Use Mode 1 with high Trend Consistency
- **Swing Trading**: Mode 2-3 with good Predictive Power
- **Avoid**: Modes with low Overall Utility Score

## 🔧 Advanced Features

### **Chart Customization**
- All charts are **fully interactive**
- **Zoom, pan, hover** for detailed analysis
- **Export to PNG/HTML** via Plotly menu

### **Session State**
- **Automatic saving** of analysis results
- **No need** to repeat calculations
- **Quick switching** between views

### **Responsive Design**
- **Screen adaptation** - works on desktop and mobile
- **Column layout** for readability
- **Expanders** for additional information

## 🎯 Benefits vs Tkinter

| Feature | Streamlit | Tkinter |
|---------|-----------|---------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Interactivity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Appearance** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Responsiveness** | ⭐⭐⭐⭐⭐ | ⭐ |
| **Shareable** | ⭐⭐⭐⭐⭐ | ⭐ |
| **Deployment** | ⭐⭐⭐⭐⭐ | ⭐ |

## 🚀 Next Steps

1. **Launch the application** and test with different stocks
2. **Experiment** with VMD parameters
3. **Compare results** for different periods
4. **Use recommendations** in real trading strategies

**Enjoy your advanced VMD analysis! 📊🎯**