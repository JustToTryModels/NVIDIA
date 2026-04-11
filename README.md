<div align="center"> 
  
# 📈 NVIDIA Stock Price Prediction 
  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>
  
A comprehensive time-series forecasting project designed to predict the future stock price of **NVIDIA Corporation (NVDA)**. By analyzing over two decades of historical market data and benchmarking multiple statistical and deep learning models, this project identifies complex market patterns to forecast short-term price movements, empowering data-driven investment strategies.

<br>

---

## 📋 Table of Contents

- [Why NVIDIA?](#-why-nvidia)
- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Comparison & Selection](#-model-comparison--selection)
- [Final Model Results](#-final-model-results)
- [Key EDA Insights](#-key-eda-insights)
- [10-Day Future Forecast](#-10-day-future-forecast)
- [Installation & Usage](#-installation--usage)

<br>

---

## 🤖 Why NVIDIA?

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png" alt="NVIDIA Logo" width="500"/>
</div>

NVIDIA presents a compelling subject for financial forecasting due to its dominant role in the modern technological revolution:
- **The AI Boom:** NVIDIA GPUs (like A100 and H100) are the undisputed backbone for training Large Language Models (LLMs) used by Google, OpenAI, Microsoft, and Meta.
- **Extreme Growth & Volatility:** Evolving from a gaming hardware company to a trillion-dollar AI powerhouse has resulted in fascinating, highly volatile stock behavior perfect for complex machine learning analysis.

<br>

---

## 🎯 Project Overview

### Objective

To accurately forecast NVIDIA's daily closing stock price using historical trading data (1999–2024) by evaluating traditional time-series methods against modern deep learning architectures.

<div align="center">

### 🛣️ Approach

| Component | Description |
|-----------|-------------|
| **Evaluation Scope** | Benchmarked **6 Time-Series Models** |
| **Selected Model** | **LSTM** (Long Short-Term Memory Neural Network) |
| **Data Preprocessing** | MinMaxScaler, Differencing, Stationarity Checks (ADF) |
| **Hyperparameter Tuning** | Grid Search for ARIMA (p,d,q), Look-back windows for LSTM |
| **Forecasting Horizon** | Next 10 Business Days |
| **Deployment** | Streamlit Web Application |

</div>

<br>

---

## 🚀 Live Demo

Try the real-time prediction model here:

[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nvidia-stock-price-predictor.streamlit.app/) 

<div align="center">
  <table>
    <tr>
      <th>📈 Historical Trends Analysis</th>
      <th>🔮 Future Price Forecasting</th>
    </tr>
    <tr>
      <td><img src="https://raw.githubusercontent.com/MarpakaPradeepSai/Employee-Churn-Prediction/main/Data/Images%20&%20GIFs/placeholder_gif.gif" alt="Historical Analysis" width="470"/></td>
      <td><img src="https://raw.githubusercontent.com/MarpakaPradeepSai/Employee-Churn-Prediction/main/Data/Images%20&%20GIFs/placeholder_gif.gif" alt="Future Prediction" width="470"/></td>
    </tr>
  </table>
  <p><i>(Replace the placeholder image URLs above with actual GIFs of your Streamlit app)</i></p>
</div>

> Interact with historical data, apply moving averages, and visualize the LSTM model's 10-day future forecast!

<br>

---

## 📁 Project Structure

```
nvidia-stock-prediction/
├── Data/
│   └── NVIDIA.csv                          # Historical stock dataset (1999-2024)
├── Notebook/
│   └── NVIDIA_Stock_Analysis.ipynb         # EDA, Model Training, and Evaluation
├── Models/
│   └── NVIDIA_LSTM_LB(5)_U(150).keras      # Saved optimal LSTM model
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

<br>

---

## 📊 Dataset

<div align="center">

### Overview

| Metric | Value |
|--------|-------|
| **Timeframe** | Jan 22, 1999 – Aug 2, 2024 |
| **Total Trading Days** | 6,424 rows |
| **Missing Values** | 0 (Clean Dataset) |
| **Highest Recorded Price** | $140.76 |

</div>

<br>

<div align="center">

### Features

| Feature | Description |
|---------|-------------|
| `Date` | The specific trading day |
| `Open` | Price when the market opened |
| `High / Low` | Peak and lowest price during the trading session |
| `Close` | **Target Variable** - Final trading price at market close |
| `Adj Close` | Closing price adjusted for stock splits and dividends |
| `Volume` | Total number of shares traded |

</div>

<br>

---

## 🔬 Methodology

### 📊 1. Exploratory Data Analysis (EDA)
- **Distribution Analysis:** Histograms revealed heavily right-skewed price distributions and fat-tailed daily returns.
- **Volatility Tracking:** Calculated Daily (3.79%), Weekly (8.47%), and Annualized (60.16%) volatility.
- **Moving Averages:** Implemented 30-day, 50-day, and 200-day SMAs to identify underlying macro trends.

### ⚙️ 2. Data Preprocessing & Stationarity
- **ADF Testing:** Augmented Dickey-Fuller tests proved the original data was non-stationary.
- **Transformations:** Applied first-order differencing and MinMax scaling (0 to 1) to prepare data for neural networks and ARIMA.
- **Data Splitting:** Chronological split (80% Train : 20% Test) to prevent data leakage.

### 🧠 3. Model Engineering
- Iterative hyperparameter tuning for Statistical models (Alpha, Beta, Gamma).
- Look-back window optimization (5, 21, 63, 252 days) and unit optimization for deep learning.

<br>

---

## ⚔️ Model Comparison & Selection

To ensure optimal forecasting accuracy, **6 different time-series models** were rigorously benchmarked. 

### 🚀 Performance Leaderboard (Test Set)

<div align="center">

| Rank | Model | RMSE (Root Mean Squared Error) | Type |
|:---:|:---|:---:|:---:|
| **1** 🏆 | **LSTM Neural Network** | **1.32** | Deep Learning |
| 2 | Double Exponential Smoothing (Holt's) | 18.22 | Statistical |
| 3 | Triple Exponential Smoothing (Holt-Winters) | 18.26 | Statistical |
| 4 | Facebook Prophet | 31.29 | Additive / Machine Learning |
| 5 | ARIMA (4,0,3) | 34.73 | AutoRegressive Integrated |
| 6 | Simple Exponential Smoothing | 36.21 | Statistical |

</div>

<br>

### 🧠 Why LSTM Dominated

The **Long Short-Term Memory (LSTM)** model vastly outperformed traditional statistical models. Here is why:

1. **Non-Linear Complexity:** Stock markets are highly non-linear. While ARIMA and Smoothing techniques assume linear relationships, LSTM dynamically maps complex, hidden patterns in price action.
2. **Long-Term Dependencies:** LSTMs are uniquely designed with "memory cells" and gating mechanisms, allowing them to remember crucial price trends from weeks ago while forgetting irrelevant market noise.
3. **Adaptability to Extreme Volatility:** Traditional models struggled with NVIDIA's massive, sudden price spikes driven by AI news. LSTM successfully adapted to these regime changes.

<br>

---

## 📈 Final Model Results


<div align="center">

### Optimal LSTM Architecture

| Hyperparameter | Selected Value |
|----------------|----------------|
| **Look-back Window** | 5 Days (1 Trading Week) |
| **LSTM Units** | 150 |
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error (MSE) |

</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/MarpakaPradeepSai/Employee-Churn-Prediction/main/Data/Images%20&%20GIFs/placeholder_chart.png" alt="LSTM Actual vs Predicted" width="750"/>
  <p><i>(Add the plot image of your LSTM Actual vs Predicted output here)</i></p>
</div>

> ✅ **The LSTM model tracked NVIDIA's explosive growth trend with remarkable precision, achieving an error margin of just ~$1.32 per share.**

<br>

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights

Before forecasting, deep analysis of the historical data revealed several crucial characteristics of NVDA stock:

### 1. Deceptive Stability & "Fat Tails"
The distribution of daily returns is clustered tightly near zero, making the stock look stable on an average day. However, the distribution has **extreme fat tails** and high kurtosis (130.18). This means the stock's behavior is actually defined by rare, massive price jumps (both positive and negative)—classic high event-driven risk.

### 2. Extreme Volatility
*   **Daily Volatility:** 3.79%
*   **Annualized Volatility:** 60.16%
*   *Insight:* This confirms NVIDIA is a high-risk, high-reward asset. While quiet on most days, periodic dramatic events completely redefine its valuation.

### 3. Volume vs. Price Disconnect
Correlation analysis revealed a correlation coefficient of **-0.12** between Volume and Closing Price. Unlike many assets, a surge in trading volume for NVDA does not inherently predict a positive or negative price swing.

<br>

---

## 🔮 10-Day Future Forecast

Leveraging the winning LSTM model, we forecasted the next two trading weeks (10 business days) immediately following the dataset's end date (August 2, 2024).

<div align="center">

| Date | Forecasted Close Price ($) | Trend |
|:---:|:---:|:---:|
| 2024-08-05 | 113.83 | 🔼 |
| 2024-08-07 | 112.88 | 🔽 |
| 2024-08-09 | 111.03 | 🔽 |
| 2024-08-13 | 108.87 | 🔽 |
| 2024-08-16 | 105.50 | 🔽 |

</div>

**Strategic Takeaway:** The model detected a short-term cooling-off / correction period following recent highs, suggesting a slight downward trend over the 10-day horizon before stabilizing. *(Note: Stock predictions are for analytical purposes only, not financial advice).*

<br>

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/NVIDIA-Stock-Prediction.git
   cd NVIDIA-Stock-Prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in the default browser at `http://localhost:8501`

<br>

---

## 🙏 Thank You

<div align="center">
  <img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/thank-you-33.gif?raw=true" alt="Thank You" width="400">
  
  If you found this financial machine learning project insightful, please consider giving it a ⭐
</div>
