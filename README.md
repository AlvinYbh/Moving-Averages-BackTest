# Quantitative Backtest: Demystifying "Guru" Trading Strategies 📈

> An academic, data-driven approach to testing the viability of SMA & EMA Crossover strategies against real-world market friction.

🌙 **Note:** For the best visual experience of the interactive charts, please set your browser or system to **Dark Mode**.

## 📖 Project Overview
This project was developed as an independent quantitative research study. The goal is to demystify popular technical indicators (like the Simple Moving Average and Exponential Moving Average crossovers) heavily promoted by financial influencers. 

Instead of relying on theoretical perfect conditions, this backtest introduces real-world variables such as **slippage**, **transaction fees (0.1%)**, and **intraday stop-losses** across different market regimes (Bull, Bear, and Volatile ranges) to calculate the true mathematical expectancy of these strategies.

## ⚙️ How to Run the Code Locally

To explore the interactive dashboard and the data visualizations, follow these simple steps:

### Prerequisites
You need to have **Python** installed on your computer. An IDE like PyCharm, VS Code, or a simple terminal is sufficient.

### Step-by-step Installation
1. **Download the project**: Clone this repository to your local machine or download the ZIP file.
2. **Open your terminal**: Navigate to the folder where the project is saved.
3. **Install the dependencies**: The dashboard relies on several Python libraries. Install them by running this command in your terminal:
   ```bash
   pip install streamlit pandas numpy yfinance plotly

   streamlit run "BackTest.py"
