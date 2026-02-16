# P/BV Backtest Dashboard

A comprehensive backtesting system for Price-to-Book Value (P/BV) factor analysis with interactive Streamlit dashboard.

## Features

- **Factor Research Backtest**: Tests P/BV as a predictive metric
- **Multiple Exit Strategies**: Time-based and threshold-based exits
- **Comprehensive Parameter Sweep**: 1,024 strategy combinations
- **Interactive Dashboard**: Streamlit web app for results visualization
- **Performance Metrics**: CAGR, Sharpe, Calmar, Max Drawdown, Win Ratio

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run backtest:
```bash
python code_v2.py
```

3. Launch dashboard:
```bash
streamlit run front_end.py
```

## Streamlit Cloud Deployment

1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Set main file path: `front_end.py`
4. Deploy!

The app will be available at: `pricetobookvalue.streamlit.app`

## File Structure

- `code_v2.py` - Main backtesting engine with multiprocessing
- `front_end.py` - Streamlit dashboard
- `backtest_results/` - Generated results (CSV files and equity curves)
- `requirements.txt` - Python dependencies

## Notes

- Ensure `backtest_results/master_results.csv` exists before running the dashboard
- The dashboard requires the backtest to be completed first
- Large CSV files in `backtest_results/` are included in git for the dashboard



