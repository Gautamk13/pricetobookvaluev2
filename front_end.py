# To run: streamlit run front_end.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="P/BV Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Load Benchmark File (NIFTY)
# =========================
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv("benchmark_nifty.csv", parse_dates=["date"])
        df = df.sort_values("date")
        df["benchmark_normalized"] = df["equity"] / df["equity"].iloc[0]
        return df
    except FileNotFoundError:
        st.warning("⚠️ Benchmark file not found. Benchmark comparison will be disabled.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading benchmark: {e}")
        return None

# =========================
# Load Master File (Summary)
# =========================
@st.cache_data
def load_master_results():
    try:
        # Read the CSV file - skip the first blank line
        # The file has a blank first line, then header on line 2
        df = pd.read_csv("master_results.csv", skiprows=[0])
        
        # Strip any whitespace from column names and values
        df.columns = df.columns.str.strip()
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows where all key columns are NaN
        key_cols = ["lookback_quarters", "threshold", "exit_method", "exit_param"]
        if all(col in df.columns for col in key_cols):
            df = df.dropna(subset=key_cols, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Convert numeric columns to proper types
        numeric_cols = [
            "lookback_quarters", "threshold", "CAGR", "Sharpe", "Calmar", 
            "max_drawdown", "Max Drawdown", "win_ratio", "num_trades", "Trades",
            "initial_value", "final_value", "initial_capital", "na_allowed_pct",
            "max_positions", "position_size_pct"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except FileNotFoundError:
        st.error("❌ master_results.csv not found. Please run the backtest first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

def load_equity_curve(strategy_id):
    path = f"backtest_results/{strategy_id}/equity_curve.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date")
            df["strategy_normalized"] = df["equity"] / df["equity"].iloc[0]
            return df
        except Exception as e:
            st.warning(f"Error loading equity curve: {e}")
            return None
    return None

def load_trades(strategy_id):
    path = f"backtest_results/{strategy_id}/trades.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
        except Exception as e:
            st.warning(f"Error loading trades: {e}")
            return None
    return None

def safe_format_percent(value):
    """Safely format a value as percentage, handling strings and NaN"""
    if pd.isna(value):
        return "N/A"
    try:
        # Convert to float if it's a string
        num_val = float(value) if isinstance(value, str) else value
        return f"{num_val:.2%}"
    except (ValueError, TypeError):
        return "N/A"

def safe_format_float(value, decimals=2):
    """Safely format a value as float, handling strings and NaN"""
    if pd.isna(value):
        return "N/A"
    try:
        # Convert to float if it's a string
        num_val = float(value) if isinstance(value, str) else value
        return f"{num_val:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

# =========================
# Streamlit Front-End
# =========================
st.markdown('<h1 class="main-header">📈 P/BV Backtest Result Dashboard</h1>', unsafe_allow_html=True)

# Load master results
master_df = load_master_results()

# Check if dataframe is empty
if master_df.empty:
    st.error("❌ Master results file is empty. Please regenerate with code_v2.py")
    st.stop()

# Check if dataframe has required columns
required_cols = ["lookback_quarters", "threshold", "exit_method", "exit_param"]
missing_cols = [col for col in required_cols if col not in master_df.columns]
if missing_cols:
    st.error(f"❌ Master results file is missing required columns: {missing_cols}")
    st.error(f"Found columns: {list(master_df.columns)}")
    st.error(f"DataFrame shape: {master_df.shape}")
    st.error("Please regenerate with code_v2.py")
    st.stop()

# Ensure threshold column has valid data
if master_df["threshold"].isna().all():
    st.error("❌ Threshold column contains only NaN values")
    st.stop()

# ---------------------- Sidebar Features ----------------------
st.sidebar.header("📊 Strategy Selection")

mode = st.sidebar.radio(
    "Select Mode:",
    ("Filter Manually", "Filter by Best"),
    help="Choose to manually filter strategies or find the best one by a metric"
)

# ---------------------- Mode 1: Manual Selection ----------------------
if mode == "Filter Manually":
    st.sidebar.subheader("🔍 Filter Parameters")
    
    # Get unique values safely and convert to proper types
    lookback_values = pd.to_numeric(master_df["lookback_quarters"].dropna(), errors='coerce').unique()
    threshold_values = pd.to_numeric(master_df["threshold"].dropna(), errors='coerce').unique()
    
    if len(lookback_values) == 0:
        st.error("❌ No lookback_quarters values found in data")
        st.stop()
    if len(threshold_values) == 0:
        st.error("❌ No threshold values found in data")
        st.stop()
    
    # Sort and format for display
    lookback_sorted = sorted([float(x) for x in lookback_values if pd.notna(x)])
    threshold_sorted = sorted([float(x) for x in threshold_values if pd.notna(x)])
    
    # Create display labels (integers for lookback, percentages for threshold)
    lookback_options = [int(x) if x == int(x) else x for x in lookback_sorted]
    threshold_options = threshold_sorted
    
    lookback = st.sidebar.selectbox(
        "Select Lookback Quarters", 
        lookback_options,
        format_func=lambda x: f"{int(x)}" if x == int(x) else f"{x}",
        help="Number of quarters to look back for P/BV calculation"
    )
    threshold = st.sidebar.selectbox(
        "Select Threshold", 
        threshold_options,
        format_func=lambda x: f"{x:.0%}",
        help="P/BV threshold for buy signal (lower is better)"
    )
    
    # Filter by exit method
    exit_method = st.sidebar.selectbox(
        "Select Exit Method",
        ["holding_period", "sell_threshold"],
        help="Time-based exit (holding_period) or profit target exit (sell_threshold)"
    )
    
    # Get available exit params for this method
    # Convert filter values to match dataframe types
    lookback_float = float(lookback)
    threshold_float = float(threshold)
    
    method_df = master_df[
        (pd.to_numeric(master_df["lookback_quarters"], errors='coerce') == lookback_float) &
        (pd.to_numeric(master_df["threshold"], errors='coerce') == threshold_float) &
        (master_df["exit_method"].astype(str) == str(exit_method))
    ]
    
    if method_df.empty:
        st.warning("⚠️ No strategies found with these parameters.")
        selected = pd.DataFrame()
    else:
        exit_params = sorted(method_df["exit_param"].unique())
        
        if exit_method == "holding_period":
            exit_param = st.sidebar.selectbox(
                "Select Holding Period", 
                exit_params,
                help="Time period to hold the position (1Q=1 Quarter, 2Y=2 Years, etc.)"
            )
        else:
            exit_param = st.sidebar.selectbox(
                "Select Profit Target", 
                exit_params,
                help="Profit percentage target to exit (e.g., 10pct = 10% gain)"
            )
        
        # Convert filter values to match dataframe types
        lookback_float = float(lookback)
        threshold_float = float(threshold)
        
        # Filter with proper type matching
        # Filter with proper type matching
        # Convert columns to numeric for comparison
        master_df_filtered = master_df.copy()
        master_df_filtered["lookback_quarters_num"] = pd.to_numeric(master_df_filtered["lookback_quarters"], errors='coerce')
        master_df_filtered["threshold_num"] = pd.to_numeric(master_df_filtered["threshold"], errors='coerce')
        
        selected = master_df_filtered[
            (master_df_filtered["lookback_quarters_num"] == lookback_float) &
            (master_df_filtered["threshold_num"] == threshold_float) &
            (master_df_filtered["exit_method"].astype(str).str.strip() == str(exit_method).strip()) &
            (master_df_filtered["exit_param"].astype(str).str.strip() == str(exit_param).strip())
        ]
        
        # Drop the temporary numeric columns
        if not selected.empty:
            selected = selected.drop(columns=["lookback_quarters_num", "threshold_num"], errors='ignore')
        else:
            st.warning(f"⚠️ No strategy found with these exact parameters.")
            st.info(f"Debug: Lookback={lookback_float}, Threshold={threshold_float}, Exit Method={exit_method}, Exit Param={exit_param}")

# ---------------------- Mode 2: Best Strategy Finder ----------------------
else:
    st.sidebar.subheader("🏆 Find Best Strategy")
    
    # Get available metrics
    available_metrics = []
    metric_mapping = {}
    
    if "CAGR" in master_df.columns:
        available_metrics.append("CAGR")
        metric_mapping["CAGR"] = "CAGR"
    if "Sharpe" in master_df.columns:
        available_metrics.append("Sharpe Ratio")
        metric_mapping["Sharpe Ratio"] = "Sharpe"
    if "Calmar" in master_df.columns:
        available_metrics.append("Calmar Ratio")
        metric_mapping["Calmar Ratio"] = "Calmar"
    if "max_drawdown" in master_df.columns:
        available_metrics.append("Max Drawdown (Minimize)")
        metric_mapping["Max Drawdown (Minimize)"] = "max_drawdown"
    elif "Max Drawdown" in master_df.columns:
        available_metrics.append("Max Drawdown (Minimize)")
        metric_mapping["Max Drawdown (Minimize)"] = "Max Drawdown"
    if "win_ratio" in master_df.columns:
        available_metrics.append("Win Ratio")
        metric_mapping["Win Ratio"] = "win_ratio"
    
    if not available_metrics:
        st.error("No valid metrics found in results file.")
        selected = pd.DataFrame()
    else:
        criteria = st.sidebar.selectbox(
            "Optimize by:", 
            available_metrics,
            help="Select the metric to optimize for"
        )
        metric_col = metric_mapping[criteria]
        
        # Filter out NaN values
        valid_df = master_df[master_df[metric_col].notna()]
        
        if valid_df.empty:
            st.warning("No valid strategies found.")
            selected = pd.DataFrame()
        else:
            if criteria == "Max Drawdown (Minimize)":
                # For drawdown, we want the least negative (closest to 0)
                selected = valid_df.loc[valid_df[metric_col].idxmax()]
            else:
                selected = valid_df.loc[valid_df[metric_col].idxmax()]
            
            selected = pd.DataFrame([selected])  # convert to DataFrame
            st.success(f"🏆 **Best Strategy Selected by {criteria}**")

# ---------------------- Display Strategy Results ----------------------
if selected.empty:
    st.warning("⚠️ No strategy found matching your criteria.")
    # Debug information
    with st.expander("🔍 Debug Information"):
        st.write("**Master DataFrame Info:**")
        st.write(f"Shape: {master_df.shape}")
        st.write(f"Columns: {list(master_df.columns)}")
        st.write(f"Sample rows:")
        st.dataframe(master_df.head())
else:
    # Debug: Show what was selected (can be removed later)
    if st.sidebar.checkbox("🔍 Show Debug Info", False):
        st.sidebar.write("**Selected Strategy:**")
        st.sidebar.dataframe(selected)
        st.sidebar.write(f"**Columns:** {list(selected.columns)}")
        st.sidebar.write(f"**Shape:** {selected.shape}")
    # Safely extract values with type conversion
    try:
        lookback_raw = selected["lookback_quarters"].values[0]
        threshold_raw = selected["threshold"].values[0]
        exit_method_raw = selected["exit_method"].values[0]
        exit_param_raw = selected["exit_param"].values[0]
        
        # Convert to proper types
        lookback = float(lookback_raw) if pd.notna(lookback_raw) else None
        threshold = float(threshold_raw) if pd.notna(threshold_raw) else None
        exit_method = str(exit_method_raw) if pd.notna(exit_method_raw) else ""
        exit_param = str(exit_param_raw) if pd.notna(exit_param_raw) else ""
        
        # Validate values
        if lookback is None or threshold is None or not exit_method or not exit_param:
            st.error(f"❌ Invalid strategy parameters extracted")
            st.error(f"Lookback: {lookback}, Threshold: {threshold}, Exit Method: {exit_method}, Exit Param: {exit_param}")
            st.stop()
            
    except (IndexError, KeyError, ValueError, TypeError) as e:
        st.error(f"❌ Error extracting strategy parameters: {e}")
        st.error(f"Selected dataframe shape: {selected.shape}")
        st.error(f"Selected columns: {list(selected.columns)}")
        if not selected.empty:
            st.error(f"First row: {selected.iloc[0].to_dict()}")
        st.stop()
    
    # Build strategy ID based on exit method
    # Convert lookback to int to avoid L1.0 format
    lookback_int = int(lookback)
    threshold_int = int(threshold * 100)
    
    if exit_method == "holding_period":
        strategy_id = f"L{lookback_int}_th{threshold_int}_{exit_param}"
    else:
        # exit_param is like "10pct", extract the number
        pct_value = exit_param.replace("pct", "")
        strategy_id = f"L{lookback_int}_th{threshold_int}_SELL_{pct_value}"
    
    # Header with Strategy ID
    st.markdown(f"### 📌 Strategy ID: `{strategy_id}`")
    
    # Display Strategy Parameters in a nice layout
    st.subheader("⚙️ Strategy Parameters")
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    with param_col1:
        lookback_display = str(lookback_int) if lookback_int is not None else "N/A"
        st.metric("Lookback Quarters", lookback_display, help="Historical quarters for P/BV calculation")
    
    with param_col2:
        try:
            if threshold is not None:
                threshold_pct = f"{threshold:.0%}"
            else:
                threshold_pct = "N/A"
        except (ValueError, TypeError):
            try:
                threshold_pct = f"{float(threshold)*100:.0f}%" if threshold is not None else "N/A"
            except:
                threshold_pct = "N/A"
        st.metric("Buy Threshold", threshold_pct, help="P/BV threshold to trigger buy")
    
    with param_col3:
        if exit_method:
            if exit_method == "holding_period":
                exit_method_display = "Time-Based Exit"
            elif exit_method == "sell_threshold":
                exit_method_display = "Profit Target Exit"
            else:
                exit_method_display = exit_method.replace("_", " ").title()
        else:
            exit_method_display = "N/A"
        st.metric("Exit Method", exit_method_display)
    
    with param_col4:
        if exit_method == "holding_period":
            exit_param_display = str(exit_param) if exit_param else "N/A"
            st.metric("Holding Period", exit_param_display, help="Time to hold position")
        else:
            if exit_param:
                pct_display = exit_param.replace("pct", "") + "%"
            else:
                pct_display = "N/A"
            st.metric("Profit Target", pct_display, help="Target profit percentage")

    # Display Performance Metrics
    st.subheader("📊 Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Handle both old and new column names
    cagr_col = "CAGR" if "CAGR" in selected.columns else None
    sharpe_col = "Sharpe" if "Sharpe" in selected.columns else None
    calmar_col = "Calmar" if "Calmar" in selected.columns else None
    dd_col = "max_drawdown" if "max_drawdown" in selected.columns else ("Max Drawdown" if "Max Drawdown" in selected.columns else None)
    trades_col = "num_trades" if "num_trades" in selected.columns else ("Trades" if "Trades" in selected.columns else None)
    
    with metrics_col1:
        if cagr_col:
            try:
                cagr_val = selected[cagr_col].values[0]
                cagr_display = safe_format_percent(cagr_val)
            except (IndexError, KeyError, ValueError, TypeError):
                cagr_display = "N/A"
            st.metric("CAGR", cagr_display, 
                     help="Compound Annual Growth Rate")
        else:
            st.metric("CAGR", "N/A", help="Compound Annual Growth Rate")
    
    with metrics_col2:
        if sharpe_col:
            try:
                sharpe_val = selected[sharpe_col].values[0]
                sharpe_display = safe_format_float(sharpe_val)
            except (IndexError, KeyError, ValueError, TypeError):
                sharpe_display = "N/A"
            st.metric("Sharpe Ratio", sharpe_display,
                     help="Risk-adjusted return measure")
        else:
            st.metric("Sharpe Ratio", "N/A", help="Risk-adjusted return measure")
    
    with metrics_col3:
        if calmar_col:
            try:
                calmar_val = selected[calmar_col].values[0]
                calmar_display = safe_format_float(calmar_val)
            except (IndexError, KeyError, ValueError, TypeError):
                calmar_display = "N/A"
            st.metric("Calmar Ratio", calmar_display,
                     help="Return to max drawdown ratio")
        else:
            st.metric("Calmar Ratio", "N/A", help="Return to max drawdown ratio")
    
    with metrics_col4:
        if dd_col:
            try:
                dd_val = selected[dd_col].values[0]
                dd_formatted = safe_format_percent(dd_val)
            except (IndexError, KeyError, ValueError, TypeError):
                dd_formatted = "N/A"
        else:
            dd_formatted = "N/A"
        st.metric("Max Drawdown", dd_formatted,
                 delta=dd_formatted if dd_formatted != "N/A" else None,
                 help="Maximum peak-to-trough decline")
    
    # Additional metrics in a second row
    metrics_col5, metrics_col6, metrics_col7, metrics_col8 = st.columns(4)
    with metrics_col5:
        if trades_col:
            try:
                trades_val = selected[trades_col].values[0]
                if pd.notna(trades_val):
                    trades_formatted = f"{int(float(trades_val)):,}"
                else:
                    trades_formatted = "N/A"
            except (ValueError, TypeError, KeyError, IndexError):
                trades_formatted = "N/A"
        else:
            trades_formatted = "N/A"
        st.metric("Total Trades", trades_formatted,
                 help="Total number of trades executed")
    with metrics_col6:
        if "win_ratio" in selected.columns:
            win_ratio_val = selected["win_ratio"].values[0]
            st.metric("Win Ratio", safe_format_percent(win_ratio_val),
                     help="Percentage of profitable trades")
    with metrics_col7:
        if "initial_value" in selected.columns:
            init_val = selected["initial_value"].values[0]
            try:
                init_formatted = f"₹{float(init_val):,.0f}" if pd.notna(init_val) else "N/A"
            except (ValueError, TypeError):
                init_formatted = "N/A"
            st.metric("Initial Capital", init_formatted)
    with metrics_col8:
        if "final_value" in selected.columns:
            final_val = selected["final_value"].values[0]
            try:
                final_formatted = f"₹{float(final_val):,.0f}" if pd.notna(final_val) else "N/A"
                if pd.notna(final_val) and "initial_value" in selected.columns:
                    init_val = selected["initial_value"].values[0]
                    delta_val = float(final_val) - float(init_val) if pd.notna(init_val) else None
                    delta_formatted = f"₹{delta_val:,.0f}" if delta_val is not None else None
                else:
                    delta_formatted = None
            except (ValueError, TypeError):
                final_formatted = "N/A"
                delta_formatted = None
            st.metric("Final Value", final_formatted, delta=delta_formatted)

    # Load Equity Curve and Trades
    equity_df = load_equity_curve(strategy_id)
    trades_df = load_trades(strategy_id)

    # Download Buttons
    st.subheader("📂 Download Strategy Files")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        if trades_df is not None:
            trades_csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trade Log (CSV)",
                data=trades_csv,
                file_name=f"{strategy_id}_trades.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Trade log not available.")
    
    with download_col2:
        if equity_df is not None:
            equity_csv = equity_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Equity Curve (CSV)",
                data=equity_csv,
                file_name=f"{strategy_id}_equity_curve.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Equity curve not available.")
    
    with download_col3:
        if equity_df is not None:
            # Create a summary CSV
            # Get values safely
            cagr_val = selected[cagr_col].values[0] if cagr_col else None
            sharpe_val = selected[sharpe_col].values[0] if sharpe_col else None
            calmar_val = selected[calmar_col].values[0] if calmar_col else None
            dd_val = selected[dd_col].values[0] if dd_col in selected.columns else None
            win_ratio_val = selected['win_ratio'].values[0] if 'win_ratio' in selected.columns else None
            trades_val = selected[trades_col].values[0] if trades_col in selected.columns else None
            init_val = selected['initial_value'].values[0] if 'initial_value' in selected.columns else None
            final_val = selected['final_value'].values[0] if 'final_value' in selected.columns else None
            
            summary_data = {
                'Metric': ['CAGR', 'Sharpe', 'Calmar', 'Max Drawdown', 'Win Ratio', 'Total Trades', 'Initial Value', 'Final Value'],
                'Value': [
                    safe_format_percent(cagr_val) if cagr_val is not None else "N/A",
                    safe_format_float(sharpe_val) if sharpe_val is not None else "N/A",
                    safe_format_float(calmar_val) if calmar_val is not None else "N/A",
                    safe_format_percent(dd_val) if dd_val is not None else "N/A",
                    safe_format_percent(win_ratio_val) if win_ratio_val is not None else "N/A",
                    f"{int(float(trades_val)):,}" if trades_val is not None and pd.notna(trades_val) else "N/A",
                    f"₹{float(init_val):,.0f}" if init_val is not None and pd.notna(init_val) else "N/A",
                    f"₹{float(final_val):,.0f}" if final_val is not None and pd.notna(final_val) else "N/A"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Summary (CSV)",
                data=summary_csv,
                file_name=f"{strategy_id}_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Plot Equity Curve
    if equity_df is not None:
        st.subheader("📉 Equity Curve")
        
        # Load benchmark if available
        benchmark_df = load_benchmark()
        
        # Create figure with better styling
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Normalize to percentage returns for better visualization
        initial_equity = equity_df["equity"].iloc[0]
        normalized_equity = equity_df["equity"] / initial_equity
        
        # Plot strategy
        ax.plot(equity_df["date"], normalized_equity, linewidth=2.5, label="Strategy", color="#1f77b4", alpha=0.9)
        
        # Plot benchmark if available
        if benchmark_df is not None:
            # Merge on date
            merged = pd.merge(equity_df[["date"]], benchmark_df[["date", "benchmark_normalized"]], 
                            on="date", how="inner")
            if not merged.empty:
                ax.plot(merged["date"], merged["benchmark_normalized"], 
                       linewidth=2, label="NIFTY Benchmark", linestyle="--", color="#ff7f0e", alpha=0.8)
        
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='Initial Capital')
        ax.set_title(f"Equity Curve - {strategy_id}", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylabel("Normalized Equity (Indexed to 1.0)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional statistics
        if benchmark_df is not None and not merged.empty:
            st.subheader("📊 Strategy vs Benchmark Comparison")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            strategy_return = (normalized_equity.iloc[-1] - 1) * 100
            benchmark_return = (merged["benchmark_normalized"].iloc[-1] - 1) * 100
            
            with comp_col1:
                st.metric("Strategy Return", f"{float(strategy_return):.2f}%")
            with comp_col2:
                st.metric("Benchmark Return", f"{float(benchmark_return):.2f}%")
            with comp_col3:
                excess_return = strategy_return - benchmark_return
                st.metric("Excess Return", f"{float(excess_return):.2f}%", 
                         delta=f"{float(excess_return):.2f}%")
    else:
        st.warning(f"⚠️ Equity curve not found for strategy: {strategy_id}")

# ================================
# 🚀 Performance Heatmap Section
# ================================
st.markdown("---")
st.subheader("🔥 Strategy Performance Heatmap")

# Filter by exit method for heatmap
heatmap_exit_method = st.selectbox(
    "Select Exit Method for Heatmap:",
    ["holding_period", "sell_threshold"],
    key="heatmap_exit",
    help="Choose which exit method to visualize in the heatmap"
)

# Filter data for heatmap
heatmap_df_filtered = master_df[master_df["exit_method"] == heatmap_exit_method]

if heatmap_df_filtered.empty:
    st.warning(f"⚠️ No strategies found for exit method: {heatmap_exit_method}")
else:
    # Get available metrics
    metric_options = []
    if "CAGR" in master_df.columns:
        metric_options.append("CAGR")
    if "Sharpe" in master_df.columns:
        metric_options.append("Sharpe")
    if "Calmar" in master_df.columns:
        metric_options.append("Calmar")
    if "max_drawdown" in master_df.columns:
        metric_options.append("max_drawdown")
    elif "Max Drawdown" in master_df.columns:
        metric_options.append("Max Drawdown")
    if "win_ratio" in master_df.columns:
        metric_options.append("win_ratio")
    
    if metric_options:
        metric = st.selectbox(
            "Select metric to visualize:",
            metric_options,
            help="Choose which performance metric to display in the heatmap"
        )
        
        # Pivot table: Rows = Lookback, Columns = Threshold
        # Average across exit params for same lookback/threshold combo
        heatmap_pivot = heatmap_df_filtered.pivot_table(
            index="lookback_quarters",
            columns="threshold",
            values=metric,
            aggfunc='mean'  # Average if multiple exit params exist
        )
        
        st.write(f"📊 Heatmap of **{metric}** by Lookback & Threshold (Exit Method: {heatmap_exit_method.replace('_', ' ').title()})")
        
        # Format display
        if metric in ["CAGR", "max_drawdown", "Max Drawdown", "win_ratio"]:
            display_df = heatmap_pivot.style.format("{:.2%}").background_gradient(cmap='RdYlGn' if metric != "max_drawdown" and metric != "Max Drawdown" else 'RdYlGn_r')
        else:
            display_df = heatmap_pivot.style.format("{:.2f}").background_gradient(cmap='RdYlGn')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Plot the heatmap with better styling
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Choose colormap based on metric
        if metric in ["CAGR", "Sharpe", "Calmar", "win_ratio"]:
            cmap = 'RdYlGn'
        else:
            cmap = 'RdYlGn_r'
        
        im = ax.imshow(heatmap_pivot, aspect='auto', cmap=cmap, interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, label=metric.replace('_', ' ').title(), pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        
        # Label formatting
        ax.set_xticks(range(len(heatmap_pivot.columns)))
        ax.set_xticklabels([f"{x:.0%}" for x in heatmap_pivot.columns], rotation=45, ha='right', fontsize=10)
        ax.set_xlabel("Buy Threshold", fontsize=12, fontweight='bold')
        
        ax.set_yticks(range(len(heatmap_pivot.index)))
        ax.set_yticklabels([f"{int(x)}Q" for x in heatmap_pivot.index], fontsize=10)
        ax.set_ylabel("Lookback Quarters", fontsize=12, fontweight='bold')
        
        ax.set_title(f"Heatmap of {metric.replace('_', ' ').title()} ({heatmap_exit_method.replace('_', ' ').title()})", 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add text annotations
        for i in range(len(heatmap_pivot.index)):
            for j in range(len(heatmap_pivot.columns)):
                value = heatmap_pivot.iloc[i, j]
                if pd.notna(value):
                    if metric in ["CAGR", "max_drawdown", "Max Drawdown", "win_ratio"]:
                        text = f"{value:.1%}"
                    else:
                        text = f"{value:.2f}"
                    ax.text(j, i, text, ha="center", va="center", 
                           color="white" if abs(value - heatmap_pivot.min().min()) < abs(value - heatmap_pivot.max().max()) else "black",
                           fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠️ No valid metrics found for heatmap.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "📊 P/BV Backtest Dashboard | Generated with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
