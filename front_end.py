# To run: streamlit run front_end.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
<<<<<<< HEAD
=======
# Load Benchmark File (NIFTY)
# =========================
@st.cache_data
def load_benchmark():
    df = pd.read_csv("benchmark_nifty.csv", parse_dates=["date"])
    df = df.sort_values("date")
    df["benchmark_normalized"] = df["equity"] / df["equity"].iloc[0]
    return df

# =========================
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
# Load Master File (Summary)
# =========================
@st.cache_data
def load_master_results():
<<<<<<< HEAD
    try:
        df = pd.read_csv("backtest_results/master_results.csv")
        return df
    except FileNotFoundError:
        st.error("❌ master_results.csv not found. Please run the backtest first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")
        st.stop()
=======
    return pd.read_csv("backtest_results/master_results.csv")
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43

def load_equity_curve(strategy_id):
    path = f"backtest_results/{strategy_id}/equity_curve.csv"
    if os.path.exists(path):
<<<<<<< HEAD
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            st.warning(f"Error loading equity curve: {e}")
            return None
    return None

def load_trades(strategy_id):
    path = f"backtest_results/{strategy_id}/trades.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
=======
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date")
        df["strategy_normalized"] = df["equity"] / df["equity"].iloc[0]
        return df
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
    return None

# =========================
# Streamlit Front-End
# =========================
<<<<<<< HEAD
st.set_page_config(
    page_title="P/BV Backtest Dashboard",
    page_icon="📈",
    layout="wide"
)

=======
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
st.title("📈 P/BV Backtest Result Dashboard")

# Load master results
master_df = load_master_results()

<<<<<<< HEAD
# Check if dataframe has required columns
required_cols = ["lookback_quarters", "threshold", "exit_method", "exit_param"]
if not all(col in master_df.columns for col in required_cols):
    st.error("❌ Master results file is missing required columns. Please regenerate with code_v2.py")
    st.stop()

=======
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
# ---------------------- Sidebar Features ----------------------
st.sidebar.header("📊 Strategy Selection")

mode = st.sidebar.radio(
    "Select Mode:",
    ("Filter Manually", "Filter by Best")
)

# ---------------------- Mode 1: Manual Selection ----------------------
if mode == "Filter Manually":
<<<<<<< HEAD
    lookback = st.sidebar.selectbox(
        "Select Lookback Quarters", 
        sorted(master_df["lookback_quarters"].unique())
    )
    threshold = st.sidebar.selectbox(
        "Select Threshold", 
        sorted(master_df["threshold"].unique())
    )
    
    # Filter by exit method
    exit_method = st.sidebar.selectbox(
        "Select Exit Method",
        ["holding_period", "sell_threshold"]
    )
    
    # Get available exit params for this method
    method_df = master_df[
        (master_df["lookback_quarters"] == lookback) &
        (master_df["threshold"] == threshold) &
        (master_df["exit_method"] == exit_method)
    ]
    
    if method_df.empty:
        st.warning("No strategies found with these parameters.")
        selected = pd.DataFrame()
    else:
        exit_params = sorted(method_df["exit_param"].unique())
        exit_param = st.sidebar.selectbox("Select Exit Parameter", exit_params)
        
        selected = master_df[
            (master_df["lookback_quarters"] == lookback) &
            (master_df["threshold"] == threshold) &
            (master_df["exit_method"] == exit_method) &
            (master_df["exit_param"] == exit_param)
        ]
=======
    lookback = st.selectbox("Select Lookback Quarters", sorted(master_df["lookback_quarters"].unique()))
    threshold = st.selectbox("Select Threshold", sorted(master_df["threshold"].unique()))
    holding = st.selectbox("Select Holding Period", sorted(master_df["holding_period"].unique()))

    selected = master_df[
        (master_df["lookback_quarters"] == lookback) &
        (master_df["threshold"] == threshold) &
        (master_df["holding_period"] == holding)
    ]
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43

# ---------------------- Mode 2: Best Strategy Finder ----------------------
else:
    st.sidebar.write("📌 Choose Optimization Criteria")
<<<<<<< HEAD
    
    # Get available metrics (handle both old and new column names)
    available_metrics = []
    metric_mapping = {}
    
    if "CAGR" in master_df.columns:
        available_metrics.append("CAGR")
        metric_mapping["CAGR"] = "CAGR"
    if "Sharpe" in master_df.columns:
        available_metrics.append("Sharpe")
        metric_mapping["Sharpe"] = "Sharpe"
    if "Calmar" in master_df.columns:
        available_metrics.append("Calmar")
        metric_mapping["Calmar"] = "Calmar"
    if "max_drawdown" in master_df.columns:
        available_metrics.append("Max Drawdown (Minimize)")
        metric_mapping["Max Drawdown (Minimize)"] = "max_drawdown"
    elif "Max Drawdown" in master_df.columns:
        available_metrics.append("Max Drawdown (Minimize)")
        metric_mapping["Max Drawdown (Minimize)"] = "Max Drawdown"
    if "num_trades" in master_df.columns:
        available_metrics.append("Trades")
        metric_mapping["Trades"] = "num_trades"
    elif "Trades" in master_df.columns:
        available_metrics.append("Trades")
        metric_mapping["Trades"] = "Trades"
    
    if not available_metrics:
        st.error("No valid metrics found in results file.")
        selected = pd.DataFrame()
    else:
        criteria = st.sidebar.selectbox("Optimize by:", available_metrics)
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
            st.success(f"🏆 Best Strategy Selected by **{criteria}**")

# ---------------------- Display Strategy Results ----------------------
=======
    criteria = st.sidebar.selectbox(
        "Optimize by:",
        ["CAGR", "Sharpe", "Calmar", "Max Drawdown (Minimize)", "Trades"]
    )

    if criteria == "Max Drawdown (Minimize)":
        selected = master_df.loc[master_df["Max Drawdown"].idxmax()]
    else:
        selected = master_df.loc[master_df[criteria].idxmax()]

    selected = pd.DataFrame([selected])  # convert to DataFrame

    st.success(f"🏆 Best Strategy Selected by **{criteria}**")

# ---------------------- Display Strategy Results ----------------------

>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
if selected.empty:
    st.warning("No strategy found matching your criteria.")
else:
    lookback = selected["lookback_quarters"].values[0]
    threshold = selected["threshold"].values[0]
<<<<<<< HEAD
    exit_method = selected["exit_method"].values[0]
    exit_param = selected["exit_param"].values[0]
    
    # Build strategy ID based on exit method
    if exit_method == "holding_period":
        strategy_id = f"L{lookback}_th{int(threshold*100)}_HOLD_{exit_param}"
    else:
        # exit_param is like "10pct", extract the number
        pct_value = exit_param.replace("pct", "")
        strategy_id = f"L{lookback}_th{int(threshold*100)}_SELL_{pct_value}"
    
    st.write(f"📌 **Strategy ID:** `{strategy_id}`")
    
    # Display Strategy Parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lookback Quarters", lookback)
    with col2:
        st.metric("Buy Threshold", f"{threshold:.0%}")
    with col3:
        st.metric("Exit Method", exit_method.replace("_", " ").title())
    with col4:
        st.metric("Exit Parameter", exit_param)

    # Display Metrics
    st.subheader("📊 Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Handle both old and new column names
    cagr_col = "CAGR" if "CAGR" in selected.columns else None
    sharpe_col = "Sharpe" if "Sharpe" in selected.columns else None
    calmar_col = "Calmar" if "Calmar" in selected.columns else None
    dd_col = "max_drawdown" if "max_drawdown" in selected.columns else "Max Drawdown"
    trades_col = "num_trades" if "num_trades" in selected.columns else "Trades"
    
    with metrics_col1:
        if cagr_col:
            cagr_val = selected[cagr_col].values[0]
            st.metric("CAGR", f"{cagr_val:.2%}" if pd.notna(cagr_val) else "N/A")
    with metrics_col2:
        if sharpe_col:
            sharpe_val = selected[sharpe_col].values[0]
            st.metric("Sharpe Ratio", f"{sharpe_val:.2f}" if pd.notna(sharpe_val) else "N/A")
    with metrics_col3:
        if calmar_col:
            calmar_val = selected[calmar_col].values[0]
            st.metric("Calmar Ratio", f"{calmar_val:.2f}" if pd.notna(calmar_val) else "N/A")
    with metrics_col4:
        dd_val = selected[dd_col].values[0]
        st.metric("Max Drawdown", f"{dd_val:.2%}" if pd.notna(dd_val) else "N/A")
    
    # Additional metrics
    st.write("**Additional Metrics:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        trades_val = selected[trades_col].values[0]
        st.write(f"**Total Trades:** {int(trades_val) if pd.notna(trades_val) else 0}")
    with col2:
        if "win_ratio" in selected.columns:
            win_ratio_val = selected["win_ratio"].values[0]
            st.write(f"**Win Ratio:** {win_ratio_val:.2%}" if pd.notna(win_ratio_val) else "N/A")
    with col3:
        if "initial_value" in selected.columns and "final_value" in selected.columns:
            init_val = selected["initial_value"].values[0]
            final_val = selected["final_value"].values[0]
            st.write(f"**Initial:** ₹{init_val:,.0f}" if pd.notna(init_val) else "N/A")
            st.write(f"**Final:** ₹{final_val:,.0f}" if pd.notna(final_val) else "N/A")

    # Load Equity Curve
    equity_df = load_equity_curve(strategy_id)
    trades_df = load_trades(strategy_id)

    # Download Buttons
    st.subheader("📂 Download Strategy Files")
    
    col1, col2 = st.columns(2)
    with col1:
        if trades_df is not None:
            trades_csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trade Log (CSV)",
                data=trades_csv,
                file_name=f"{strategy_id}_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("Trade log not available.")
    
    with col2:
        if equity_df is not None:
            equity_csv = equity_df.to_csv().encode('utf-8')
=======
    holding = selected["holding_period"].values[0]
    strategy_id = f"L{lookback}_th{int(threshold*100)}_{holding}"

    st.write(f"📌 **Strategy ID:** `{strategy_id}`")

    # Display Metrics
    st.subheader("📊 Performance Metrics")
    st.write(f"**CAGR:** {selected['CAGR'].values[0]:.2%}")
    st.write(f"**Max Drawdown:** {selected['Max Drawdown'].values[0]:.2%}")
    st.write(f"**Sharpe Ratio:** {selected['Sharpe'].values[0]:.2f}")
    st.write(f"**Calmar Ratio:** {selected['Calmar'].values[0]:.2f}")
    st.write(f"**Trades:** {int(selected['Trades'].values[0])}")

    # Load Equity Curve
    equity_df = load_equity_curve(strategy_id)
    trades_path = f"backtest_results/{strategy_id}/trades.csv"

    # 🚀 New: Show Download Buttons if files exist
    st.subheader("📂 Download Strategy Files")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(trades_path):
            with open(trades_path, "rb") as f:
                st.download_button(
                    label="📥 Download Trade Log (CSV)",
                    data=f,
                    file_name=f"{strategy_id}_trades.csv",
                    mime="text/csv"
                )
        else:
            st.error("Trade log not found.")

    with col2:
        if equity_df is not None:
            equity_csv = equity_df.to_csv(index=False).encode('utf-8')
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
            st.download_button(
                label="📥 Download Equity Curve (CSV)",
                data=equity_csv,
                file_name=f"{strategy_id}_equity_curve.csv",
                mime="text/csv"
            )
        else:
<<<<<<< HEAD
            st.info("Equity curve not available.")

    # Plot Equity Curve
    if equity_df is not None:
        st.subheader("📉 Equity Curve")
        
        # Normalize to percentage returns for better visualization
        initial_equity = equity_df["equity"].iloc[0]
        normalized_equity = equity_df["equity"] / initial_equity
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity_df.index, normalized_equity, linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_title(f"Equity Curve - {strategy_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Equity (Normalized to 1.0)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning(f"Equity curve not found for strategy: {strategy_id}")
=======
            st.error("Equity curve not found.")

    # ======================
    # 📉 Plot Strategy vs Benchmark
    # ======================
    if equity_df is not None:

        benchmark_df = load_benchmark()

        # Merge both on date
        merged = pd.merge(equity_df, benchmark_df, on="date", how="inner")

        st.subheader("📈 Performance vs Benchmark")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(merged["date"], merged["strategy_normalized"], label="Strategy", linewidth=2)
        ax.plot(merged["date"], merged["benchmark_normalized"], label="NIFTY Benchmark", linestyle="--", linewidth=2)

        ax.set_title(f"Strategy vs Benchmark | {strategy_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value (Indexed to 1)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43

# ================================
# 🚀 Performance Heatmap Section
# ================================
st.subheader("🔥 Strategy Performance Heatmap")

<<<<<<< HEAD
# Filter by exit method for heatmap
heatmap_exit_method = st.selectbox(
    "Select Exit Method for Heatmap:",
    ["holding_period", "sell_threshold"],
    key="heatmap_exit"
)

# Filter data for heatmap
heatmap_df_filtered = master_df[master_df["exit_method"] == heatmap_exit_method]

if heatmap_df_filtered.empty:
    st.warning(f"No strategies found for exit method: {heatmap_exit_method}")
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
    if "num_trades" in master_df.columns:
        metric_options.append("num_trades")
    elif "Trades" in master_df.columns:
        metric_options.append("Trades")
    
    if metric_options:
        metric = st.selectbox(
            "Select metric to visualize:",
            metric_options
        )
        
        # Pivot table: Rows = Lookback, Columns = Threshold
        # Average across exit params for same lookback/threshold combo
        heatmap_pivot = heatmap_df_filtered.pivot_table(
            index="lookback_quarters",
            columns="threshold",
            values=metric,
            aggfunc='mean'  # Average if multiple exit params exist
        )
        
        st.write(f"📊 Heatmap of **{metric}** by Lookback & Threshold (Exit Method: {heatmap_exit_method})")
        
        # Format display
        if metric in ["CAGR", "max_drawdown", "Max Drawdown"]:
            display_df = heatmap_pivot.style.format("{:.2%}")
        else:
            display_df = heatmap_pivot.style.format("{:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_pivot, aspect='auto', cmap='RdYlGn' if metric in ["CAGR", "Sharpe", "Calmar"] else 'RdYlGn_r')
        fig.colorbar(im, ax=ax, label=metric)
        
        # Label formatting
        ax.set_xticks(range(len(heatmap_pivot.columns)))
        ax.set_xticklabels([f"{x:.0%}" for x in heatmap_pivot.columns], rotation=45)
        ax.set_xlabel("Buy Threshold", fontsize=12)
        
        ax.set_yticks(range(len(heatmap_pivot.index)))
        ax.set_yticklabels(heatmap_pivot.index)
        ax.set_ylabel("Lookback Quarters", fontsize=12)
        
        ax.set_title(f"Heatmap of {metric} ({heatmap_exit_method})", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No valid metrics found for heatmap.")
=======
metric = st.selectbox(
    "Select metric to visualize:",
    ["CAGR", "Sharpe", "Calmar", "Max Drawdown", "Trades"]
)

# Pivot table: Rows = Lookback, Columns = Threshold
heatmap_df = master_df.pivot_table(
    index="lookback_quarters",
    columns="threshold",
    values=metric
)

st.write(f"📊 Heatmap of **{metric}** by Lookback & Threshold")
st.dataframe(heatmap_df.style.format("{:.2%}" if metric != "Trades" else "{:.0f}"))

# Plot the heatmap
fig, ax = plt.subplots(figsize=(8, 5))
cax = ax.matshow(heatmap_df, interpolation='nearest')
fig.colorbar(cax)

# Label formatting
ax.set_xticks(range(len(heatmap_df.columns)))
ax.set_xticklabels(heatmap_df.columns, rotation=45)
ax.set_yticks(range(len(heatmap_df.index)))
ax.set_yticklabels(heatmap_df.index)
ax.set_xlabel("Threshold")
ax.set_ylabel("Lookback Quarters")
ax.set_title(f"Heatmap of {metric}")

st.pyplot(fig)
>>>>>>> d4aeb5712c4c0f51302106981294c50967235a43
