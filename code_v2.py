import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Progress bar with ETA


# ==========================================
# 0️⃣ LOAD & CLEAN DATA
# ==========================================

# Read as string to avoid mixed-type warnings, then convert to numeric
pbv_df = pd.read_csv(
    "pricetobook_daily.csv",
    index_col=0,
    parse_dates=True,
    dtype=str,
    low_memory=False
)

mcap_df = pd.read_csv(
    "MarketCap.csv",
    index_col=0,
    parse_dates=True,
    dtype=str,
    low_memory=False
)

# Convert all data cells to numeric (invalid -> NaN)
pbv_df = pbv_df.apply(pd.to_numeric, errors="coerce")
mcap_df = mcap_df.apply(pd.to_numeric, errors="coerce")


# ==========================================
# 1️⃣ USER CONFIGURATION
# ==========================================
config = {
    # Multiple lookback periods in quarters (1Q ~ 90 days)
    "lookback_quarters_list": [1, 2, 3, 4, 8, 12, 16, 20],

    # "mean" or "median"
    "statistic": "mean",

    # Allowable NaN share in rolling window (30% → need 70% valid)
    "na_allowed_pct": 0.30,

    # Buy triggers: P/BV is X% below rolling average
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    # Holding periods (calendar days) - for time-based exit
    "holding_periods": {
        "1Q": 90,
        "2Q": 180,
        "3Q": 270,
        "4Q": 360,
        "2Y": 720,
        "3Y": 1080,
        "5Y": 1800,
        "7Y": 2520,
    },

    # Sell thresholds: P/BV is X% above rolling median - for threshold-based exit
    "sell_thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

    # Portfolio settings
    "initial_capital": 10_000_000,
    "max_positions": 200,  # Changed from 100 to 200
    "position_size_pct": 0.005,  # Changed from 0.01 to 0.005 (0.5%)

    # Output directory
    "output_dir": "backtest_results"
}


# ==========================================
# 2️⃣ DATA PREP HELPERS
# ==========================================

def align_data(pbv_df: pd.DataFrame, mcap_df: pd.DataFrame):
    """Align both dataframes on common dates and tickers."""
    common_index = pbv_df.index.intersection(mcap_df.index)
    common_cols = pbv_df.columns.intersection(mcap_df.columns)
    pbv_df = pbv_df.loc[common_index, common_cols].sort_index()
    mcap_df = mcap_df.loc[common_index, common_cols].sort_index()
    return pbv_df, mcap_df


def drop_high_na_stocks(pbv_df: pd.DataFrame, na_allowed_pct: float) -> pd.DataFrame:
    """
    Drop stocks where the share of NaN over the entire series exceeds na_allowed_pct.
    (If NaN > 30%, drop stock from further processing)
    """
    na_share = pbv_df.isna().mean(axis=0)  # per column
    keep_cols = na_share[na_share <= na_allowed_pct].index
    return pbv_df[keep_cols]


# ==========================================
# 3️⃣ ROLLING STAT CALC (CALENDAR-AWARE)
# ==========================================

def compute_rolling_stat(pbv_df: pd.DataFrame, cfg: dict, lookback_quarters: int) -> pd.DataFrame:
    """
    For each date:
      - Determine target start date = today - lookback_days
      - If missing, use nearest available date
      - Window = [start_date, day before today]
      - If >= (1 - na_allowed_pct) valid in window, compute mean/median, else NaN
    """
    lookback_days = lookback_quarters * 90
    stat = cfg["statistic"]
    na_limit = cfg["na_allowed_pct"]

    rolling_df = pd.DataFrame(index=pbv_df.index, columns=pbv_df.columns)

    for today in pbv_df.index:
        start_target = today - pd.Timedelta(days=lookback_days)

        # snap target to nearest available date in index
        if start_target not in pbv_df.index:
            idx = pbv_df.index.get_indexer([start_target], method="nearest")[0]
            start_date = pbv_df.index[idx]
        else:
            start_date = start_target

        # window: from start_date up to day before today
        window = pbv_df.loc[start_date:today].iloc[:-1]
        if window.empty:
            continue

        for stock in pbv_df.columns:
            series = window[stock]
            valid_ratio = series.notna().mean()
            if valid_ratio >= (1 - na_limit):
                if stat == "mean":
                    rolling_df.at[today, stock] = series.mean()
                else:
                    rolling_df.at[today, stock] = series.median()
            else:
                rolling_df.at[today, stock] = np.nan

    return rolling_df


# ==========================================
# 4️⃣ SIGNAL GENERATION
# ==========================================

def generate_signals(pbv_df: pd.DataFrame,
                     rolling_df: pd.DataFrame,
                     threshold: float,
                     lookback_quarters: int,
                     cfg: dict) -> pd.DataFrame:
    """
    Signal True when:
      current P/BV <= (1 - threshold) * rolling_stat
    and rolling_stat not NaN
    and date is at least lookback_days after start.
    """
    lookback_days = lookback_quarters * 90
    signals = (pbv_df <= (1 - threshold) * rolling_df) & rolling_df.notna()

    # No signal before full lookback period is available
    min_signal_date = pbv_df.index[0] + pd.Timedelta(days=lookback_days)
    signals.loc[signals.index < min_signal_date, :] = False

    return signals


# ==========================================
# 5️⃣ CORE BACKTEST (ONE STRATEGY)
# ==========================================

def run_single_strategy(pbv_df: pd.DataFrame,
                        mcap_df: pd.DataFrame,
                        signals: pd.DataFrame,
                        rolling_df: pd.DataFrame,
                        exit_method: str,  # "holding_period" or "sell_threshold"
                        exit_param: float,  # holding_days (int) or sell_threshold (float)
                        cfg: dict):

    initial_capital = cfg["initial_capital"]
    max_positions = cfg["max_positions"]
    pos_size = initial_capital * cfg["position_size_pct"]

    cash = initial_capital
    positions = {}  # stock -> dict(entry_date, shares, invested, entry_pbv, entry_rolling_stat)

    equity_records = []
    trades = []

    for date in pbv_df.index:

        # 1️⃣ Close positions based on exit method
        to_close = []
        for stock, pos in positions.items():
            should_close = False
            exit_reason = ""
            
            if exit_method == "holding_period":
                # Time-based exit
                days_held = (date - pos["entry_date"]).days
                if days_held >= exit_param:
                    should_close = True
                    exit_reason = "holding_period"
            elif exit_method == "sell_threshold":
                # Threshold-based exit: sell when P/BV goes X% above rolling median
                current_pbv = pbv_df.loc[date, stock]
                entry_rolling_stat = pos["entry_rolling_stat"]
                
                if pd.notna(current_pbv) and pd.notna(entry_rolling_stat) and entry_rolling_stat > 0:
                    # Check if current P/BV >= (1 + sell_threshold) * entry_rolling_stat
                    if current_pbv >= (1 + exit_param) * entry_rolling_stat:
                        should_close = True
                        exit_reason = "sell_threshold"
            
            if should_close:
                # Use forward-filled market cap up to this date
                series = mcap_df[stock].loc[:date].ffill()
                if series.dropna().empty:
                    continue
                mcap_today = series.iloc[-1]
                if mcap_today <= 0:
                    continue

                exit_value = pos["shares"] * mcap_today
                cash += exit_value

                trades.append({
                    "stock": stock,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "holding_days": (date - pos["entry_date"]).days,
                    "exit_reason": exit_reason,
                    "invested": pos["invested"],
                    "exit_value": exit_value,
                    "pnl": exit_value - pos["invested"],
                    "return_pct": exit_value / pos["invested"] - 1.0
                })
                to_close.append(stock)

        for stock in to_close:
            del positions[stock]

        # 2️⃣ Compute portfolio equity
        portfolio_value = cash
        for stock, pos in positions.items():
            series = mcap_df[stock].loc[:date].ffill()
            if series.dropna().empty:
                continue
            mcap_today = series.iloc[-1]
            if mcap_today <= 0:
                continue
            portfolio_value += pos["shares"] * mcap_today

        equity_records.append({"date": date, "equity": portfolio_value})

        # 3️⃣ Open new positions based on today's signals
        if len(positions) >= max_positions or cash < pos_size:
            continue

        todays_signals = signals.loc[date]
        candidate_stocks = todays_signals[todays_signals].index

        for stock in candidate_stocks:
            if stock in positions:
                continue
            if cash < pos_size:
                break

            series = mcap_df[stock].loc[:date].ffill()
            if series.dropna().empty:
                continue
            mcap_today = series.iloc[-1]
            if mcap_today <= 0:
                continue

            # Store entry P/BV and rolling stat for threshold-based exit
            entry_pbv = pbv_df.loc[date, stock]
            entry_rolling_stat = rolling_df.loc[date, stock]

            shares = pos_size / mcap_today
            positions[stock] = {
                "entry_date": date,
                "shares": shares,
                "invested": pos_size,
                "entry_pbv": entry_pbv,
                "entry_rolling_stat": entry_rolling_stat
            }
            cash -= pos_size

    equity_df = pd.DataFrame(equity_records).set_index("date")
    trades_df = pd.DataFrame(trades)

    return equity_df, trades_df


# ==========================================
# 6️⃣ PERFORMANCE METRICS
# ==========================================

def compute_performance_metrics(equity_df: pd.DataFrame,
                                trades_df: pd.DataFrame,
                                cfg: dict,
                                threshold: float,
                                exit_method: str,
                                exit_label: str,
                                lookback_quarters: int) -> dict:
    """
    Compute CAGR, max drawdown, Sharpe, win ratio, Calmar,
    initial and final equity, plus parameters.
    """
    if equity_df.empty:
        return {}

    equity = equity_df["equity"].astype(float)

    initial_value = equity.iloc[0]
    final_value = equity.iloc[-1]

    # Time in years using calendar days
    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = days / 365.25 if days > 0 else np.nan

    # CAGR
    if years > 0 and initial_value > 0:
        cagr = (final_value / initial_value) ** (1 / years) - 1
    else:
        cagr = np.nan

    # Daily returns
    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) > 1:
        mean_ret = daily_ret.mean()
        std_ret = daily_ret.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else np.nan
    else:
        sharpe = np.nan

    # Max drawdown (as negative number)
    running_max = equity.cummax()
    dd_series = (equity - running_max) / running_max
    max_drawdown = dd_series.min()  # <= 0

    # Calmar = CAGR / |MaxDD|
    if max_drawdown < 0:
        calmar = cagr / abs(max_drawdown) if not np.isnan(cagr) else np.nan
    else:
        calmar = np.nan

    # Win ratio (trade-level)
    if not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        total_trades = len(trades_df)
        win_ratio = wins / total_trades if total_trades > 0 else np.nan
    else:
        win_ratio = np.nan

    metrics = {
        "lookback_quarters": lookback_quarters,
        "threshold": threshold,
        "exit_method": exit_method,
        "exit_param": exit_label,
        "statistic": cfg["statistic"],
        "na_allowed_pct": cfg["na_allowed_pct"],
        "initial_capital": cfg["initial_capital"],
        "max_positions": cfg["max_positions"],
        "position_size_pct": cfg["position_size_pct"],
        "initial_value": initial_value,
        "final_value": final_value,
        "CAGR": cagr,
        "max_drawdown": max_drawdown,
        "Sharpe": sharpe,
        "win_ratio": win_ratio,
        "Calmar": calmar,
        "num_trades": 0 if trades_df is None else len(trades_df),
        "start_date": equity_df.index[0],
        "end_date": equity_df.index[-1],
    }

    return metrics


# ==========================================
# 7️⃣ SAVE RESULTS PER STRATEGY
# ==========================================

def save_strategy_results(strategy_id: str,
                          equity_df: pd.DataFrame,
                          trades_df: pd.DataFrame,
                          cfg: dict):
    """
    Create a folder per strategy:
      - equity_curve.csv
      - trades.csv
      - equity_curve.png
    """
    base_dir = cfg["output_dir"]
    folder = os.path.join(base_dir, strategy_id)
    os.makedirs(folder, exist_ok=True)

    # Save equity curve
    equity_path = os.path.join(folder, "equity_curve.csv")
    equity_df.to_csv(equity_path)

    # Save trades
    trades_path = os.path.join(folder, "trades.csv")
    trades_df.to_csv(trades_path, index=False)

    # Plot and save equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df.index, equity_df["equity"])
    plt.title(f"Equity Curve - {strategy_id}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plot_path = os.path.join(folder, "equity_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# ==========================================
# 8️⃣ MULTIPROCESSING TASK FUNCTION
# ==========================================

def run_single_strategy_combination(args):
    """
    Task function for multiprocessing.
    Handles both holding_period and sell_threshold exit methods.
    """
    (lookback_quarters, threshold, exit_method, exit_param, exit_label, 
     pbv_df, mcap_df, cfg) = args
    
    # Compute rolling stats for this lookback period
    rolling_df = compute_rolling_stat(pbv_df, cfg, lookback_quarters)
    
    # Generate buy signals
    signals = generate_signals(pbv_df, rolling_df, threshold, lookback_quarters, cfg)
    
    # Create strategy ID
    if exit_method == "holding_period":
        strategy_id = f"L{lookback_quarters}_th{int(threshold*100)}_HOLD_{exit_label}"
    else:
        strategy_id = f"L{lookback_quarters}_th{int(threshold*100)}_SELL_{int(exit_param*100)}"
    
    # Run backtest
    equity_df, trades_df = run_single_strategy(
        pbv_df, mcap_df, signals, rolling_df, exit_method, exit_param, cfg
    )
    
    # Compute metrics
    metrics = compute_performance_metrics(
        equity_df, trades_df, cfg, threshold, exit_method, exit_label, lookback_quarters
    )
    
    # Save results
    if metrics:
        save_strategy_results(strategy_id, equity_df, trades_df, cfg)
    
    return metrics


# ==========================================
# 9️⃣ MASTER RUNNER (PARALLEL PROCESSING)
# ==========================================

def run_full_backtest_and_export(pbv_df: pd.DataFrame,
                                 mcap_df: pd.DataFrame,
                                 cfg: dict):
    """
    - Aligns data
    - Drops stocks with >30% NaN overall
    - Creates all strategy combinations
    - Runs ALL (lookback x threshold x exit_method x exit_param) combos in parallel
    - Computes metrics
    - Writes:
        - master_results.csv
        - per-strategy folders with:
            - equity_curve.csv
            - trades.csv
            - equity_curve.png
    """
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 1. Align and clean
    pbv_df, mcap_df = align_data(pbv_df, mcap_df)
    pbv_df = drop_high_na_stocks(pbv_df, cfg["na_allowed_pct"])
    mcap_df = mcap_df[pbv_df.columns]

    # 2. Create all task combinations
    tasks = []
    for lookback_quarters in cfg["lookback_quarters_list"]:
        for threshold in cfg["thresholds"]:
            # Add holding period tasks
            for holding_label, holding_days in cfg["holding_periods"].items():
                tasks.append((
                    lookback_quarters, threshold, "holding_period", 
                    holding_days, holding_label, pbv_df, mcap_df, cfg
                ))
            
            # Add sell threshold tasks
            for sell_threshold in cfg["sell_thresholds"]:
                exit_label = f"{int(sell_threshold*100)}pct"
                tasks.append((
                    lookback_quarters, threshold, "sell_threshold",
                    sell_threshold, exit_label, pbv_df, mcap_df, cfg
                ))

    total = len(tasks)
    print(f"\n🚀 Running with {cpu_count()} cores... Total tasks: {total}\n")

    # 3. Run tasks in parallel with progress bar
    results = []
    with Pool(cpu_count()) as pool:
        for r in tqdm(pool.imap_unordered(run_single_strategy_combination, tasks),
                      total=len(tasks),
                      desc="📈 Backtesting Progress",
                      ncols=100,
                      smoothing=0.1):
            results.append(r)

    # 4. Save master CSV
    master_df = pd.DataFrame([r for r in results if r])
    master_path = os.path.join(cfg["output_dir"], "master_results.csv")
    master_df.to_csv(master_path, index=False)

    return master_df


# ==========================================
# 🔟 ACTUAL RUN
# ==========================================

if __name__ == "__main__":
    master_results = run_full_backtest_and_export(pbv_df, mcap_df, config)
    print("\n✅ Backtest complete.")
    print(master_results.head())
    print(f"\n📁 Results saved in folder: {config['output_dir']}")
    print(f"📊 Total strategies tested: {len(master_results)}")
