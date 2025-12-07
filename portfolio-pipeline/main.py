import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition

import os
import sys
import subprocess

import sys
import os


# ------------------------------------------------------------
# Ensure plots render correctly in Colab
# ------------------------------------------------------------
in_colab = "google.colab" in sys.modules
if in_colab:
    print("Running in Colab — setting matplotlib to inline backend")
    matplotlib.use('module://matplotlib_inline.backend_inline')

plt.ion()  # Enable interactive mode

from IPython.display import display

# ------------------------------------------------------------
# List of tickers and date range
# ------------------------------------------------------------
tickers_list = ['AES','LNT','AEE','AEP','AWK','APD','ALB','AMCR','AVY','BALL','ALL', 'AON', 'CPAY', 'EG', 'IVZ']
start = '2022-01-01'
end = '2024-01-01'

# ------------------------------------------------------------
# Function to calculate monthly returns
# ------------------------------------------------------------
def calculate_monthly_returns(tickers_list, start_date, end_date):
    dow_prices = {}
    for t in tickers_list:
        try:
            df = yf.download(t, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=False)
            if not df.empty:
                dow_prices[t] = df
            else:
                print(f'Warning: no data returned for {t}')
        except Exception as e:
            print(f'Failed {t}: {e}')

    if not dow_prices:
        print("No stock data was downloaded. Please check tickers and dates.")
        return None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 1:
                return_data_dict[ticker] = returns

    if not return_data_dict:
        print("No valid stock data available after calculating returns.")
        return None

    daily_returns = pd.concat(return_data_dict.values(), axis=1, keys=return_data_dict.keys())
    monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
    return monthly_returns.dropna()

# ------------------------------------------------------------
# Optimization and plotting
# ------------------------------------------------------------
def optimize_and_plot_portfolio(df_returns, ipopt_executable):
    m = ConcreteModel()
    assets = df_returns.columns.tolist()
    m.Assets = Set(initialize=assets)
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0, 1))
    avg_returns = df_returns.mean().to_dict()
    m.mu = Param(m.Assets, initialize=avg_returns)
    cov_df = df_returns.cov()
    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=cov_dict)

    def total_return_rule(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)
    m.objective = Objective(rule=total_return_rule, sense=maximize)

    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1
    m.budget = Constraint(rule=budget_constraint_rule)

    print("Pyomo model initialized with sets, variables, parameters, objective, and budget constraint.")

    # Setup IPOPT solver
    solver = SolverFactory("ipopt")
    if not solver.available():
        solver = SolverFactory("ipopt", executable=ipopt_executable)

    max_possible_variance = np.max(np.diag(cov_df.values))
    max_risk_for_range = max_possible_variance * 1.5
    min_risk_for_range = 1e-6
    risk_limits = np.arange(min_risk_for_range, max_risk_for_range + 1e-6,
                            (max_risk_for_range - min_risk_for_range) / 200)

    param_analysis = {}
    returns = {}

    print(f"Starting portfolio optimization for {len(risk_limits)} risk levels...")
    for r in risk_limits:
        if hasattr(m, 'variance_constraint'):
            m.del_component(m.variance_constraint)

        def variance_constraint_rule(m):
            return sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets) <= r
        m.variance_constraint = Constraint(rule=variance_constraint_rule)

        result = solver.solve(m)

        if result.solver.termination_condition in [TerminationCondition.infeasible,
                                                   TerminationCondition.other]:
            continue

        if result.solver.termination_condition in [TerminationCondition.optimal,
                                                   TerminationCondition.locallyOptimal]:
            param_analysis[r] = [m.x[a]() for a in m.Assets]
            returns[r] = m.objective()

    df_results = pd.DataFrame({'Risk': list(returns.keys()), 'Return': list(returns.values())})
    df_results = df_results.sort_values(by='Risk')

    # Efficient frontier plot
    plt.figure(figsize=(10,6))
    plt.plot(df_results['Risk'], df_results['Return'], marker='o', linestyle='-')
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    display(plt.gcf())
    plt.close()

    df_allocations = pd.DataFrame(param_analysis).T
    df_allocations.columns = assets
    df_allocations['Risk'] = df_allocations.index

    # Asset allocation plot
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(df_allocations['Risk'], df_allocations[asset], label=str(asset), marker='o', markersize=4)
    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

    print("Portfolio optimization and plotting complete.")
    return df_results, df_allocations

print("Finished defining `optimize_and_plot_portfolio` function.")

# ------------------------------------------------------------
# Perform full portfolio analysis
# ------------------------------------------------------------
def perform_full_portfolio_analysis(tickers_list, start_date, end_date, ipopt_executable):
    print("Starting full portfolio analysis...")
    dow_prices = {}
    for t in tickers_list:
        try:
            df = yf.download(t, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=False)
            if not df.empty:
                dow_prices[t] = df
        except Exception as e:
            print(f'Failed {t}: {e}')

    if not dow_prices:
        print("No stock data downloaded.")
        return None, None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 1:
            return_data_dict[ticker] = returns

    df_returns = pd.concat(return_data_dict.values(), axis=1, keys=return_data_dict.keys())
    df_returns = (1 + df_returns).resample('ME').prod() - 1
    df_returns = df_returns.dropna()

    if df_returns.empty:
        print("No valid return data.")
        return None, None

    return optimize_and_plot_portfolio(df_returns, ipopt_executable)

print("Defined `perform_full_portfolio_analysis` function.")

# ------------------------------------------------------------
# Run workflow
# ------------------------------------------------------------
ipopt_executable = "/content/bin/ipopt"  # For Colab
df_returns = calculate_monthly_returns(tickers_list, start, end)

if df_returns is not None:
    df_results, df_allocations = optimize_and_plot_portfolio(df_returns, ipopt_executable)
    print("Functions called successfully and plots generated.")
else:
    print("No valid return data available; skipping optimization.")

# Full workflow example
my_tickers = ['GE','KO','NVDA']
my_start_date = '2020-01-01'
my_end_date = '2024-01-01'

final_df_results, final_df_allocations = perform_full_portfolio_analysis(
    my_tickers, my_start_date, my_end_date, ipopt_executable
)

if final_df_results is not None and final_df_allocations is not None:
    print("Final results and allocations obtained:")
    display(final_df_results.head())
    display(final_df_allocations.head())
else:
    print("Portfolio analysis did not produce results.")

#-------VERSION 2-------

import os
import math
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, Var, Set, Param, Constraint, Objective, SolverFactory, NonNegativeReals, Binary, minimize
from pyomo.opt import TerminationCondition

# CONFIG (edit as needed)
ENHANCED_OUTPUT_DIR = "output"                 # folder to save CSVs and plots
BONMIN_EXECUTABLE = None                       # e.g. "/usr/local/bin/bonmin" if not on PATH
TRAIN_START = "2024-01-01"                     # training window start
TRAIN_END = "2025-07-31"                       # training window end (inclusive)
TEST_START = "2025-08-01"                      # backtest start
TEST_END = "2025-12-31"                        # backtest end (change as desired)
MIN_WEIGHT = 0.02                              # 2%
MAX_WEIGHT = 0.20                              # 20%
MIN_SELECTED = 5                               # choose at least N stocks
REQUIRE_ONE_PER_SECTOR = False                 # set True if you provide sector_map
PLOT_DPI = 150

os.makedirs(ENHANCED_OUTPUT_DIR, exist_ok=True)

# ---- util: scrape S&P500 tickers + sectors from Wikipedia (optional) ----
def fetch_sp500_tickers_and_sectors():
    """
    Returns DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        sp500 = tables[0]
        sp500 = sp500.rename(columns={"Symbol":"ticker", "Security":"name", "GICS Sector":"sector", "GICS Sub-Industry":"subindustry"})
        sp500 = sp500[['ticker','name','sector','subindustry']]
        return sp500
    except Exception as e:
        print("Failed to fetch S&P500 table:", e)
        return None

# ---- util: fetch daily prices for a list of tickers ----
def fetch_daily_prices(tickers, start_date, end_date, interval='1d', auto_adjust=False):
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=auto_adjust)
            if df is None or df.empty:
                print(f"Warning: {t} returned no data")
            else:
                out[t] = df
        except Exception as e:
            print(f"Failed to download {t}: {e}")
    return out

# ---- prepare monthly returns DataFrame from daily price dictionary ----
def prices_to_monthly_returns(price_dict):
    series = {}
    for t, df in price_dict.items():
        if 'Close' not in df.columns:
            continue
        rets = df['Close'].pct_change().dropna()
        if len(rets) > 1:
            mon = (1 + rets).resample('ME').prod() - 1
            if not mon.empty:
                series[t] = mon
    if not series:
        return pd.DataFrame()
    df = pd.concat(series.values(), axis=1, keys=series.keys()).dropna()
    return df

# ---- Build Pyomo MINLP (binary + linking constraints) ----
def build_enhanced_minlp(tickers, mu_series, sigma_df, min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT,
                         min_selected=MIN_SELECTED, sector_map=None, require_one_per_sector=REQUIRE_ONE_PER_SECTOR, alpha=1.0):
    """
    Build model:
      - w[i] continuous in [0,1]
      - y[i] binary activation
      - linking: w[i] >= min_weight*y[i], w[i] <= max_weight*y[i]
      - budget: sum w = 1
      - cardinality: sum y >= min_selected
      - optional per-sector: sum y in sector >= 1
      - objective: minimize variance - alpha * expected_return
    """
    model = ConcreteModel()
    assets = list(tickers)
    model.Assets = Set(initialize=assets)
    model.w = Var(model.Assets, within=NonNegativeReals, bounds=(0,1))
    model.y = Var(model.Assets, domain=Binary)

    mu_dict = {a: float(mu_series[a]) for a in assets}
    sigma_dict = {(i,j): float(sigma_df.loc[i,j]) for i in assets for j in assets}
    model.mu = Param(model.Assets, initialize=mu_dict)
    model.Sigma = Param(model.Assets, model.Assets, initialize=sigma_dict)

    # budget
    model.budget = Constraint(expr = sum(model.w[a] for a in model.Assets) == 1.0)

    # linking constraints
    model.minlink = Constraint(model.Assets, rule=lambda m, a: m.w[a] >= min_weight * m.y[a])
    model.maxlink = Constraint(model.Assets, rule=lambda m, a: m.w[a] <= max_weight * m.y[a])

    # cardinality
    model.card = Constraint(expr = sum(model.y[a] for a in model.Assets) >= min_selected)

    # sector constraints
    if require_one_per_sector and (sector_map is not None):
        # sector_map must be in same order as tickers
        sectors = sorted(set(sector_map))
        for s in sectors:
            members = [t for t,sec in zip(assets, sector_map) if sec == s]
            if members:
                model.add_component(f"sector_{s.replace(' ','_')}", Constraint(expr = sum(model.y[m] for m in members) >= 1))

    # objective: variance - alpha * return
    def var_expr(m):
        return sum(m.w[i]*m.w[j]*m.Sigma[i,j] for i in m.Assets for j in m.Assets)
    def ret_expr(m):
        return sum(m.mu[i] * m.w[i] for i in m.Assets)
    model.obj = Objective(expr = var_expr(model) - alpha * ret_expr(model), sense=minimize)

    return model

# ---- Solve MINLP with Bonmin ----
def solve_with_bonmin(model, bonmin_executable=BONMIN_EXECUTABLE, tee=True):
    if bonmin_executable:
        solver = SolverFactory('bonmin', executable=bonmin_executable)
    else:
        solver = SolverFactory('bonmin')
    if not solver.available():
        print("Bonmin solver not available. Set BONMIN_EXECUTABLE to the full path of bonmin or install bonmin.")
        return None
    try:
        result = solver.solve(model, tee=tee)
    except Exception as e:
        print("Bonmin solve failed:", e)
        return None
    return result

# ---- extract solution to pandas Series (weights) ----
def extract_weights_from_model(model):
    w = {a: float(model.w[a]()) for a in model.Assets}
    s = pd.Series(w)
    # normalize small numerical errors
    if s.sum() > 0:
        s = s / s.sum()
    return s

# ---- equal weight helper ----
def equal_weight_solution(tickers, k=MIN_SELECTED):
    selected = tickers[:k]
    w = {t: 0.0 for t in tickers}
    for t in selected:
        w[t] = 1.0 / k
    return pd.Series(w)

# ---- backtest simple: apply daily returns to weights and compute cumulative ----
def backtest_weights(weights: pd.Series, start_date=TEST_START, end_date=TEST_END):
    if weights is None or weights.sum() <= 0:
        print("Invalid weights for backtest.")
        return None
    tickers = [t for t,w in weights.items() if w > 0]
    prices = fetch_daily_prices(tickers, start_date, end_date)
    if not prices:
        print("No price data for backtest tickers.")
        return None
    # align
    price_df = pd.concat([prices[t]['Close'].rename(t) for t in tickers], axis=1).dropna()
    if price_df.empty:
        print("Empty aligned price DataFrame for backtest.")
        return None
    daily_rets = price_df.pct_change().fillna(0)
    w_arr = np.array([weights.get(t,0.0) for t in price_df.columns], dtype=float)
    w_arr = w_arr / w_arr.sum()
    port_daily = daily_rets.values.dot(w_arr)
    port_series = pd.Series(port_daily, index=daily_rets.index, name="portfolio_return")
    cumulative = (1 + port_series).cumprod()
    summary = {
        "total_return": cumulative.iloc[-1] - 1.0,
        "annualized_return": (cumulative.iloc[-1]) ** (252.0 / len(port_series)) - 1.0 if len(port_series)>0 else np.nan,
        "annualized_vol": port_series.std() * np.sqrt(252)
    }
    return {"daily": port_series, "cumulative": cumulative, "summary": summary}

# ---- run 3 scenarios and save outputs ----
def run_enhanced_minlp(universe_tickers,
                       train_start=TRAIN_START, train_end=TRAIN_END,
                       test_start=TEST_START, test_end=TEST_END,
                       bonmin_executable=BONMIN_EXECUTABLE,
                       min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT,
                       min_selected=MIN_SELECTED, require_one_per_sector=REQUIRE_ONE_PER_SECTOR,
                       sector_map=None,
                       output_dir=ENHANCED_OUTPUT_DIR):
    """
    High-level runner:
     - prepares returns on training window
     - computes mu & sigma (annualized)
     - builds + solves MINLP for min-variance and max-return scenarios
     - equal weight scenario
     - backtests each scenario on test window
     - writes CSVs & plots to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1) prepare training returns
    print("Downloading training price data...")
    prices = fetch_daily_prices(universe_tickers, train_start, train_end)
    monthly = prices_to_monthly_returns(prices)
    if monthly.empty:
        raise RuntimeError("No monthly training return series - check tickers and dates.")
    # align tickers
    tickers = list(universe_tickers)
    monthly = monthly[tickers].dropna()
    if monthly.empty:
        raise RuntimeError("Training monthly returns empty after alignment/dropping NA.")

    # 2) compute mu and Sigma (annualize)
    mu_month = monthly.mean()
    sigma_month = monthly.cov()
    mu_ann = mu_month * 12.0
    sigma_ann = sigma_month * 12.0

    # Save mu & sigma for debugging
    mu_ann.to_csv(os.path.join(output_dir, "mu_annual.csv"))
    sigma_ann.to_csv(os.path.join(output_dir, "sigma_annual.csv"))

    scenarios = {}

    # Scenario 1: equal weight
    print("Scenario: equal-weight")
    w_eq = equal_weight_solution(tickers, k=min_selected)
    w_eq.name = "equal_weight"
    w_eq.to_csv(os.path.join(output_dir, "weights_equal_weight.csv"))
    scenarios['equal_weight'] = w_eq

    # Scenario 2: min-variance (alpha = 0)
    print("Scenario: min-variance (solving MINLP, may take time)...")
    model_minvar = build_enhanced_minlp(tickers, mu_ann, sigma_ann, min_weight=min_weight, max_weight=max_weight,
                                       min_selected=min_selected, sector_map=sector_map, require_one_per_sector=require_one_per_sector,
                                       alpha=0.0)
    res_minvar = solve_with_bonmin(model_minvar, bonmin_executable, tee=True)
    w_minvar = None
    if res_minvar is not None and res_minvar.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]:
        w_minvar = extract_weights_from_model(model_minvar)
        w_minvar.name = "min_variance"
        w_minvar.to_csv(os.path.join(output_dir, "weights_min_variance.csv"))
        scenarios['min_variance'] = w_minvar
    else:
        print("Min variance solve did not return optimal solution. Inspect solver logs.")

    # Scenario 3: max-return (encourage return)
    print("Scenario: max-return (solving MINLP, may take time)...")
    # allow concentration by setting linking bounds wide and min_selected small
    model_maxret = build_enhanced_minlp(tickers, mu_ann, sigma_ann, min_weight=0.0, max_weight=1.0,
                                       min_selected=1, sector_map=sector_map, require_one_per_sector=False,
                                       alpha=-50.0)
    res_maxret = solve_with_bonmin(model_maxret, bonmin_executable, tee=True)
    w_maxret = None
    if res_maxret is not None and res_maxret.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]:
        w_maxret = extract_weights_from_model(model_maxret)
        w_maxret.name = "max_return"
        w_maxret.to_csv(os.path.join(output_dir, "weights_max_return.csv"))
        scenarios['max_return'] = w_maxret
    else:
        print("Max-return solve did not return optimal solution. Inspect solver logs.")

    # Run backtests and save
    for name, w in scenarios.items():
        print(f"Backtesting scenario: {name}")
        bt = backtest_weights(w, start_date=test_start, end_date=test_end)
        if bt is None:
            print(f"Backtest failed for {name}")
            continue
        # save returns & cumulative & summary
        bt['daily'].to_csv(os.path.join(output_dir, f"backtest_daily_{name}.csv"))
        bt['cumulative'].to_csv(os.path.join(output_dir, f"backtest_cumulative_{name}.csv"))
        pd.Series(bt['summary']).to_csv(os.path.join(output_dir, f"backtest_summary_{name}.csv"))

        # save a small plot
        plt.figure(figsize=(8,4), dpi=PLOT_DPI)
        bt['cumulative'].plot(title=f"Cumulative Return: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"backtest_cumulative_{name}.png"))
        plt.close()

    print("Enhanced MINLP finished. Outputs saved to:", os.path.abspath(output_dir))
    return scenarios

