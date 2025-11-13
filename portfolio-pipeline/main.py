import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition

import os
import sys
import subprocess

# def setup_ipopt_for_colab():
#     """
#     Automatically installs and configures IPOPT for Pyomo on Google Colab or local environments.
#     Detects where IPOPT was installed by IDAES and makes it executable.
#     """
#     import glob

#     ipopt_path = "/content/bin/ipopt"
#     in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

#     if in_colab:
#         print("Running in Colab environment — setting up IPOPT...")

#         # Install IDAES and IPOPT
#         subprocess.run(["pip", "install", "idaes-pse", "--pre", "-q"], check=True)
#         subprocess.run(["idaes", "get-extensions", "--to", "./bin"], check=True)

#         # ✅ Try to automatically detect the correct IPOPT path
#         ipopt_candidates = glob.glob("/content/bin/ipopt*")
#         if ipopt_candidates:
#             ipopt_path = ipopt_candidates[0]
#             print(f"Detected IPOPT binary at: {ipopt_path}")
#         else:
#             print("Could not detect IPOPT in /content/bin. Checking deeper...")
#             ipopt_candidates = glob.glob("/usr/local/lib*/idaes/bin/ipopt*")
#             if ipopt_candidates:
#                 ipopt_path = ipopt_candidates[0]
#                 print(f"Found IPOPT at: {ipopt_path}")
#             else:
#                 print("⚠️ IPOPT binary still not found — falling back to default path.")

#         # ✅ Fix permissions and PATH
#         if os.path.exists(ipopt_path):
#             subprocess.run(["chmod", "+x", ipopt_path], check=True)
#             os.environ["PATH"] += os.pathsep + os.path.dirname(ipopt_path)
#             print(f"IPOPT permissions fixed. PATH updated with {os.path.dirname(ipopt_path)}")
#             subprocess.run([ipopt_path, "--version"], check=False)
#         else:
#             print("⚠️ IPOPT not found even after searching. Optimization may fail.")
#     else:
#         print("Non-Colab environment detected — ensure IPOPT is installed locally.")

#     return ipopt_path



# ------------------------------------------------------------
# Function to calculate monthly returns
# ------------------------------------------------------------
tickers_list = ['AES','LNT','AEE','AEP','AWK','APD','ALB','AMCR','AVY','BALL','ALL', 'AON', 'CPAY', 'EG', 'IVZ']

start = '2022-01-01'
end = '2024-01-01'

def calculate_monthly_returns(tickers_list, start_date, end_date):
    """
    Downloads daily stock data, calculates daily and monthly returns.
    """
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
        print("No stock data was downloaded. Please check the ticker symbols and date range.")
        return None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 1:
                return_data_dict[ticker] = returns

    if not return_data_dict:
        print("No valid stock data available after calculating daily returns.")
        return None

    daily_returns = pd.concat(return_data_dict.values(), axis=1, keys=return_data_dict.keys())
    monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
    return monthly_returns.dropna()


# ------------------------------------------------------------
# Optimization and plotting
# ------------------------------------------------------------
def optimize_and_plot_portfolio(df_returns, ipopt_executable):
    from pyomo.environ import ConcreteModel, Set, Var, NonNegativeReals, Param, Objective, maximize, Constraint
    from pyomo.opt import SolverFactory, TerminationCondition

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

    solver = SolverFactory("ipopt")
    if not solver.available():
        solver = SolverFactory("ipopt", executable="/content/bin/ipopt")


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

    plt.figure(figsize=(10,6))
    plt.plot(df_results['Risk'], df_results['Return'], marker='o', linestyle='-')
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.show()

    df_allocations = pd.DataFrame(param_analysis).T
    df_allocations.columns = assets
    df_allocations['Risk'] = df_allocations.index

    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(df_allocations['Risk'], df_allocations[asset], label=str(asset), marker='o', markersize=4)
    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Portfolio optimization and plotting complete.")
    return df_results, df_allocations


print("Finished defining `optimize_and_plot_portfolio` function.")

ipopt_executable = "/content/bin/ipopt"
df_returns = calculate_monthly_returns(tickers_list, start, end)

if df_returns is not None:
    df_results, df_allocations = optimize_and_plot_portfolio(df_returns, ipopt_executable)
    print("Functions called successfully and plots generated.")
else:
    print("No valid return data available; skipping optimization.")


# ------------------------------------------------------------
# One concise function for full workflow
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
# Run full workflow
# ------------------------------------------------------------
my_tickers = ['GE','KO','NVDA']
my_start_date = '2020-01-01'
my_end_date = '2024-01-01'
my_ipopt_executable = ipopt_executable

from IPython.display import display
final_df_results, final_df_allocations = perform_full_portfolio_analysis(
    my_tickers, my_start_date, my_end_date, my_ipopt_executable
)

if final_df_results is not None and final_df_allocations is not None:
    print("Final results and allocations obtained:")
    display(final_df_results.head())
    display(final_df_allocations.head())
else:
    print("Portfolio analysis did not produce results.")
