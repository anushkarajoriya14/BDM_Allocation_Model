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

import os
import sys
import subprocess

def setup_ipopt_for_colab():
    """
    Automatically installs and configures IPOPT for Pyomo on Google Colab or local environments.
    """
    ipopt_path = "/content/bin/ipopt"

    # Check both Colab runtime and file existence
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

    if in_colab:
        if not os.path.exists(ipopt_path):
            print("Installing IDAES and IPOPT solver for Colab environment...")
            subprocess.run(["pip", "install", "idaes-pse", "--pre"], check=True)
            subprocess.run(["idaes", "get-extensions", "--to", "./bin"], check=True)
        else:
            print("IPOPT solver already installed.")
    else:
        print("Non-Colab environment detected — ensure IPOPT is installed locally.")

    return ipopt_path



#Two seperate functions for calculation of monthly returns and optimization

tickers_list = ['AES','LNT','AEE','AEP','AWK','APD','ALB','AMCR','AVY','BALL','ALL', 'AON', 'CPAY', 'EG', 'IVZ']

start = '2022-01-01' # Changed to a more recent start date
end = '2024-01-01' # Changed to a more recent end date
# Function to calculate monthly returns
def calculate_monthly_returns(tickers_list, start_date, end_date):
    """
    Downloads daily stock data, calculates daily and monthly returns.

    Args:
        tickers_list (list): A list of stock ticker symbols.
        start_date (str): The start date for data download (YYYY-MM-DD).
        end_date (str): The end date for data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame of monthly returns, or None if no data.
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
    else:
        return_data_dict = {}
        for ticker, data in dow_prices.items():
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 1:
                    return_data_dict[ticker] = returns

        if not return_data_dict:
            print("No valid stock data available after calculating daily returns.")
            return None
        else:
            daily_returns = pd.concat(return_data_dict.values(), axis=1, keys=return_data_dict.keys())
            # Resample to monthly returns - taking the sum of daily returns within each month
            monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
            return monthly_returns.dropna() # Drop any months with no data for any ticker


def optimize_and_plot_portfolio(df_returns, ipopt_executable):
    from pyomo.environ import ConcreteModel, Set, Var, NonNegativeReals, Param, Objective, maximize, Constraint
    from pyomo.opt import SolverFactory, TerminationCondition
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize model
    m = ConcreteModel()

    # Asset list
    assets = df_returns.columns.tolist()
    m.Assets = Set(initialize=assets)

    # Define decision variables for each asset
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0,1))

    # Calculate average returns per asset and create a Pyomo Param
    avg_returns = df_returns.mean().to_dict()
    m.mu = Param(m.Assets, initialize=avg_returns)

    # Covariance matrix (Sigma) and create a Pyomo Param
    cov_df = df_returns.cov()
    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=cov_dict)

    # Objective: Maximize expected return
    def total_return_rule(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)
    m.objective = Objective(rule=total_return_rule, sense=maximize)

    # Constraint: Sum of allocations must be 1
    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1
    m.budget = Constraint(rule=budget_constraint_rule)

    # Remove the dummy total_risk constraint if it exists (it will be replaced dynamically later)
    if hasattr(m, 'total_risk'):
        m.del_component(m.total_risk)

    print("Pyomo model initialized with sets, variables, parameters, objective, and budget constraint.")

    # Define solver
    solver = SolverFactory("ipopt", executable=ipopt_executable)

    # Determine maximum risk value for plotting purposes if needed
    # For this dataset, a reasonable maximum risk value needs to be determined.
    # One approach is to calculate the variance of an equally weighted portfolio or max individual asset variance.
    # Or, as suggested in the instructions, the max value in the weighted cov matrix where p=1. Let's use max(cov_df.values.flatten())
    # Or, as was done before, simply use the original max_risk value as a starting point, but adjust if necessary.
    # Let's try to infer a reasonable max_risk from the data.
    # A simple heuristic could be the max variance of any single asset, or the max element in the cov matrix.
    # Let's use the max individual asset variance as a conservative upper bound for risk_limits, multiplied by a factor
    # to explore beyond minimal risk.
    max_possible_variance = np.max(np.diag(cov_df.values))
    # Adjusted max_risk to be slightly above the max individual variance or a broader range.
    # This part can be tuned based on the specific dataset characteristics to ensure feasibility.
    # Let's try to infer a reasonable max_risk from the data.
    # A simple heuristic could be the max variance of any single asset, or the max element in the cov matrix.
    # Let's use the max individual asset variance as a conservative upper bound for risk_limits, multiplied by a factor
    # to explore beyond minimal risk.
    max_risk_for_range = np.max(np.diag(cov_df.values)) * 1.5 # Scale up a bit to see more of the frontier
    # Ensure min_risk_for_range is not 0 to avoid division by zero or empty range for arange
    min_risk_for_range = 1e-6 # Start from a very small positive risk

    # Create risk limits array - if max_risk_for_range is too small, np.arange might return empty array
    if max_risk_for_range > min_risk_for_range:
        risk_limits = np.arange(min_risk_for_range, max_risk_for_range + 1e-6, (max_risk_for_range - min_risk_for_range) / 200)
    else:
        # Fallback for very low variance data or specific scenarios
        risk_limits = np.array([min_risk_for_range])


    # Result storage
    param_analysis = {}
    returns = {}

    print(f"Starting portfolio optimization for {len(risk_limits)} risk levels...")
    for i, r in enumerate(risk_limits):
        # Remove old variance constraint if it exists
        if hasattr(m, 'variance_constraint'):
            m.del_component(m.variance_constraint)

        # Add new variance constraint for this risk level
        def variance_constraint_rule(m):
            return sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets) <= r
        m.variance_constraint = Constraint(rule=variance_constraint_rule)

        # Solve
        result = solver.solve(m)

        # Skip infeasible solutions
        if result.solver.termination_condition == TerminationCondition.infeasible or \
           result.solver.termination_condition == TerminationCondition.other:
            print(f"Warning: Model infeasible for risk level {r:.6f}. Skipping.")
            continue

        # Check if the solution is optimal or locally optimal
        if result.solver.termination_condition == TerminationCondition.optimal or \
           result.solver.termination_condition == TerminationCondition.locallyOptimal:
            # Save allocations and returns
            param_analysis[r] = [m.x[a]() for a in m.Assets]
            returns[r] = m.objective()
        else:
            print(f"Warning: Solver terminated with condition {result.solver.termination_condition} for risk level {r:.6f}. Skipping.")


    # Create DataFrame for plotting
    df_results = pd.DataFrame({
        'Risk': list(returns.keys()),
        'Return': list(returns.values())
    })

    # Sort by Risk (just in case)
    df_results = df_results.sort_values(by='Risk')

    # Plot Efficient Frontier
    plt.figure(figsize=(10,6))
    plt.plot(df_results['Risk'], df_results['Return'], marker='o', linestyle='-')
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.show()

    # Convert allocation results to DataFrame for plotting
    df_allocations = pd.DataFrame(param_analysis).T  # rows = risk, columns = assets
    df_allocations.columns = assets
    df_allocations['Risk'] = df_allocations.index

    # Plot asset allocation proportions by asset
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(df_allocations['Risk'], df_allocations[asset], label=str(asset), marker='o', markersize=4)

    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.xlim(0, 0.006)
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Portfolio optimization and plotting complete.")
    return df_results, df_allocations

print("Finished defining `optimize_and_plot_portfolio` function.")

ipopt_executable = setup_ipopt_for_colab()

# Call the prepare_returns_data function
# Make sure 'mydata (1).csv' is available in the environment
df_returns = calculate_monthly_returns(tickers_list, start, end)

# Call the optimize_and_plot_portfolio function
df_results, df_allocations = optimize_and_plot_portfolio(df_returns, ipopt_executable)

print("Functions called successfully and plots generated.")



#One Concise function
def perform_full_portfolio_analysis(tickers_list, start_date, end_date, ipopt_executable):
    """
    Orchestrates the full portfolio optimization process: downloads stock data,
    calculates monthly returns, performs portfolio optimization, and plots results.

    Args:
        tickers_list (list): A list of stock ticker symbols.
        start_date (str): The start date for data download (YYYY-MM-DD).
        end_date (str): The end date for data download (YYYY-MM-DD).
        ipopt_executable (str): Path to the IPOPT solver executable.

    Returns:
        tuple: (df_results, df_allocations) containing the efficient frontier
               and asset allocation dataframes, or (None, None) if an error occurs.
    """
    print("Starting full portfolio analysis...")

    # --- Part 1: Calculate monthly returns (logic from calculate_monthly_returns) ---
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
        return None, None

    return_data_dict = {}
    for ticker, data in dow_prices.items():
        if not data.empty:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 1:
                return_data_dict[ticker] = returns

    if not return_data_dict:
        print("No valid stock data available after calculating daily returns.")
        return None, None

    df_returns = pd.concat(return_data_dict.values(), axis=1, keys=return_data_dict.keys())
    # Resample to monthly returns - taking the product of (1+daily returns) to get monthly factor, then subtract 1
    monthly_returns_raw = (1 + df_returns).resample('ME').prod() - 1
    df_returns = monthly_returns_raw.dropna(axis=1, how='all') # Drop columns where all monthly returns are NaN

    if df_returns.empty:
        print("No valid monthly return data after processing. Aborting optimization.")
        return None, None

    print("Monthly returns calculated successfully.")

    # --- Part 2: Optimize portfolio and plot results (logic from optimize_and_plot_portfolio) ---

    # Initialize model
    m = ConcreteModel()

    # Asset list
    assets = df_returns.columns.tolist()
    m.Assets = Set(initialize=assets)

    # Define decision variables for each asset
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0,1))

    # Calculate average returns per asset and create a Pyomo Param
    avg_returns = df_returns.mean().to_dict()
    m.mu = Param(m.Assets, initialize=avg_returns)

    # Covariance matrix (Sigma) and create a Pyomo Param
    cov_df = df_returns.cov()
    # Ensure cov_df is not empty before proceeding
    if cov_df.empty:
        print("Covariance matrix is empty. Aborting optimization.")
        return None, None

    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=cov_dict)

    # Objective: Maximize expected return
    def total_return_rule(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)
    m.objective = Objective(rule=total_return_rule, sense=maximize)

    # Constraint: Sum of allocations must be 1
    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1
    m.budget = Constraint(rule=budget_constraint_rule)

    print("Pyomo model initialized with sets, variables, parameters, objective, and budget constraint.")

    # Define solver
    solver = SolverFactory("ipopt", executable=ipopt_executable)

    max_possible_variance = np.max(np.diag(cov_df.values))
    max_risk_for_range = max_possible_variance * 1.5
    min_risk_for_range = 1e-6

    if max_risk_for_range > min_risk_for_range:
        risk_limits = np.arange(min_risk_for_range, max_risk_for_range + 1e-6, (max_risk_for_range - min_risk_for_range) / 200)
    else:
        risk_limits = np.array([min_risk_for_range])

    # Result storage
    param_analysis = {}
    returns = {}

    print(f"Starting portfolio optimization for {len(risk_limits)} risk levels...")
    for i, r in enumerate(risk_limits):
        if hasattr(m, 'variance_constraint'):
            m.del_component(m.variance_constraint)

        def variance_constraint_rule(m):
            return sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets) <= r
        m.variance_constraint = Constraint(rule=variance_constraint_rule)

        result = solver.solve(m)

        if result.solver.termination_condition == TerminationCondition.infeasible or \
           result.solver.termination_condition == TerminationCondition.other:
            # print(f"Warning: Model infeasible for risk level {r:.6f}. Skipping.")
            continue

        if result.solver.termination_condition == TerminationCondition.optimal or \
           result.solver.termination_condition == TerminationCondition.locallyOptimal:
            param_analysis[r] = [m.x[a]() for a in m.Assets]
            returns[r] = m.objective()
        else:
            # print(f"Warning: Solver terminated with condition {result.solver.termination_condition} for risk level {r:.6f}. Skipping.")
            pass # Suppress repeated warnings for better readability

    if not returns:
        print("No feasible solutions found for any risk level. Cannot plot results.")
        return None, None

    # Create DataFrame for plotting
    df_results = pd.DataFrame({
        'Risk': list(returns.keys()),
        'Return': list(returns.values())
    })
    df_results = df_results.sort_values(by='Risk')

    # Plot Efficient Frontier
    plt.figure(figsize=(10,6))
    plt.plot(df_results['Risk'], df_results['Return'], marker='o', linestyle='-')
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.show()

    # Convert allocation results to DataFrame for plotting
    df_allocations = pd.DataFrame(param_analysis).T
    df_allocations.columns = assets
    df_allocations['Risk'] = df_allocations.index

    # Plot asset allocation proportions by asset
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(df_allocations['Risk'], df_allocations[asset], label=str(asset), marker='o', markersize=4)

    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.xlim(left=0) # Ensure x-axis starts from 0 or positive values
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xlim(0, 0.035) # Set x-axis limit
    plt.tight_layout()
    plt.show()

    print("Portfolio optimization and plotting complete.")
    return df_results, df_allocations

print("Defined `perform_full_portfolio_analysis` function.")



# Define parameters for the full analysis
#my_tickers = ['AES','LNT','AEE','AEP','AWK','APD','ALB','AMCR','AVY','BALL','ALL', 'AON', 'CPAY', 'EG', 'IVZ']
my_tickers = ['GE','KO','NVDA']
my_start_date = '2020-01-01'
my_end_date = '2024-01-01'
my_ipopt_executable = setup_ipopt_for_colab()

# Import display from IPython.display for rich output
from IPython.display import display

# Call the consolidated function
final_df_results, final_df_allocations = perform_full_portfolio_analysis(my_tickers, my_start_date, my_end_date, my_ipopt_executable)

if final_df_results is not None and final_df_allocations is not None:
    print("Final results and allocations obtained:")
    display(final_df_results.head())
    display(final_df_allocations.head())
else:
    print("Portfolio analysis did not produce results.")
