import os
import pandas as pd
import numpy as np
import yfinance as yf
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, Param, Set,
    NonNegativeReals, Binary, minimize, SolverFactory
)

# ============================================================
# Create output folder
# ============================================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Fetch S&P 500 tickers + sectors
# ============================================================
def load_sp500():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df  # columns: Symbol, Name, Sector


# ============================================================
# Download price data + compute monthly returns
# ============================================================
def get_monthly_returns(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, interval="1d", progress=False)
    if "Close" not in data:
        raise ValueError("Failed to download price data")

    prices = data["Close"].ffill()
    daily_returns = prices.pct_change().dropna()
    monthly_returns = (1 + daily_returns).resample("M").prod() - 1
    return monthly_returns


# ============================================================
# Build MINLP Portfolio Model (Bonmin)
# ============================================================
def build_minlp_model(returns_df, sector_map, min_weight=0.02, max_weight=0.20,
                      min_assets=5, enforce_sector_constraint=False,
                      risk_level="medium"):

    tickers = list(returns_df.columns)
    mu = returns_df.mean().to_dict()
    Sigma_df = returns_df.cov()

    m = ConcreteModel()

    # Sets
    m.Assets = Set(initialize=tickers)

    # Variables
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0, 1))   # weights
    m.z = Var(m.Assets, within=Binary)                            # binary selection var

    # Parameters
    m.mu = Param(m.Assets, initialize=mu)
    Sigma = {(i, j): Sigma_df.loc[i, j] for i in tickers for j in tickers}
    m.Sigma = Param(m.Assets, m.Assets, initialize=Sigma)

    # Objective: minimize variance or maximize return
    if risk_level == "low":
        # minimize risk only
        m.obj = Objective(
            expr=sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets),
            sense=minimize,
        )
    elif risk_level == "high":
        # maximize return only
        m.obj = Objective(
            expr=sum(m.mu[a] * m.x[a] for a in m.Assets),
            sense=-1  # maximize (Pyomo does minimize by default)
        )
    else:
        # medium: trades off risk–return
        m.obj = Objective(
            expr=sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets)
            - 0.5 * sum(m.mu[a] * m.x[a] for a in m.Assets),
            sense=minimize
        )

    # Budget constraint
    m.budget = Constraint(expr=sum(m.x[a] for a in m.Assets) == 1)

    # Linking constraints (if invested → must be between 2% and 20%)
    def min_weight_rule(m, a):
        return m.x[a] >= min_weight * m.z[a]
    m.min_weight = Constraint(m.Assets, rule=min_weight_rule)

    def max_weight_rule(m, a):
        return m.x[a] <= max_weight * m.z[a]
    m.max_weight = Constraint(m.Assets, rule=max_weight_rule)

    # Choose at least N stocks
    m.min_assets_rule = Constraint(expr=sum(m.z[a] for a in m.Assets) >= min_assets)

    # Sector constraint: choose at least 1 per sector
    if enforce_sector_constraint:
        sectors = set(sector_map.values())

        def sector_rule(m, s):
            tickers_in_sector = [t for t in tickers if sector_map[t] == s]
            return sum(m.z[t] for t in tickers_in_sector) >= 1

        m.sector_con = Constraint(sectors, rule=sector_rule)

    return m


# ============================================================
# Solve with Bonmin
# ============================================================
def solve_model(model):
    solver = SolverFactory("bonmin", executable="/usr/bin/bonmin")

    result = solver.solve(model, tee=True)
    return result


# ============================================================
# Extract solution
# ============================================================
def extract_solution(model):
    weights = {a: model.x[a]() for a in model.Assets}
    return pd.DataFrame({"Ticker": list(weights.keys()), "Weight": list(weights.values())})


# ============================================================
# Main workflow
# ============================================================
def run_portfolio_version2(start="2022-01-01", end="2024-01-01",
                           min_assets=5, enforce_sector_constraint=False,
                           risk_level="medium"):

    print("Loading S&P 500 sector map...")
    sp500 = load_sp500()
    sector_map = dict(zip(sp500["Symbol"], sp500["Sector"]))

    tickers = list(sp500["Symbol"].head(40))  # use first 40 tickers

    print("Fetching prices...")
    returns_df = get_monthly_returns(tickers, start, end)

    print("Building MINLP model (Bonmin)...")
    model = build_minlp_model(
        returns_df,
        sector_map=sector_map,
        min_assets=min_assets,
        enforce_sector_constraint=enforce_sector_constraint,
        risk_level=risk_level
    )

    print("Solving with Bonmin...")
    solve_model(model)

    print("Extracting solution...")
    df = extract_solution(model)

    output_path = os.path.join(OUTPUT_DIR, "portfolio_v2_weights.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved optimized weights → {output_path}")
    return df


# ============================================================
# Run Version 2 when main_v2.py executes
# ============================================================
if __name__ == "__main__":
    result = run_portfolio_version2(
        risk_level="medium",
        min_assets=5,
        enforce_sector_constraint=False
    )

    print("\nOptimized Portfolio Weights:")
    print(result)
