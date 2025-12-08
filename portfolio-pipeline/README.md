# Portfolio Optimization using Pyomo, IPOPT and BONMIN
## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

This repository implements a portfolio optimization pipeline using Pyomo, Bonmin/Ipopt, and custom Python scripts.
The project is executed through a Jupyter notebook (BDM_FinalProject.ipynb) that performs reproducible steps, installs dependencies, runs optimization scripts, and stores outputs.

Project Overview
This project evaluates financial portfolios by:
Downloading historical stock price data
Computing monthly returns
Running optimization using Pyomo (Ipopt / Bonmin)
Generating efficient frontier plots
Saving optimized results to an output folder
Experimenting with nonlinear MINLP solvers (Bonmin)
Running multiple versions of main scripts (main.py, main_v2.py)

Notebook Workflow (Step-by-Step)
Below is an exact reconstruction of your notebook steps.

1. Clone the GitHub Repository
!git clone https://github.com/anushkarajoriya14/BDM_Allocation_Model

2. Move into the portfolio-pipeline Directory
%cd BDM_Allocation_Model/portfolio-pipeline

3. Install Dependencies

Installs project packages from the repository:
!pip install -r requirements.txt

4. Install Ipopt Solver via IDAES
!idaes get-extensions --to /content/bin

This installs solvers such as Ipopt into /content/bin.

5. Configure Matplotlib for Inline Plotting
%matplotlib inline
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

6. Create Output Folder
The notebook ensures an output/ directory exists for saving results:

import os
os.makedirs("output", exist_ok=True)

7. Run Existing Optimization Pipeline (main.py)
%run main.py


This script:
Downloads stock data
Computes monthly returns
Performs efficient frontier optimization
Plots results

8. Save Results to Output Folder
After running main.py, the notebook stores results:

df_results.to_csv("output/result.csv", index=False)

9. Verify Output Files
!ls output/

Version 2 Tasks — Using Bonmin Solver

10. Update solver initialization to:
SolverFactory("bonmin", executable="/usr/bin/bonmin")

11. Create Symlink to Bonmin Executable

%%bash
ln -sf /content/bin/bonmin /usr/bin/bonmin

This ensures Pyomo can detect the solver.

13. Run main_v2.py with Bonmin
%run main_v2.py


Final Notebook Outputs 

The notebook produces (Files Stored in output/):

Initial Asset Allocation

Final Asset Allocation

Initial Efficient Frontier

Final Efficient Frontier

result.csv

portfolio_v2_weights
 
