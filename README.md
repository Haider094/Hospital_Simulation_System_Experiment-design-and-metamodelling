# Hospital Surgery Unit Simulation: Assignment 4

## Overview
This project implements a discrete-event simulation model for a hospital surgery unit using Python and SimPy. It addresses Assignment 4 requirements from the TIES481 Simulation course, focusing on experiment design, serial correlation analysis, and metamodelling for system variants. The model simulates patient flows through preparation, operation, and recovery phases, analyzing outputs like queue lengths, utilization, and blocking rates under different configurations.

Key features:
- Full factorial experiment design (2^6 = 64 configurations) across interarrival times, distributions, and capacities.
- Serial correlation testing with ACF and batch means.
- Metamodelling using Weighted Least Squares (WLS) regression with main effects and two-way interactions.
- Variance reduction via Common Random Numbers (CRN).

The code is modular, with separate files for models, monitoring, utilities, simulation, analysis, and a main entry point.

## Dependencies
- Python 3.8+
- Libraries:
  - `simpy` (for simulation)
  - `numpy` (for numerical operations)
  - `pandas` (for data handling)
  - `statsmodels` (for ACF)
  - `itertools` (for factorial design)

Install via:
```
pip install simpy numpy pandas statsmodels
```

## Project Structure
```
hospital_simulation/
├── main.py            
└── README.md          # This file
```

## How to Run
1. Place all files in a directory.
2. Run the main script:
   ```
   python main.py
   ```
   - Outputs: Serial correlation results (ACF, batch means), metamodel coefficients, std devs, significance, and prediction errors for blocking rate (default). Change `output_idx` in `analysis.py` for other outputs (0: utilization, 2: queue length).

- **Customization**:
  - Adjust parameters in `run_simulation_with_crn` (e.g., `warm_time`, `sim_time`, `repeats`, `check`).
  - For queue length metamodel: Set `output_idx=2` in `perform_full_factorial_and_metamodel`.
  - Add twist: Modify `factors` dict in `analysis.py` and rerun.

## Key Components
- **Simulation Logic**: Patients arrive via customizable streams, request resources, and are monitored for queues/blocking.
- **Serial Correlation**: Tested on high-load config; batching reduces dependence.
- **Experiment Design**: Full factorial for comprehensive analysis.
- **Metamodel**: WLS regression; accounts for CRN covariances with regularization for stability.
- **Twist**: In patient/setup.
- **Plots**: ACF visuals. 
- **Pruning**: In metamodel. 

## Sample Output Interpretation
- ACF: Raw lag-1 0.96 (twist effect); batch 0.69.
- Metamodel: Twist +9.77 (increases queues); sig high; errors -7 to 0.5.

## Limitations
- Computational: 640 runs may take time (~10-20 min on standard hardware).
- No visuals/plots; add matplotlib if needed.

