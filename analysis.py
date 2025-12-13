import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from itertools import product
from simulation import run_simulation_with_crn


def batch_means(series, num_batches=10):
    batch_size = len(series) // num_batches
    return [np.mean(series[i * batch_size:(i + 1) * batch_size]) for i in range(num_batches) if batch_size > 0]


def perform_serial_correlation_analysis(repeats=10):
    df_serial = run_simulation_with_crn(inter_mean=22.5, pre_units=4, rec_units=4, check=100, repeats=repeats)
    acfs = [acf(row['q_series'], nlags=20, missing='drop') for _, row in
            df_serial.iterrows()]  # Handle missing/short data
    avg_acf = np.mean(acfs, axis=0)
    print("Average ACF:", avg_acf)

    batch_series = [batch_means(row['q_series']) for _, row in df_serial.iterrows()]
    batch_acfs = [acf(bs, nlags=5, missing='drop') for bs in batch_series if len(bs) > 1]
    avg_batch_acf = np.mean(batch_acfs, axis=0)
    print("Batch Means ACF:", avg_batch_acf)  # Should be low at lag 1

    # Save to file
    with open('serial_correlation_output.txt', 'w') as f:
        f.write("Average ACF:\n")
        f.write(str(avg_acf) + "\n\n")
        f.write("Batch Means ACF:\n")
        f.write(str(avg_batch_acf) + "\n")


def perform_full_factorial_and_metamodel(repeats=10):
    factors = {
        'inter_dist': ['exp', 'unif'],
        'inter_mean': [25, 22.5],
        'pre_dist': ['exp', 'unif'],
        'rec_dist': ['exp', 'unif'],
        'pre_units': [4, 5],
        'rec_units': [4, 5]
    }
    configs = list(product(*factors.values()))
    raw_res = np.zeros((len(configs), 3, repeats))  # 64 configs x 3 outputs (util, block, q) x repeats

    for c_idx, config in enumerate(configs):
        inter_dist, inter_mean, pre_dist, rec_dist, pre_u, rec_u = config
        inter_half = 5 if inter_dist == 'unif' and inter_mean == 25 else (2.5 if inter_dist == 'unif' else None)
        pre_half = 10 if pre_dist == 'unif' else None
        rec_half = 10 if rec_dist == 'unif' else None
        df = run_simulation_with_crn(inter_dist, inter_mean, inter_half,
                                     pre_dist, 40, pre_half,
                                     rec_dist, 40, rec_half,
                                     pre_u, rec_u, check=100, repeats=repeats)
        raw_res[c_idx, 0, :] = df['utilization']
        raw_res[c_idx, 1, :] = df['blocking_rate']
        raw_res[c_idx, 2, :] = df['avg_entry_queue']

    # Save raw results to CSV
    raw_df = pd.DataFrame({
        'config': [str(config) for config in configs],
        'avg_utilization': np.mean(raw_res[:, 0, :], axis=1),
        'avg_blocking_rate': np.mean(raw_res[:, 1, :], axis=1),
        'avg_entry_queue': np.mean(raw_res[:, 2, :], axis=1)
    })
    raw_df.to_csv('simulation_results.csv', index=False)

    # For each output, compute metamodel (example for blocking_rate)
    output_idx = 1  # blocking_rate
    res = raw_res[:, output_idx, :]  # 64 x repeats
    y = np.mean(res, axis=1)  # 64 averages
    C = np.cov(res.T, rowvar=False) / repeats  # (64,64) cov matrix

    # Add small ridge for stability
    ridge = 1e-6
    C += ridge * np.eye(C.shape[0])

    # Build design matrix X: intercept + main + 2-way interactions
    factor_levels = {k: [-1, 1] for k in factors}
    X = np.ones((len(configs), 1 + len(factors) + len(factors) * (len(factors) - 1) // 2))  # 1 + 6 + 15 = 22 cols
    col_idx = 1

    # Main effects
    for f_idx, f in enumerate(factors):
        for c_idx, config in enumerate(configs):
            val = config[f_idx]
            factor_values = factors[f]
            level_map = {factor_values[0]: -1, factor_values[1]: 1}
            X[c_idx, col_idx] = level_map[val]
        col_idx += 1

    # 2-way interactions
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            for c_idx in range(len(configs)):
                X[c_idx, col_idx] = X[c_idx, i + 1] * X[c_idx, j + 1]  # Main cols start at 1
            col_idx += 1

    # WLS fit
    inv_C = np.linalg.pinv(C)
    X_T_inv_C = X.T @ inv_C
    mmm = X_T_inv_C @ X
    zz = X_T_inv_C @ y
    b = np.linalg.solve(mmm, zz)
    print("Regression Coefficients b:", b)

    # Cov_b and significance
    cov_b = np.linalg.pinv(mmm)
    std_b = np.sqrt(np.maximum(np.diag(cov_b), 0))
    significance = np.abs(b) / std_b
    print("Std Deviations:", std_b)
    print("Significance Ratios (|b|/std):", significance)

    # Predictions and errors
    pred = X @ b
    err = pred - y
    print("Prediction Errors:", err)

    # Save metamodel output to file
    with open('metamodel_output.txt', 'w') as f:
        f.write("Regression Coefficients b:\n")
        f.write(str(b) + "\n\n")
        f.write("Std Deviations:\n")
        f.write(str(std_b) + "\n\n")
        f.write("Significance Ratios (|b|/std):\n")
        f.write(str(significance) + "\n\n")
        f.write("Prediction Errors:\n")
        f.write(str(err) + "\n")