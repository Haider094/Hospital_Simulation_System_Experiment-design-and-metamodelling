import random
import simpy
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from itertools import product
import matplotlib.pyplot as plt

repeats = 10  # Define repeats globally

# From models.py
class stream:
    def __init__(self, dist, mean, halfwidth=None):
        self.dist = dist
        self.mean = mean
        self.halfwidth = halfwidth
        if dist == 'exp':
            self.rng = lambda: random.expovariate(1 / mean)
        elif dist == 'unif':
            if halfwidth is None:
                raise ValueError("Uniform requires halfwidth")
            self.rng = lambda: random.uniform(mean - halfwidth, mean + halfwidth)

    def new(self):
        return self.rng()

class clinic(object):
    def __init__(self, env, num_pre, num_op, num_post):
        self.env = env
        self.Preparation = simpy.Resource(env, num_pre)
        self.Operation = simpy.Resource(env, num_op)
        self.Recovery = simpy.Resource(env, num_post)
        self.prewait = 0
        self.opwait = 0
        self.postwait = 0
        self.isblocking = False
        self.isoperational = False

    def reset(self):
        self.prewait = 0
        self.opwait = 0
        self.postwait = 0

    def report(self):
        print(self.prewait, self.opwait, self.postwait)

class patient(object):
    def __init__(self, env, ward, pre, op, post, is_severe=False):
        self.env = env
        self.ward = ward
        # Twist: Severe patients have longer times
        if is_severe:
            self.pre = pre * 1.25  # e.g., 25% longer
            self.op = op * 1.5     # 50% longer op
            self.post = post * 1.25
        else:
            self.pre = pre
            self.op = op
            self.post = post

    def run(self):
        Prep = self.ward.Preparation
        Op = self.ward.Operation
        Post = self.ward.Recovery

        arrivaltime = self.env.now
        pre_req = Prep.request()
        yield pre_req
        yield self.env.timeout(self.pre)
        op_req = Op.request()
        yield op_req
        self.ward.isoperational = True
        opwaittime = self.env.now
        self.ward.prewait += self.env.now - arrivaltime
        Prep.release(pre_req)

        yield self.env.timeout(self.op)
        self.ward.isoperational = False
        self.ward.isblocking = True
        post_req = Post.request()
        yield post_req
        self.ward.isblocking = False
        postwaittime = self.env.now
        self.ward.opwait += self.env.now - opwaittime
        Op.release(op_req)

        yield self.env.timeout(self.post)
        Post.release(post_req)
        self.ward.postwait += self.env.now - postwaittime

# From monitor.py
class monitor(object):
    def __init__(self, checkfreq, itercount):
        self.check = checkfreq
        self.iter = itercount
        self.blockfrq = 0
        self.operfrq = 0
        self.checkfrq = 0
        self.entryq_sum = 0  # For average queue length
        self.bf_dump = []
        self.op_dump = []
        self.q_dump = []  # List of queue lengths for serial correlation

    def reset(self):
        self.blockfrq = 0
        self.checkfrq = 0
        self.operfrq = 0
        self.entryq_sum = 0
        self.q_dump = []

    def report(self, it):
        self.bf_dump.append(self.blockfrq)
        self.op_dump.append(self.operfrq)

    def run(self, machine, env):
        while True:
            yield env.timeout(self.check)
            self.checkfrq += 1
            entryq_len = len(machine.Preparation.queue)
            self.q_dump.append(entryq_len)  # Collect for ACF
            self.entryq_sum += entryq_len  # For average
            if machine.isblocking:
                self.blockfrq += 1
            if machine.isoperational:
                self.operfrq += 1

    def dump(self):
        print("Blocking frequencies:", self.bf_dump)
        print("Operation frequencies:", self.op_dump)

    def get_avg_entryq(self):
        return self.entryq_sum / self.checkfrq if self.checkfrq > 0 else 0

# From utils.py
def setup(env, ward, t_inter, pretime, optime, posttime, endtime, severe_prob=0.0):
    while env.now < endtime:
        interarrivaltime = t_inter.new()
        pre = pretime.new()
        op = optime.new()
        post = posttime.new()
        # Twist: Randomly assign severe type
        is_severe = random.random() < severe_prob
        new = patient(env, ward, pre, op, post, is_severe)
        env.process(new.run())
        yield env.timeout(interarrivaltime)

# From simulation.py
# Updated main simulation function with CRN
def run_simulation_with_crn(inter_dist='exp', inter_mean=25, inter_half=None,
                            pre_dist='exp', pre_mean=40, pre_half=None,
                            rec_dist='exp', rec_mean=40, rec_half=None,
                            pre_units=4, rec_units=4, severe_prob=0.0,
                            warm_time=1000, sim_time=10000, repeats=10, check=100,
                            seed_base=42):
    results = []  # List of dicts for each replication
    for rep in range(repeats):
        random.seed(seed_base + rep)  # CRN: same seed sequence per rep across configs
        env = simpy.Environment()
        ward = clinic(env, pre_units, 1, rec_units)
        interarrival = stream(inter_dist, inter_mean, inter_half)
        preparationtime = stream(pre_dist, pre_mean, pre_half)
        operationtime = stream('exp', 20)
        recoverytime = stream(rec_dist, rec_mean, rec_half)
        reporter = monitor(check, repeats)
        runtime = warm_time + sim_time
        env.process(setup(env, ward, interarrival, preparationtime, operationtime, recoverytime, runtime, severe_prob))
        env.process(reporter.run(ward, env))

        env.run(until=warm_time)
        reporter.reset()
        env.run(until=env.now + sim_time)

        util = reporter.operfrq / reporter.checkfrq
        block_rate = reporter.blockfrq / reporter.checkfrq
        avg_entryq = reporter.get_avg_entryq()
        results.append({
            'utilization': util,
            'blocking_rate': block_rate,
            'avg_entry_queue': avg_entryq,
            'q_series': reporter.q_dump  # For ACF
        })

    return pd.DataFrame(results)

# From analysis.py
def batch_means(series, num_batches=10):
    batch_size = len(series) // num_batches
    return [np.mean(series[i*batch_size:(i+1)*batch_size]) for i in range(num_batches) if batch_size > 0]

def plot_acf(acf_vals, title='ACF Plot', filename='acf_plot.png'):
    lags = np.arange(len(acf_vals))
    plt.figure()
    plt.stem(lags, acf_vals)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.savefig(filename)
    print(f'Plot saved as {filename}')

def perform_serial_correlation_analysis():
    df_serial = run_simulation_with_crn(inter_mean=22.5, pre_units=4, rec_units=4, check=100, severe_prob=0.5)  # With twist
    acfs = [acf(row['q_series'], nlags=20, missing='drop') for _, row in df_serial.iterrows()]
    avg_acf = np.mean(acfs, axis=0)
    print("Average ACF:", avg_acf)
    plot_acf(avg_acf, 'Average Raw ACF')

    batch_series = [batch_means(row['q_series']) for _, row in df_serial.iterrows()]
    batch_acfs = [acf(bs, nlags=5, missing='drop') for bs in batch_series if len(bs) > 1]
    avg_batch_acf = np.mean(batch_acfs, axis=0)
    print("Batch Means ACF:", avg_batch_acf)
    plot_acf(avg_batch_acf[:6], 'Average Batch Means ACF', 'batch_acf_plot.png')

    # Adjusted CI example (safety margin as in sample notebook)
    corr_factor = 1 + 2 * np.sum(avg_batch_acf[1:])  # Approximate adjustment
    print(f'Correlation adjustment factor for CI: {corr_factor}')

def perform_full_factorial_and_metamodel():
    factors = {
        'inter_dist': ['exp', 'unif'],
        'inter_mean': [25, 22.5],
        'severe_prob': [0.0, 0.5],  # Twist replacement
        'rec_dist': ['exp', 'unif'],
        'pre_units': [4, 5],
        'rec_units': [4, 5]
    }
    configs = list(product(*factors.values()))
    raw_res = np.zeros((len(configs), 3, 10))  # 64 configs x 3 outputs x 10 reps

    for c_idx, config in enumerate(configs):
        inter_dist, inter_mean, severe_prob, rec_dist, pre_u, rec_u = config
        inter_half = 5 if inter_dist == 'unif' and inter_mean == 25 else (2.5 if inter_dist == 'unif' else None)
        pre_half = None  # Twist: No pre_dist variation
        rec_half = 10 if rec_dist == 'unif' else None
        df = run_simulation_with_crn(inter_dist, inter_mean, inter_half,
                                     'exp', 40, pre_half,  # Fixed prep to exp
                                     rec_dist, 40, rec_half,
                                     pre_u, rec_u, severe_prob=severe_prob, check=100)
        raw_res[c_idx, 0, :] = df['utilization']
        raw_res[c_idx, 1, :] = df['blocking_rate']
        raw_res[c_idx, 2, :] = df['avg_entry_queue']

    # Metamodel for queue length (assignment focus)
    output_idx = 2  # avg_entry_queue
    res = raw_res[:, output_idx, :]  # 64 x 10
    y = np.mean(res, axis=1)  # 64 averages
    C = np.cov(res.T, rowvar=False) / 10  # Cov of means

    ridge = 1e-6
    C += ridge * np.eye(C.shape[0])

    # X: intercept + 5 mains (twist replaced one) + 10 two-ways (adjusted)
    num_factors = len(factors)
    X = np.ones((len(configs), 1 + num_factors + num_factors*(num_factors-1)//2))
    col_idx = 1

    for f_idx, f in enumerate(factors):
        for c_idx, config in enumerate(configs):
            val = config[f_idx]
            factor_values = factors[f]
            level_map = {factor_values[0]: -1, factor_values[1]: 1}
            X[c_idx, col_idx] = level_map[val]
        col_idx += 1

    for i in range(num_factors):
        for j in range(i+1, num_factors):
            for c_idx in range(len(configs)):
                X[c_idx, col_idx] = X[c_idx, i+1] * X[c_idx, j+1]
            col_idx += 1

    inv_C = np.linalg.pinv(C)
    X_T_inv_C = X.T @ inv_C
    mmm = X_T_inv_C @ X
    zz = X_T_inv_C @ y
    b = np.linalg.solve(mmm, zz)
    print("Regression Coefficients b:", b)

    cov_b = np.linalg.pinv(mmm)
    std_b = np.sqrt(np.maximum(np.diag(cov_b), 0))
    significance = np.abs(b) / std_b
    print("Std Deviations:", std_b)
    print("Significance Ratios (|b|/std):", significance)

    # Prune insignificant (<2) and refit
    sig_mask = significance > 2
    X_pruned = X[:, sig_mask]
    if X_pruned.shape[1] > 0:
        X_T_inv_C_pruned = X_pruned.T @ inv_C
        mmm_pruned = X_T_inv_C_pruned @ X_pruned
        zz_pruned = X_T_inv_C_pruned @ y
        b_pruned = np.linalg.solve(mmm_pruned, zz_pruned)
        print("Pruned Coefficients:", b_pruned)

    pred = X @ b
    err = pred - y
    print("Prediction Errors:", err)

# From main.py
print("Performing Serial Correlation Analysis (with Twist):")
perform_serial_correlation_analysis()

print("\nPerforming Full Factorial Experiment and Metamodeling (with Twist):")
perform_full_factorial_and_metamodel()
