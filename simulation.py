import random
import simpy
import numpy as np
import pandas as pd
from models import stream, clinic
from monitor import monitor
from utils import setup

# Updated main simulation function with CRN
def run_simulation_with_crn(inter_dist='exp', inter_mean=25, inter_half=None,
                            pre_dist='exp', pre_mean=40, pre_half=None,
                            rec_dist='exp', rec_mean=40, rec_half=None,
                            pre_units=4, rec_units=4,
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
        env.process(setup(env, ward, interarrival, preparationtime, operationtime, recoverytime, runtime))
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