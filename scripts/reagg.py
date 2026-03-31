import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from psai.config import SimConfig
import run_multiseed
from run_multiseed import aggregate_seeds

all_dfs = []
for i in range(2, 10):
    path = f"results/seed_{i}/tables/epoch_metrics.csv"
    if os.path.exists(path):
        all_dfs.append(pd.read_csv(path))

cfg = SimConfig()
agg_dir = "results/aggregated"
aggregate_seeds(all_dfs, agg_dir, cfg)
