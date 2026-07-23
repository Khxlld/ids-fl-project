"""End-to-end smoke test of the whole pipeline (short epochs, 1 fed round).

Run from the edgeiiot_federated/ directory:  py -3.11 smoke_test.py
"""

import time
import numpy as np
import torch

from src.config import CONFIG, DATASET_PATH, TARGET_COLUMN, ensure_dirs
from src import data as datamod
from src import preprocessing as prep
from src.partitioning import iid_partition, partition_distribution, assert_valid_partitions
from src.model import build_model, set_seed
from src.training import train_centralized, evaluate, get_parameters, set_parameters
from src.evaluation import compute_metrics, format_metrics
from src.federated import run_federated

ensure_dirs()
set_seed(CONFIG.random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| dataset:", DATASET_PATH.name)

# Shorten for the smoke test.
CONFIG.centralized_epochs = 3
CONFIG.federated_rounds = 1
CONFIG.early_stopping_patience = 3

t0 = time.time()
df = datamod.load_dataset(DATASET_PATH)
print(f"[load] {df.shape} in {time.time()-t0:.1f}s")

info = datamod.inspect_dataset(df)
print(f"[inspect] rows={info['n_rows']} cols={info['n_columns']} "
      f"dupes={info['n_duplicate_rows']} missing={info['total_missing']} "
      f"inf={info['total_infinities']} mem={info['memory_mb']}MB")
print("[inspect] label counts:", info["label_counts"])

clean, dropped = datamod.clean_dataset(df, CONFIG)
print(f"[clean] shape={clean.shape} dupes_removed={clean.attrs['n_duplicate_rows_removed']}")
print(f"[clean] dropped {len(dropped)} columns")
del df

print("[sample] before:", clean[TARGET_COLUMN].value_counts().to_dict())
sample = datamod.balanced_sample(clean, CONFIG)
print("[sample] after :", sample[TARGET_COLUMN].value_counts().to_dict(), "size=", len(sample))

train_pool, val, test = datamod.split_data(sample, CONFIG)
print(f"[split] train={len(train_pool)} val={len(val)} test={len(test)}")

Xtr_df, ytr = datamod.split_features_target(train_pool)
Xva_df, yva = datamod.split_features_target(val)
Xte_df, yte = datamod.split_features_target(test)

num_cols, cat_cols, extra_dropped = prep.identify_feature_types(Xtr_df, CONFIG)
print(f"[prep] numeric={len(num_cols)} categorical={len(cat_cols)} extra_dropped={len(extra_dropped)}")
keep = num_cols + cat_cols
pre = prep.build_preprocessor(num_cols, cat_cols).fit(Xtr_df[keep])
Xtr = prep.transform_to_float32(pre, Xtr_df[keep])
Xva = prep.transform_to_float32(pre, Xva_df[keep])
Xte = prep.transform_to_float32(pre, Xte_df[keep])
input_dim = Xtr.shape[1]
print(f"[prep] input_dim={input_dim} Xtr={Xtr.shape}")
assert not np.isnan(Xtr).any() and not np.isinf(Xtr).any(), "NaN/Inf in processed train"

# Centralized baseline (short).
model = build_model(input_dim, CONFIG)
model, hist, best_val, ttime = train_centralized(model, Xtr, ytr, Xva, yva, CONFIG, device, verbose=True)
cen = evaluate(model, Xte, yte, device, CONFIG.decision_threshold)
print("[centralized] test:", format_metrics(cen))

# IID partitioning.
parts = iid_partition(ytr, CONFIG.num_clients, CONFIG.client_names, CONFIG.random_seed)
assert_valid_partitions(parts, ytr, CONFIG.num_clients)
print("[partition]\n", partition_distribution(parts, ytr).to_string(index=False))

# Federated 1-round smoke.
print("[federated] running 1-round simulation on CPU ...")
res = run_federated(Xtr, ytr, parts, Xva, yva, input_dim, CONFIG, client_device="cpu")
print("[federated] round metrics:\n", res["round_metrics"].to_string(index=False))
print("[federated] client metrics rows:", len(res["client_metrics"]), "wall=", round(res["wall_time"],1), "s")
print("[federated] best:", res["best"].get("round"), "f1=", round(res["best"].get("f1", -1), 4))

# Evaluate saved global model on test.
from src.config import GLOBAL_MODEL_PATH
gmodel = build_model(input_dim, CONFIG)
gmodel.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=device, weights_only=True))
fed = evaluate(gmodel, Xte, yte, device, CONFIG.decision_threshold)
print("[federated] test:", format_metrics(fed))
print("SMOKE TEST OK")
