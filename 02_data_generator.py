import json
import os
import time

import numpy as np

from config import *
from core.forward_tem import TEMForwardModeler


def _load_prior_range():
    prior_path = os.path.join(OUTPUT_DIR, "prior_bounds.json")
    if not os.path.exists(prior_path):
        print(f"prior_bounds.json not found, using config range [{R_MIN}, {R_MAX}] ohm*m")
        return R_MIN, R_MAX
    with open(prior_path, "r", encoding="utf-8") as f:
        prior = json.load(f)
    r_min, r_max = float(prior["r_min"]), float(prior["r_max"])
    print(f"Loaded prior range [{r_min:.1f}, {r_max:.1f}] ohm*m")
    return r_min, r_max


def _sample_thicknesses(rng):
    thicknesses = np.zeros((SAMPLE_SIZE, LAYER_NUM - 1))
    for i in range(SAMPLE_SIZE):
        for _ in range(1000):
            vals = 10 ** rng.uniform(np.log10(THICKNESS_MIN), np.log10(THICKNESS_MAX), LAYER_NUM - 1)
            if LAYER_NUM <= 2:
                thicknesses[i] = vals
                break
            differ = np.abs(vals[1:] - vals[:-1]) / (vals[:-1] + 1e-12)
            if np.all(differ >= 0.2):
                thicknesses[i] = vals
                break
        else:
            thicknesses[i] = vals
    return thicknesses


def main():
    print("=== Step 02: physics sample generation ===")
    r_min, r_max = _load_prior_range()
    rng = np.random.default_rng(RANDOM_SEED)

    rhos = 10 ** rng.uniform(np.log10(r_min), np.log10(r_max), (SAMPLE_SIZE, LAYER_NUM))
    thicknesses = _sample_thicknesses(rng)
    time_gates = np.logspace(np.log10(TIME_MIN), np.log10(TIME_MAX), TIME_CHANNELS)

    modeler = TEMForwardModeler(tx_size_key="4", center_z=0.0)
    batch_size = min(50, max(1, SAMPLE_SIZE))
    batches = []
    total_batches = (SAMPLE_SIZE + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, SAMPLE_SIZE)
        batches.append(modeler.forward_batch(rhos[start:end], thicknesses[start:end], time_gates))
        print(f"forward progress: {end}/{SAMPLE_SIZE}")

    x_data = np.concatenate(batches, axis=0)
    y_data = np.hstack((rhos, thicknesses))
    np.save(os.path.join(DATA_DIR, "X_train.npy"), x_data)
    np.save(os.path.join(DATA_DIR, "Y_train.npy"), y_data)
    np.save(os.path.join(DATA_DIR, "time_gates.npy"), time_gates)

    print(f"Generated X_train{x_data.shape}, Y_train{y_data.shape} in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
