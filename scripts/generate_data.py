"""Pre-generate trajectory data for state estimation training.

Generates training and validation trajectories on SO(3) x SO(3) and saves
them to a .pt file for fast, reproducible loading during training.

Uses multiprocessing for ~10-20x speedup on multi-core machines.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n_train 100000 --n_val 5000
    python scripts/generate_data.py --workers 32
    python scripts/generate_data.py --help
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from articulated.shared.robot_arm import RobotArmKinematics


def _worker_init(
    pc_quats1, pc_quats2, place_cell_kappa, seq_length, dt, vel_sigma, vel_theta
):
    """Initialize worker process with shared place cell centers."""
    global _w_centers_R1, _w_centers_R2, _w_kappa, _w_seq_length, _w_dt
    global _w_vel_sigma, _w_vel_theta, _w_kinematics
    _w_centers_R1 = Rotation.from_quat(pc_quats1)
    _w_centers_R2 = Rotation.from_quat(pc_quats2)
    _w_kappa = place_cell_kappa
    _w_seq_length = seq_length
    _w_dt = dt
    _w_vel_sigma = vel_sigma
    _w_vel_theta = vel_theta
    _w_kinematics = RobotArmKinematics()


def _generate_one(seed):
    """Generate a single trajectory in a worker process."""
    rng = np.random.default_rng(seed)
    config = _w_kinematics.sample_random_configuration(rng)

    # Initial place cell activation
    init_pc = _compute_pc(config)

    n_total = len(_w_centers_R1) + len(_w_centers_R2)
    decay = np.exp(-_w_vel_theta * _w_dt)
    noise_scale = _w_vel_sigma * np.sqrt(1.0 - decay**2)

    velocities = np.zeros((_w_seq_length, 6))
    targets = np.zeros((_w_seq_length, n_total))
    omega = np.zeros(6)

    for t in range(_w_seq_length):
        omega = decay * omega + noise_scale * rng.standard_normal(6)
        velocities[t] = omega
        config = _w_kinematics.integrate_velocity(config, omega, _w_dt)
        targets[t] = _compute_pc(config)

    return velocities, targets, init_pc


def _compute_pc(config):
    """Compute place cell activations using vMF kernel, separate per joint."""
    R1_cur, R2_cur = config
    kappa = _w_kappa

    d1 = (R1_cur.inv() * _w_centers_R1).magnitude()
    d2 = (R2_cur.inv() * _w_centers_R2).magnitude()

    # vMF kernel + softmax per joint
    logits1 = kappa * np.cos(d1)
    logits1 -= logits1.max()
    exp1 = np.exp(logits1)
    pc1 = exp1 / exp1.sum()

    logits2 = kappa * np.cos(d2)
    logits2 -= logits2.max()
    exp2 = np.exp(logits2)
    pc2 = exp2 / exp2.sum()

    return np.concatenate([pc1, pc2])


def generate_parallel(
    n_trajectories,
    base_seed,
    pc_quats1,
    pc_quats2,
    place_cell_kappa,
    seq_length,
    n_place_cells,
    dt,
    vel_sigma,
    vel_theta,
    n_workers,
):
    """Generate trajectories using multiprocessing."""
    # Each trajectory gets a unique seed derived from base_seed
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31, size=n_trajectories)

    velocities = np.zeros((n_trajectories, seq_length, 6))
    targets = np.zeros((n_trajectories, seq_length, n_place_cells))
    init_pcs = np.zeros((n_trajectories, n_place_cells))

    initargs = (
        pc_quats1,
        pc_quats2,
        place_cell_kappa,
        seq_length,
        dt,
        vel_sigma,
        vel_theta,
    )

    with Pool(n_workers, initializer=_worker_init, initargs=initargs) as pool:
        results = pool.imap(_generate_one, seeds, chunksize=64)
        for i, (vel, tgt, ipc) in enumerate(
            tqdm(results, total=n_trajectories, desc="Generating trajectories")
        ):
            velocities[i] = vel
            targets[i] = tgt
            init_pcs[i] = ipc

    return velocities, targets, init_pcs


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-generate trajectory data")

    parser.add_argument(
        "--n_train", type=int, default=100000, help="Training trajectories"
    )
    parser.add_argument(
        "--n_val", type=int, default=5000, help="Validation trajectories"
    )
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length")
    parser.add_argument(
        "--n_place_cells",
        type=int,
        default=64,
        help="Total place cells (split between 2 joints)",
    )
    parser.add_argument(
        "--place_cell_kappa",
        type=float,
        default=5.0,
        help="vMF concentration parameter",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Integration timestep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers", type=int, default=32, help="Number of parallel workers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/estimation/trajectories.pt",
        help="Output .pt file path",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_per_joint = args.n_place_cells // 2
    print(f"Generating {args.n_train} train + {args.n_val} val trajectories")
    print(
        f"  seq_length={args.seq_length}, n_place_cells={args.n_place_cells} ({n_per_joint} per joint)"
    )
    print(f"  kappa={args.place_cell_kappa}, seed={args.seed}, workers={args.workers}")
    print()

    # Initialize separate place cells for each joint
    rng = np.random.default_rng(args.seed)
    centers_R1 = Rotation.random(n_per_joint, random_state=int(rng.integers(0, 2**31)))
    centers_R2 = Rotation.random(n_per_joint, random_state=int(rng.integers(0, 2**31)))
    pc_quats1 = centers_R1.as_quat()
    pc_quats2 = centers_R2.as_quat()

    common = dict(
        pc_quats1=pc_quats1,
        pc_quats2=pc_quats2,
        place_cell_kappa=args.place_cell_kappa,
        seq_length=args.seq_length,
        n_place_cells=args.n_place_cells,
        dt=args.dt,
        vel_sigma=1.0,
        vel_theta=2.0,
        n_workers=args.workers,
    )

    # Generate training data
    print("Generating training data...")
    t0 = time.time()
    train_vel, train_tgt, train_init = generate_parallel(
        args.n_train,
        base_seed=rng.integers(0, 2**31),
        **common,
    )
    t_train = time.time() - t0
    print(f"  Done in {t_train:.1f}s ({t_train / args.n_train * 1000:.1f}ms/traj)")

    # Generate validation data
    print("Generating validation data...")
    t0 = time.time()
    val_vel, val_tgt, val_init = generate_parallel(
        args.n_val,
        base_seed=rng.integers(0, 2**31),
        **common,
    )
    t_val = time.time() - t0
    print(f"  Done in {t_val:.1f}s")

    data = {
        "train_velocities": torch.from_numpy(train_vel).float(),
        "train_targets": torch.from_numpy(train_tgt).float(),
        "train_init_pcs": torch.from_numpy(train_init).float(),
        "val_velocities": torch.from_numpy(val_vel).float(),
        "val_targets": torch.from_numpy(val_tgt).float(),
        "val_init_pcs": torch.from_numpy(val_init).float(),
        "place_cell_quats1": torch.from_numpy(pc_quats1).float(),
        "place_cell_quats2": torch.from_numpy(pc_quats2).float(),
        "metadata": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "seq_length": args.seq_length,
            "n_place_cells": args.n_place_cells,
            "n_per_joint": n_per_joint,
            "place_cell_kappa": args.place_cell_kappa,
            "dt": args.dt,
            "seed": args.seed,
        },
    }

    print(f"\nSaving to {output_path}...")
    torch.save(data, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved! File size: {size_mb:.1f} MB")
    print(f"Total time: {t_train + t_val:.1f}s")


if __name__ == "__main__":
    main()
