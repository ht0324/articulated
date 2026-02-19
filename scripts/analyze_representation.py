"""Analyze learned representations from trained state estimation models.

Generates PCA, t-SNE, and tuning curve plots on validation data.

Usage:
    python scripts/analyze_representation.py logs/estimation/rnn/checkpoints/last-v3.ckpt
    python scripts/analyze_representation.py path/to/checkpoint.ckpt --n_traj 500
    python scripts/analyze_representation.py path/to/checkpoint.ckpt --help
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from articulated.estimation.model import StateEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze learned representations")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/estimation/trajectories.pt",
        help="Path to trajectories.pt",
    )
    parser.add_argument(
        "--n_traj", type=int, default=200, help="Number of val trajectories to use"
    )
    parser.add_argument(
        "--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path (default: next to ckpt)"
    )
    return parser.parse_args()


def get_joint_angles(data, n_traj, seq_len):
    """Reconstruct true continuous joint rotations by replaying velocity integration.

    This gives the actual rotation at each timestep — not the quantized
    nearest-cell-center approximation.
    """
    from articulated.shared.robot_arm import RobotArmKinematics

    kin = RobotArmKinematics()
    vel = data["val_velocities"][:n_traj].numpy()  # (n_traj, seq_len, 6)
    dt = 0.01

    # Place cell centers for reconstructing initial config
    centers_R1 = Rotation.from_quat(data["place_cell_quats1"].numpy())
    centers_R2 = Rotation.from_quat(data["place_cell_quats2"].numpy())
    n_per_joint = len(centers_R1)

    # Get initial place cell activations to find initial rotation
    init_pcs = data["val_init_pcs"][:n_traj].numpy()  # (n_traj, n_place_cells)

    # Cell IDs for coloring (from targets)
    targets = data["val_targets"][:n_traj].numpy()
    j1_tgt = targets[:, :, :n_per_joint].reshape(-1, n_per_joint)
    j2_tgt = targets[:, :, n_per_joint:].reshape(-1, n_per_joint)
    j1_cell = j1_tgt.argmax(axis=1)
    j2_cell = j2_tgt.argmax(axis=1)

    j1_rotvec = np.zeros((n_traj, seq_len, 3))
    j2_rotvec = np.zeros((n_traj, seq_len, 3))

    for i in range(n_traj):
        # Reconstruct initial config from init_pc by picking nearest cell center
        init_j1 = init_pcs[i, :n_per_joint]
        init_j2 = init_pcs[i, n_per_joint:]
        R1_init = centers_R1[int(init_j1.argmax())]
        R2_init = centers_R2[int(init_j2.argmax())]
        config = (R1_init, R2_init)

        for t in range(seq_len):
            omega = vel[i, t]
            config = kin.integrate_velocity(config, omega, dt)
            j1_rotvec[i, t] = config[0].as_rotvec()
            j2_rotvec[i, t] = config[1].as_rotvec()

    j1_rotvec = j1_rotvec.reshape(-1, 3)
    j2_rotvec = j2_rotvec.reshape(-1, 3)

    return j1_rotvec, j2_rotvec, j1_cell, j2_cell


def find_selective_neurons(hidden_flat, angles, top_k=3):
    """Find neurons most correlated with a joint angle variable."""
    correlations = np.array(
        [
            np.abs(np.corrcoef(hidden_flat[:, i], angles)[0, 1])
            for i in range(hidden_flat.shape[1])
        ]
    )
    # Replace NaN with 0 (constant neurons)
    correlations = np.nan_to_num(correlations)
    top_idx = np.argsort(correlations)[::-1][:top_k]
    return top_idx, correlations[top_idx]


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = StateEstimationModel.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.eval()

    print(f"Loading data: {args.data_path}")
    data = torch.load(args.data_path, weights_only=True)

    n_traj = min(args.n_traj, data["val_velocities"].shape[0])
    val_vel = data["val_velocities"][:n_traj]
    val_init = data["val_init_pcs"][:n_traj]
    seq_len = val_vel.shape[1]
    hidden_size = model.hidden_size

    # Forward pass
    print(f"Running inference on {n_traj} trajectories...")
    with torch.no_grad():
        h0 = model._encode_init_pos(val_init)
        _, hidden_states = model(val_vel, hidden=h0)

    hidden_flat = hidden_states.numpy().reshape(-1, hidden_size)
    n_points = hidden_flat.shape[0]
    print(f"Hidden states: {n_points} points x {hidden_size} dims")

    # Joint rotation vectors (3D per joint)
    j1_rotvec, j2_rotvec, j1_cell, j2_cell = get_joint_angles(data, n_traj, seq_len)
    axis_labels = ["x", "y", "z"]

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=min(50, hidden_size))
    hidden_pca = pca.fit_transform(hidden_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)

    # t-SNE (subsample for speed)
    n_tsne = min(5000, n_points)
    tsne_idx = np.random.default_rng(42).choice(n_points, n_tsne, replace=False)
    print(f"Running t-SNE on {n_tsne} points (perplexity={args.tsne_perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=42)
    hidden_tsne = tsne.fit_transform(hidden_flat[tsne_idx])

    # Find most selective neurons (check all 3 components per joint, pick best)
    best_j1_neurons, best_j1_corr, best_j1_comp = [], [], []
    for comp in range(3):
        idx, corr = find_selective_neurons(hidden_flat, j1_rotvec[:, comp], top_k=1)
        best_j1_neurons.append(idx[0])
        best_j1_corr.append(corr[0])
        best_j1_comp.append(comp)
    best_j2_neurons, best_j2_corr, best_j2_comp = [], [], []
    for comp in range(3):
        idx, corr = find_selective_neurons(hidden_flat, j2_rotvec[:, comp], top_k=1)
        best_j2_neurons.append(idx[0])
        best_j2_corr.append(corr[0])
        best_j2_comp.append(comp)
    print(f"Most J1-selective neurons: {best_j1_neurons} (r={best_j1_corr})")
    print(f"Most J2-selective neurons: {best_j2_neurons} (r={best_j2_corr})")

    # ===== PLOT =====
    n_plot = min(3000, n_points)  # subsample for scatter readability
    plot_idx = np.random.default_rng(0).choice(n_points, n_plot, replace=False)

    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.3)

    model_type = model.hparams.get("model_type", "unknown").upper()
    fig.suptitle(
        f"{model_type} Representation Analysis "
        f"(vMF, kappa=5, hidden={hidden_size})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # ── Row 1: PCA spectrum ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(
        range(min(20, len(pca.explained_variance_ratio_))),
        pca.explained_variance_ratio_[:20],
        color="steelblue",
    )
    ax.set_xlabel("PC Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("(A) PCA Spectrum\nVariance per principal component")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(range(1, min(21, len(cumvar) + 1)), cumvar[:20], "o-", color="steelblue")
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95%")
    ax.axvline(x=n95, color="red", linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative Variance")
    ax.set_title(f"(B) Cumulative Variance\n{n95} PCs for 95%")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.text(
        0.5,
        0.5,
        f"Model: {model_type} (hidden={hidden_size})\n"
        f"Output: {model.output_size} ({model.cells_per_joint}/joint)\n"
        f"Dropout: {model.hparams.get('dropout', '?')}\n"
        f"Checkpoint: {Path(args.checkpoint).name}\n\n"
        f"PCs for 95%: {n95}\n"
        f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%\n"
        f"Top-3: {cumvar[2]*100:.1f}%\n"
        f"Top-6: {cumvar[5]*100:.1f}%\n\n"
        f"J1-selective (x,y,z): {best_j1_neurons}\n"
        f"  r = [{', '.join(f'{c:.3f}' for c in best_j1_corr)}]\n"
        f"J2-selective (x,y,z): {best_j2_neurons}\n"
        f"  r = [{', '.join(f'{c:.3f}' for c in best_j2_corr)}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Summary")
    ax.axis("off")

    # ── Row 2: PCA colored by J1 rotvec x, y, z ─────────────────────────────
    for col, comp in enumerate(range(3)):
        ax = fig.add_subplot(gs[1, col])
        sc = ax.scatter(
            hidden_pca[plot_idx, 0],
            hidden_pca[plot_idx, 1],
            c=j1_rotvec[plot_idx, comp],
            cmap="coolwarm",
            s=2,
            alpha=0.5,
        )
        plt.colorbar(sc, ax=ax, label=f"J1 rotvec {axis_labels[comp]}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"(C{comp+1}) PC1 vs PC2\nJ1 rotation {axis_labels[comp]}-component"
        )

    # ── Row 3: t-SNE colored by J1 x, J2 x, cell ID ────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    sc = ax.scatter(
        hidden_tsne[:, 0],
        hidden_tsne[:, 1],
        c=j1_rotvec[tsne_idx, 0],
        cmap="coolwarm",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="J1 rotvec x")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("(D1) t-SNE\nJ1 rotation x-component")

    ax = fig.add_subplot(gs[2, 1])
    sc = ax.scatter(
        hidden_tsne[:, 0],
        hidden_tsne[:, 1],
        c=j2_rotvec[tsne_idx, 0],
        cmap="coolwarm",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="J2 rotvec x")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("(D2) t-SNE\nJ2 rotation x-component")

    ax = fig.add_subplot(gs[2, 2])
    sc = ax.scatter(
        hidden_tsne[:, 0],
        hidden_tsne[:, 1],
        c=j1_cell[tsne_idx],
        cmap="tab20",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Dominant cell ID")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("(D3) t-SNE\nColored by dominant cell ID")

    # ── Row 4: Tuning — most J1-selective neurons (one per x,y,z) ────────────
    for i in range(3):
        neuron_idx = best_j1_neurons[i]
        corr = best_j1_corr[i]
        comp = best_j1_comp[i]
        ax = fig.add_subplot(gs[3, i])
        _plot_tuning_curve(
            ax,
            hidden_flat[:, neuron_idx],
            j1_rotvec[:, comp],
            f"Neuron {neuron_idx} vs J1-{axis_labels[comp]} (r={corr:.3f})",
            f"Joint 1 rotvec {axis_labels[comp]}",
            "blue",
        )

    # ── Row 5: Tuning — most J2-selective neurons (one per x,y,z) ────────────
    for i in range(3):
        neuron_idx = best_j2_neurons[i]
        corr = best_j2_corr[i]
        comp = best_j2_comp[i]
        ax = fig.add_subplot(gs[4, i])
        _plot_tuning_curve(
            ax,
            hidden_flat[:, neuron_idx],
            j2_rotvec[:, comp],
            f"Neuron {neuron_idx} vs J2-{axis_labels[comp]} (r={corr:.3f})",
            f"Joint 2 rotvec {axis_labels[comp]}",
            "red",
        )

    # Save
    if args.output:
        output_path = args.output
    else:
        output_path = str(
            Path(args.checkpoint).parent.parent / "representation_analysis.png"
        )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


def _plot_tuning_curve(ax, activations, angles, title, xlabel, color):
    """Plot binned tuning curve with scatter and confidence band."""
    n_bins = 50
    bins = np.linspace(angles.min(), angles.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (angles >= bins[b]) & (angles < bins[b + 1])
        if mask.sum() > 0:
            bin_means[b] = activations[mask].mean()
            bin_stds[b] = activations[mask].std()

    ax.fill_between(
        bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.2, color=color
    )
    ax.plot(bin_centers, bin_means, "-", linewidth=2, color=color)

    # Subsample scatter
    n_show = min(2000, len(angles))
    idx = np.random.default_rng(0).choice(len(angles), n_show, replace=False)
    ax.scatter(angles[idx], activations[idx], s=1, alpha=0.08, color="gray")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Activation")
    ax.set_title(title)


if __name__ == "__main__":
    main()
