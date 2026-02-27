"""Launch scaling sweeps for estimation representation experiments.

This script composes:
1. A base training config (full YAML for `python -m articulated.estimation.train`)
2. One or more architecture override configs
3. A grid over `train_sizes x seeds`

For each run, it writes a per-run config, launches training, evaluates the best
checkpoint on validation data, computes lightweight representation metrics, and
saves `summary.json` under the run directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class SweepSettings:
    name: str
    base_config_path: Path
    architecture_paths: list[Path]
    train_sizes: list[int]
    seeds: list[int]
    config_overrides: dict[str, Any]
    output_root: Path
    python_executable: str
    skip_existing: bool
    stop_on_error: bool
    representation_n_trajectories: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run estimation scaling sweep")
    parser.add_argument(
        "--sweep-config",
        type=Path,
        required=True,
        help="Path to sweep YAML config",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root from sweep config",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Override python executable from sweep config",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optionally limit the number of runs (useful for smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without launching training",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {k: copy.deepcopy(v) for k, v in base.items()}
        for k, v in override.items():
            if k in merged:
                merged[k] = deep_merge(merged[k], v)
            else:
                merged[k] = copy.deepcopy(v)
        return merged
    return copy.deepcopy(override)


def parse_sweep_settings(
    sweep_config_path: Path,
    output_root_override: Path | None,
    python_override: str | None,
) -> SweepSettings:
    raw = load_yaml(sweep_config_path)

    def _resolve_repo_path(path_like: str) -> Path:
        path = Path(path_like)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    name = raw["name"]
    base_config_path = _resolve_repo_path(raw["base_config"])
    architecture_paths = [_resolve_repo_path(p) for p in raw["architectures"]]
    train_sizes = [int(x) for x in raw["train_sizes"]]
    seeds = [int(x) for x in raw["seeds"]]

    config_overrides = raw.get("config_overrides", {})
    run_cfg = raw.get("run", {})
    eval_cfg = raw.get("evaluation", {})

    output_root = (
        output_root_override
        if output_root_override is not None
        else _resolve_repo_path(run_cfg.get("output_root", "runs"))
    )
    python_executable = python_override or run_cfg.get("python_executable", "python")

    return SweepSettings(
        name=name,
        base_config_path=base_config_path,
        architecture_paths=architecture_paths,
        train_sizes=train_sizes,
        seeds=seeds,
        config_overrides=config_overrides,
        output_root=output_root,
        python_executable=python_executable,
        skip_existing=bool(run_cfg.get("skip_existing", True)),
        stop_on_error=bool(run_cfg.get("stop_on_error", False)),
        representation_n_trajectories=int(
            eval_cfg.get("representation_n_trajectories", 200)
        ),
    )


def architecture_name_and_overrides(path: Path) -> tuple[str, dict[str, Any]]:
    raw = load_yaml(path)
    name = raw.get("name", path.stem)
    overrides = raw.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Architecture overrides must be a dict: {path}")
    return name, overrides


def run_dir_for(settings: SweepSettings, arch_name: str, train_size: int, seed: int) -> Path:
    return (
        settings.output_root
        / settings.name
        / arch_name
        / f"n{train_size}"
        / f"seed{seed}"
    )


def configure_run(
    base_config: dict[str, Any],
    architecture_overrides: dict[str, Any],
    config_overrides: dict[str, Any],
    run_dir: Path,
    train_size: int,
    seed: int,
) -> dict[str, Any]:
    cfg = deep_merge(base_config, architecture_overrides)
    cfg = deep_merge(cfg, config_overrides)

    cfg["seed"] = seed
    cfg.setdefault("data", {})
    cfg["data"]["n_trajectories_train"] = train_size

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("wandb", False)
    cfg["logging"]["save_dir"] = str(run_dir / "logs")

    cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    return cfg


def run_training(
    python_executable: str,
    run_config_path: Path,
    run_dir: Path,
) -> tuple[int, float]:
    command = [
        python_executable,
        "-m",
        "articulated.estimation.train",
        "--config",
        str(run_config_path),
    ]

    log_path = run_dir / "train.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start
    return proc.returncode, elapsed


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_from_filename(path: Path) -> float | None:
    text = path.name
    match = re.search(r"val[_/\\-]?loss[=:_-]?([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))

    all_floats = re.findall(r"([0-9]+\.[0-9]+)", text)
    if all_floats:
        return float(all_floats[-1])
    return None


def checkpoint_score(path: Path) -> float:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        callbacks = ckpt.get("callbacks", {})
        if isinstance(callbacks, dict):
            for state in callbacks.values():
                if isinstance(state, dict):
                    current_score = _safe_float(state.get("current_score"))
                    if current_score is not None:
                        return current_score
    except Exception:
        pass

    parsed = _score_from_filename(path)
    if parsed is not None:
        return parsed

    return math.inf


def find_best_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, float | None]:
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"))
    if not ckpts:
        return None, None

    best_path = None
    best_score = math.inf
    for ckpt_path in ckpts:
        score = checkpoint_score(ckpt_path)
        if score < best_score:
            best_score = score
            best_path = ckpt_path

    if best_path is None:
        return None, None
    return best_path, (None if math.isinf(best_score) else best_score)


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _hidden_representation_metrics(hidden_flat: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for representation metrics. "
            "Install `scikit-learn` in your environment."
        ) from exc

    if hidden_flat.shape[0] < 3:
        return {
            "hidden_points": int(hidden_flat.shape[0]),
            "hidden_dim": int(hidden_flat.shape[1]),
            "pca_95_dims": float("nan"),
            "pca_pc1_var": float("nan"),
            "participation_ratio": float("nan"),
        }

    pca_n = min(50, hidden_flat.shape[1], hidden_flat.shape[0])
    pca = PCA(n_components=pca_n)
    pca.fit(hidden_flat)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    variances = np.var(hidden_flat, axis=0)
    v_sum = float(np.sum(variances))
    v_sq_sum = float(np.sum(variances**2))
    participation = (v_sum**2 / v_sq_sum) if v_sq_sum > 0.0 else float("nan")

    return {
        "hidden_points": int(hidden_flat.shape[0]),
        "hidden_dim": int(hidden_flat.shape[1]),
        "pca_95_dims": int(np.searchsorted(cumulative, 0.95) + 1),
        "pca_pc1_var": float(explained[0]),
        "participation_ratio": participation,
    }


def _so2_selectivity_metrics(
    hidden_states: np.ndarray,
    velocities: np.ndarray,
    init_encoded: np.ndarray,
    dt: float,
) -> dict[str, float]:
    n_traj, seq_len, _ = velocities.shape

    theta1 = np.arctan2(init_encoded[:, 1], init_encoded[:, 0]) % (2.0 * np.pi)
    theta2 = np.arctan2(init_encoded[:, 3], init_encoded[:, 2]) % (2.0 * np.pi)

    traj_theta1 = np.zeros((n_traj, seq_len), dtype=np.float64)
    traj_theta2 = np.zeros((n_traj, seq_len), dtype=np.float64)

    for t in range(seq_len):
        theta1 = (theta1 + velocities[:, t, 0] * dt) % (2.0 * np.pi)
        theta2 = (theta2 + velocities[:, t, 1] * dt) % (2.0 * np.pi)
        traj_theta1[:, t] = theta1
        traj_theta2[:, t] = theta2

    y1_sin = np.sin(traj_theta1).reshape(-1)
    y1_cos = np.cos(traj_theta1).reshape(-1)
    y2_sin = np.sin(traj_theta2).reshape(-1)
    y2_cos = np.cos(traj_theta2).reshape(-1)

    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

    corr_j1 = np.zeros(hidden_flat.shape[1], dtype=np.float64)
    corr_j2 = np.zeros(hidden_flat.shape[1], dtype=np.float64)

    for i in range(hidden_flat.shape[1]):
        h = hidden_flat[:, i]
        corr_j1[i] = max(_safe_abs_corr(h, y1_sin), _safe_abs_corr(h, y1_cos))
        corr_j2[i] = max(_safe_abs_corr(h, y2_sin), _safe_abs_corr(h, y2_cos))

    top3_j1 = np.sort(corr_j1)[-3:][::-1]
    top3_j2 = np.sort(corr_j2)[-3:][::-1]

    return {
        "so2_max_corr_theta1": float(np.max(corr_j1)),
        "so2_max_corr_theta2": float(np.max(corr_j2)),
        "so2_top3_mean_corr_theta1": float(np.mean(top3_j1)),
        "so2_top3_mean_corr_theta2": float(np.mean(top3_j2)),
    }


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
    n_repr_trajectories: int,
) -> dict[str, Any]:
    from articulated.estimation.datamodule import EstimationDataModule
    from articulated.estimation.model import StateEstimationModel

    dm = EstimationDataModule(**config.get("data", {}))
    dm.setup(stage="fit")

    model = StateEstimationModel.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )
    model.eval()

    val_loader = dm.val_dataloader()
    total_weight = 0.0
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in val_loader:
            loss, acc, _ = model._shared_step(batch)
            batch_size = int(batch[0].shape[0])
            total_weight += batch_size
            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc.item()) * batch_size

    if total_weight == 0:
        raise RuntimeError("Validation loader is empty; cannot compute evaluation metrics")

    eval_metrics: dict[str, Any] = {
        "val_loss": total_loss / total_weight,
        "val_acc": total_acc / total_weight,
    }

    if dm.val_dataset is None:
        return eval_metrics

    val_tensors = dm.val_dataset.tensors
    n_traj = min(int(val_tensors[0].shape[0]), int(n_repr_trajectories))
    if n_traj <= 0:
        return eval_metrics

    velocities = val_tensors[0][:n_traj]
    init_pos = val_tensors[2][:n_traj] if len(val_tensors) >= 3 else None

    with torch.no_grad():
        if model.use_init_pos and init_pos is not None:
            hidden0 = model._encode_init_pos(init_pos)
            _, hidden_states = model(velocities, hidden=hidden0)
        else:
            _, hidden_states = model(velocities)

    hidden_np = hidden_states.detach().cpu().numpy()
    hidden_flat = hidden_np.reshape(-1, hidden_np.shape[-1])
    repr_metrics = _hidden_representation_metrics(hidden_flat)

    if (
        init_pos is not None
        and velocities.shape[-1] == 2
        and init_pos.shape[-1] == 4
    ):
        repr_metrics.update(
            _so2_selectivity_metrics(
                hidden_states=hidden_np,
                velocities=velocities.detach().cpu().numpy(),
                init_encoded=init_pos.detach().cpu().numpy(),
                dt=float(config.get("data", {}).get("dt", 0.01)),
            )
        )

    eval_metrics["representation"] = repr_metrics
    return eval_metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    settings = parse_sweep_settings(
        sweep_config_path=args.sweep_config,
        output_root_override=args.output_root,
        python_override=args.python_executable,
    )

    base_config = load_yaml(settings.base_config_path)

    run_plan: list[tuple[str, Path, dict[str, Any], int, int, Path]] = []
    for arch_path in settings.architecture_paths:
        arch_name, arch_overrides = architecture_name_and_overrides(arch_path)
        for train_size in settings.train_sizes:
            for seed in settings.seeds:
                run_dir = run_dir_for(settings, arch_name, train_size, seed)
                run_plan.append(
                    (arch_name, arch_path, arch_overrides, train_size, seed, run_dir)
                )

    if args.max_runs is not None:
        run_plan = run_plan[: args.max_runs]

    print(f"Sweep: {settings.name}")
    print(f"Planned runs: {len(run_plan)}")
    print(f"Output root: {settings.output_root}")

    failures = 0
    for idx, (arch_name, arch_path, arch_overrides, train_size, seed, run_dir) in enumerate(
        run_plan, start=1
    ):
        summary_path = run_dir / "summary.json"

        if settings.skip_existing and summary_path.exists():
            print(
                f"[{idx}/{len(run_plan)}] skip existing {arch_name} n={train_size} seed={seed}"
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = configure_run(
            base_config=base_config,
            architecture_overrides=arch_overrides,
            config_overrides=settings.config_overrides,
            run_dir=run_dir,
            train_size=train_size,
            seed=seed,
        )

        run_config_path = run_dir / "config.yaml"
        dump_yaml(run_config, run_config_path)

        metadata: dict[str, Any] = {
            "sweep_name": settings.name,
            "architecture": arch_name,
            "architecture_config": str(arch_path),
            "train_size": train_size,
            "seed": seed,
            "run_dir": str(run_dir),
            "config_path": str(run_config_path),
            "status": "planned",
        }

        if args.dry_run:
            metadata["status"] = "dry_run"
            write_json(summary_path, metadata)
            print(
                f"[{idx}/{len(run_plan)}] dry-run {arch_name} n={train_size} seed={seed}"
            )
            continue

        print(f"[{idx}/{len(run_plan)}] start {arch_name} n={train_size} seed={seed}")

        train_return_code, train_seconds = run_training(
            python_executable=settings.python_executable,
            run_config_path=run_config_path,
            run_dir=run_dir,
        )

        metadata["train_return_code"] = train_return_code
        metadata["train_seconds"] = train_seconds

        if train_return_code != 0:
            metadata["status"] = "train_failed"
            write_json(summary_path, metadata)
            failures += 1
            print(
                f"[{idx}/{len(run_plan)}] FAILED training {arch_name} n={train_size} seed={seed}"
            )
            if settings.stop_on_error:
                break
            continue

        checkpoint_dir = Path(run_config["checkpoint_dir"])
        best_ckpt, best_ckpt_score = find_best_checkpoint(checkpoint_dir)

        if best_ckpt is None:
            metadata["status"] = "no_checkpoint"
            write_json(summary_path, metadata)
            failures += 1
            print(
                f"[{idx}/{len(run_plan)}] FAILED no checkpoint {arch_name} n={train_size} seed={seed}"
            )
            if settings.stop_on_error:
                break
            continue

        metadata["best_checkpoint_path"] = str(best_ckpt)
        metadata["best_checkpoint_score"] = best_ckpt_score

        try:
            eval_metrics = evaluate_checkpoint(
                config=run_config,
                checkpoint_path=best_ckpt,
                n_repr_trajectories=settings.representation_n_trajectories,
            )
        except Exception as exc:
            metadata["status"] = "eval_failed"
            metadata["error"] = str(exc)
            write_json(summary_path, metadata)
            failures += 1
            print(
                f"[{idx}/{len(run_plan)}] FAILED eval {arch_name} n={train_size} seed={seed}"
            )
            if settings.stop_on_error:
                break
            continue

        metadata["status"] = "completed"
        metadata.update(eval_metrics)
        write_json(summary_path, metadata)
        print(
            f"[{idx}/{len(run_plan)}] done {arch_name} n={train_size} seed={seed} "
            f"val_loss={eval_metrics.get('val_loss', float('nan')):.6f} "
            f"val_acc={eval_metrics.get('val_acc', float('nan')):.4f}"
        )

    print(f"Sweep finished. failures={failures}")


if __name__ == "__main__":
    main()
