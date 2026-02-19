"""Train state estimation model.

Usage:
    python scripts/train.py
    python scripts/train.py --data_path data/estimation/trajectories.pt
    python scripts/train.py --model_type gru --dropout 0.5 --epochs 200
    python scripts/train.py --wandb --project articulated-estimation
    python scripts/train.py --help
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from articulated.estimation.datamodule import EstimationDataModule
from articulated.estimation.model import StateEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train state estimation model")

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/estimation/trajectories.pt",
        help="Pre-generated .pt data",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=3000,
        help="Training trajectories (if no data_path)",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=500,
        help="Validation trajectories (if no data_path)",
    )
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length")
    parser.add_argument(
        "--n_place_cells", type=int, default=64, help="Number of place cells"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration timestep")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Model
    parser.add_argument(
        "--model_type", type=str, default="gru", choices=["rnn", "lstm", "gru"]
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden state size"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate (0.5 per Banino 2018)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="L2 regularization"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--accelerator", type=str, default="auto", help="cpu/gpu/auto")
    parser.add_argument(
        "--devices", type=str, default="auto", help="Number of devices or 'auto'"
    )
    parser.add_argument(
        "--strategy", type=str, default="auto", help="Training strategy (ddp/auto)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value"
    )

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--project",
        type=str,
        default="articulated-estimation",
        help="WandB project name",
    )
    parser.add_argument(
        "--output_dir", type=str, default="logs/estimation", help="Output directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    L.seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse devices (could be int or "auto")
    try:
        devices = int(args.devices)
    except ValueError:
        devices = args.devices

    print(
        f"Model:     {args.model_type.upper()}, hidden={args.hidden_size}, dropout={args.dropout}"
    )
    print(
        f"Training:  {args.epochs} epochs, lr={args.lr}, wd={args.weight_decay}, grad_clip={args.grad_clip}"
    )
    print(
        f"Hardware:  accelerator={args.accelerator}, devices={devices}, strategy={args.strategy}"
    )
    if args.data_path:
        print(f"Data:      loading from {args.data_path}")
    else:
        print(
            f"Data:      generating {args.n_train} train, {args.n_val} val, seq_len={args.seq_length}"
        )
    print(f"Logging:   {'WandB' if args.wandb else 'TensorBoard + CSV'}")
    print(f"Output:    {output_dir}")
    print()

    # Data
    dm = EstimationDataModule(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        n_trajectories_train=args.n_train,
        n_trajectories_val=args.n_val,
        n_place_cells=args.n_place_cells,
        dt=args.dt,
        seed=args.seed,
        provide_init_pos=True,
        data_path=args.data_path,
        num_workers=args.num_workers,
    )

    # Model
    model = StateEstimationModel(
        input_size=6,
        hidden_size=args.hidden_size,
        output_size=args.n_place_cells,
        model_type=args.model_type,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_init_pos=True,
        dropout=args.dropout,
    )

    # Loggers
    loggers = []
    loggers.append(CSVLogger(save_dir=str(output_dir), name="csv_logs"))
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(WandbLogger(project=args.project, save_dir=str(output_dir)))

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, dm)

    # Save final model
    save_path = output_dir / "trained_model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "hparams": model.hparams},
        save_path,
    )
    print(f"\nDone! Model saved to {save_path}")
    print(f"Checkpoints in {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
