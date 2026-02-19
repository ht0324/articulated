"""Data module for state estimation training.

Tasks:
1. Generate or load trajectories of joint angular velocities
2. Compute ground-truth joint configurations via integration
3. Create place cell targets tiled over SO(3) x SO(3)
"""

from typing import Optional

import lightning as L
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from articulated.shared.robot_arm import RobotArmKinematics


class EstimationDataModule(L.LightningDataModule):
    """DataModule for state estimation training.

    Generates trajectories on SO(3) x SO(3) configuration space.
    Input: angular velocities (6D per timestep)
    Target: place cell activations encoding current configuration
    """

    def __init__(
        self,
        batch_size: int = 64,
        seq_length: int = 100,
        n_trajectories_train: int = 1000,
        n_trajectories_val: int = 100,
        n_place_cells: int = 64,
        dt: float = 0.01,
        seed: Optional[int] = None,
        velocity_sigma: float = 1.0,
        velocity_theta: float = 2.0,
        provide_init_pos: bool = True,
        place_cell_kappa: float = 5.0,
        data_path: Optional[str] = None,
        num_workers: int = 4,
    ):
        """Initialize the data module.

        Args:
            batch_size: Batch size for data loaders.
            seq_length: Length of each trajectory sequence.
            n_trajectories_train: Number of training trajectories.
            n_trajectories_val: Number of validation trajectories.
            n_place_cells: Total number of place cells (split equally between
                joints, so must be even). Each joint gets n_place_cells // 2
                cells with independent softmax distributions.
            dt: Time step for integration.
            seed: Random seed for reproducibility.
            velocity_sigma: Steady-state std of OU angular velocity process.
            velocity_theta: Mean-reversion rate of OU angular velocity process.
            provide_init_pos: Whether to include initial position place cell
                activations in the dataset (needed for path integration).
            place_cell_kappa: Concentration parameter for von Mises-Fisher kernel
                on SO(3). Higher = more peaked. kappa=5 gives ~4-5 effective
                cells per joint out of 32. Gaussian RBF fails on SO(3) due to
                the bounded geometry; vMF is the natural analogue.
            data_path: Path to pre-generated .pt data file. If provided, loads
                data from disk instead of generating. Use scripts/generate_data.py
                to create these files.
            num_workers: Number of DataLoader workers.
        """
        super().__init__()
        self.save_hyperparameters()

        assert (
            n_place_cells % 2 == 0
        ), "n_place_cells must be even (split between 2 joints)"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_trajectories_train = n_trajectories_train
        self.n_trajectories_val = n_trajectories_val
        self.n_place_cells = n_place_cells
        self.n_cells_per_joint = n_place_cells // 2
        self.dt = dt
        self.seed = seed
        self.velocity_sigma = velocity_sigma
        self.velocity_theta = velocity_theta
        self.provide_init_pos = provide_init_pos
        self.place_cell_kappa = place_cell_kappa
        self.data_path = data_path
        self.num_workers = num_workers

        self.kinematics = RobotArmKinematics()
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None

        # Place cell centers - to be initialized (separate per joint)
        self.place_cell_centers: Optional[list] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Generate or load trajectory datasets."""
        if stage == "fit" or stage is None:
            if self.data_path is not None:
                self._load_from_disk()
            else:
                self._generate_all_data()

    def _load_from_disk(self) -> None:
        """Load pre-generated data from a .pt file."""
        assert self.data_path is not None
        print(f"Loading data from {self.data_path}...")
        data = torch.load(self.data_path, weights_only=True)

        train_tensors = [data["train_velocities"], data["train_targets"]]
        if self.provide_init_pos and "train_init_pcs" in data:
            train_tensors.append(data["train_init_pcs"])
        self.train_dataset = TensorDataset(*train_tensors)

        val_tensors = [data["val_velocities"], data["val_targets"]]
        if self.provide_init_pos and "val_init_pcs" in data:
            val_tensors.append(data["val_init_pcs"])
        self.val_dataset = TensorDataset(*val_tensors)

        # Restore place cell centers for analysis
        if "place_cell_quats1" in data and "place_cell_quats2" in data:
            self._centers_R1 = Rotation.from_quat(data["place_cell_quats1"].numpy())
            self._centers_R2 = Rotation.from_quat(data["place_cell_quats2"].numpy())

        n_train = data["train_velocities"].shape[0]
        n_val = data["val_velocities"].shape[0]
        print(f"Loaded {n_train} train + {n_val} val trajectories")

    def _generate_all_data(self) -> None:
        """Generate all train/val data from scratch."""
        rng = np.random.default_rng(self.seed)

        # Initialize place cells
        self._initialize_place_cells(rng)

        # Generate training data
        train_velocities, train_targets, train_init_pcs = self._generate_trajectories(
            self.n_trajectories_train, rng
        )
        train_tensors = [
            torch.from_numpy(train_velocities).float(),
            torch.from_numpy(train_targets).float(),
        ]
        if self.provide_init_pos:
            train_tensors.append(torch.from_numpy(train_init_pcs).float())
        self.train_dataset = TensorDataset(*train_tensors)

        # Generate validation data
        val_velocities, val_targets, val_init_pcs = self._generate_trajectories(
            self.n_trajectories_val, rng
        )
        val_tensors = [
            torch.from_numpy(val_velocities).float(),
            torch.from_numpy(val_targets).float(),
        ]
        if self.provide_init_pos:
            val_tensors.append(torch.from_numpy(val_init_pcs).float())
        self.val_dataset = TensorDataset(*val_tensors)

    def _initialize_place_cells(self, rng: np.random.Generator) -> None:
        """Initialize place cell centers separately for each joint on SO(3).

        Samples n_cells_per_joint random rotations independently for each joint.
        This avoids the curse of dimensionality: 32 cells on 3D SO(3) is much
        more tractable than 64 cells on 6D SO(3) x SO(3).
        """
        n = self.n_cells_per_joint
        self._centers_R1 = Rotation.random(n, random_state=int(rng.integers(0, 2**31)))
        self._centers_R2 = Rotation.random(n, random_state=int(rng.integers(0, 2**31)))
        # Store as list of pairs for compatibility
        self.place_cell_centers = [
            (self._centers_R1[i], self._centers_R2[i]) for i in range(n)
        ]

    def _generate_trajectories(
        self, n_trajectories: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random arm trajectories.

        Args:
            n_trajectories: Number of trajectories to generate.
            rng: Random number generator.

        Returns:
            Tuple of (velocities, place_cell_targets, init_place_cells).
            velocities: Shape (n_trajectories, seq_length, 6).
            targets: Shape (n_trajectories, seq_length, n_place_cells).
            init_place_cells: Shape (n_trajectories, n_place_cells).
        """
        velocities = np.zeros((n_trajectories, self.seq_length, 6))
        targets = np.zeros((n_trajectories, self.seq_length, self.n_place_cells))
        init_pcs = np.zeros((n_trajectories, self.n_place_cells))

        for i in tqdm(range(n_trajectories), desc="Generating trajectories"):
            vel, target, init_pc = self._generate_single_trajectory(rng)
            velocities[i] = vel
            targets[i] = target
            init_pcs[i] = init_pc

        return velocities, targets, init_pcs

    def _generate_single_trajectory(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single trajectory.

        Uses an Ornstein-Uhlenbeck process to generate smooth angular velocities,
        integrates on SO(3) x SO(3), and computes place cell activations.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of (velocities, place_cell_targets, init_place_cells).
            velocities: Shape (seq_length, 6).
            targets: Shape (seq_length, n_place_cells).
            init_place_cells: Shape (n_place_cells,).
        """
        config = self.kinematics.sample_random_configuration(rng)

        # Place cell activation at the starting position
        init_pc = self._compute_place_cell_activations(config)

        decay = np.exp(-self.velocity_theta * self.dt)
        noise_scale = self.velocity_sigma * np.sqrt(1.0 - decay**2)

        velocities = np.zeros((self.seq_length, 6))
        targets = np.zeros((self.seq_length, self.n_place_cells))

        # Initialize angular velocity at zero
        omega = np.zeros(6)

        for t in range(self.seq_length):
            # OU update for angular velocity
            omega = decay * omega + noise_scale * rng.standard_normal(6)
            velocities[t] = omega

            # Integrate to get new configuration
            config = self.kinematics.integrate_velocity(config, omega, self.dt)

            # Compute place cell activations
            targets[t] = self._compute_place_cell_activations(config)

        return velocities, targets, init_pc

    def _compute_place_cell_activations(self, configuration: tuple) -> np.ndarray:
        """Compute place cell activations for a single configuration.

        Uses von Mises-Fisher kernel with separate softmax per joint.
        The vMF kernel exp(kappa * cos(theta)) is the natural analogue of
        Gaussian RBF on the rotation group SO(3), and produces properly
        peaked distributions unlike Gaussian RBF which fails due to the
        bounded geometry of SO(3).

        Args:
            configuration: Current configuration as (Rotation, Rotation).

        Returns:
            Place cell activations of shape (n_place_cells,), which is the
            concatenation of two independent distributions of size
            n_cells_per_joint.
        """
        R1_cur, R2_cur = configuration
        kappa = self.place_cell_kappa

        # Geodesic distances to all centers for each joint
        d1 = (R1_cur.inv() * self._centers_R1).magnitude()
        d2 = (R2_cur.inv() * self._centers_R2).magnitude()

        # vMF kernel + softmax per joint (numerically stable)
        logits1 = kappa * np.cos(d1)
        logits1 -= logits1.max()
        exp1 = np.exp(logits1)
        pc1 = exp1 / exp1.sum()

        logits2 = kappa * np.cos(d2)
        logits2 -= logits2.max()
        exp2 = np.exp(logits2)
        pc2 = exp2 / exp2.sum()

        return np.concatenate([pc1, pc2])

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
