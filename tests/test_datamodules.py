"""Tests for data modules."""

import numpy as np

from articulated.estimation.datamodule import EstimationDataModule


class TestEstimationDataModule:
    """Tests for EstimationDataModule."""

    def test_initialization(self):
        """Test that data module initializes with correct parameters."""
        dm = EstimationDataModule(
            batch_size=16,
            seq_length=50,
            n_trajectories_train=100,
            n_trajectories_val=20,
            n_place_cells=64,
            seed=42,
        )

        assert dm.batch_size == 16
        assert dm.seq_length == 50
        assert dm.n_trajectories_train == 100
        assert dm.n_place_cells == 64
        assert dm.n_cells_per_joint == 32

    def test_setup_runs(self):
        """Setup should complete without error."""
        dm = EstimationDataModule(
            batch_size=4,
            seq_length=10,
            n_trajectories_train=5,
            n_trajectories_val=2,
            n_place_cells=8,
            seed=42,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_dataset_shapes_with_init_pos(self):
        """Datasets should have 3 tensors when provide_init_pos=True."""
        seq_len = 10
        n_train = 5
        n_val = 2
        n_pc = 8

        dm = EstimationDataModule(
            batch_size=4,
            seq_length=seq_len,
            n_trajectories_train=n_train,
            n_trajectories_val=n_val,
            n_place_cells=n_pc,
            seed=42,
            provide_init_pos=True,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        train_vel, train_tgt, train_init = dm.train_dataset.tensors
        assert train_vel.shape == (n_train, seq_len, 6)
        assert train_tgt.shape == (n_train, seq_len, n_pc)
        assert train_init.shape == (n_train, n_pc)

        assert dm.val_dataset is not None
        val_vel, val_tgt, val_init = dm.val_dataset.tensors
        assert val_vel.shape == (n_val, seq_len, 6)
        assert val_tgt.shape == (n_val, seq_len, n_pc)
        assert val_init.shape == (n_val, n_pc)

    def test_dataset_shapes_without_init_pos(self):
        """Datasets should have 2 tensors when provide_init_pos=False."""
        seq_len = 10
        n_train = 5
        n_val = 2
        n_pc = 8

        dm = EstimationDataModule(
            batch_size=4,
            seq_length=seq_len,
            n_trajectories_train=n_train,
            n_trajectories_val=n_val,
            n_place_cells=n_pc,
            seed=42,
            provide_init_pos=False,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert len(dm.train_dataset.tensors) == 2
        train_vel, train_tgt = dm.train_dataset.tensors
        assert train_vel.shape == (n_train, seq_len, 6)
        assert train_tgt.shape == (n_train, seq_len, n_pc)

    def test_per_joint_targets_are_distributions(self):
        """Each joint's place cell targets should sum to ~1."""
        n_pc = 16
        n_per_joint = n_pc // 2
        dm = EstimationDataModule(
            batch_size=4,
            seq_length=10,
            n_trajectories_train=3,
            n_trajectories_val=1,
            n_place_cells=n_pc,
            seed=42,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        targets = dm.train_dataset.tensors[1]
        # Each joint half should sum to 1
        j1_sums = targets[..., :n_per_joint].sum(dim=-1).numpy()
        j2_sums = targets[..., n_per_joint:].sum(dim=-1).numpy()
        np.testing.assert_allclose(j1_sums, 1.0, atol=1e-5)
        np.testing.assert_allclose(j2_sums, 1.0, atol=1e-5)

    def test_per_joint_init_pcs_are_distributions(self):
        """Initial place cell activations should sum to ~1 per joint."""
        n_pc = 16
        n_per_joint = n_pc // 2
        dm = EstimationDataModule(
            batch_size=4,
            seq_length=10,
            n_trajectories_train=3,
            n_trajectories_val=1,
            n_place_cells=n_pc,
            seed=42,
            provide_init_pos=True,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        init_pcs = dm.train_dataset.tensors[2]
        j1_sums = init_pcs[:, :n_per_joint].sum(dim=-1).numpy()
        j2_sums = init_pcs[:, n_per_joint:].sum(dim=-1).numpy()
        np.testing.assert_allclose(j1_sums, 1.0, atol=1e-5)
        np.testing.assert_allclose(j2_sums, 1.0, atol=1e-5)

    def test_place_cell_targets_nonnegative(self):
        """Place cell targets should be non-negative."""
        dm = EstimationDataModule(
            batch_size=4,
            seq_length=10,
            n_trajectories_train=3,
            n_trajectories_val=1,
            n_place_cells=16,
            seed=42,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        targets = dm.train_dataset.tensors[1]
        assert (targets >= 0).all()

    def test_targets_not_uniform(self):
        """vMF targets should NOT be uniform (key regression test)."""
        n_pc = 16
        n_per_joint = n_pc // 2
        dm = EstimationDataModule(
            batch_size=4,
            seq_length=10,
            n_trajectories_train=3,
            n_trajectories_val=1,
            n_place_cells=n_pc,
            place_cell_kappa=5.0,
            seed=42,
        )
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        targets = dm.train_dataset.tensors[1]
        j1 = targets[0, 0, :n_per_joint].numpy()
        # Max activation should be significantly above uniform
        assert j1.max() > 2.0 / n_per_joint

    def test_dataloader_batch_shape(self):
        """Dataloaders should yield batches of the correct shape."""
        batch_size = 4
        seq_len = 10
        n_pc = 8

        dm = EstimationDataModule(
            batch_size=batch_size,
            seq_length=seq_len,
            n_trajectories_train=8,
            n_trajectories_val=4,
            n_place_cells=n_pc,
            seed=42,
        )
        dm.setup(stage="fit")

        batch = next(iter(dm.train_dataloader()))
        vel, tgt, init_pc = batch
        assert vel.shape == (batch_size, seq_len, 6)
        assert tgt.shape == (batch_size, seq_len, n_pc)
        assert init_pc.shape == (batch_size, n_pc)

    def test_place_cells_initialized(self):
        """Place cell centers should be initialized after setup."""
        dm = EstimationDataModule(
            batch_size=4,
            seq_length=10,
            n_trajectories_train=3,
            n_trajectories_val=1,
            n_place_cells=16,
            seed=42,
        )
        dm.setup(stage="fit")

        assert dm.place_cell_centers is not None
        assert len(dm.place_cell_centers) == 8  # n_cells_per_joint
