"""Tests for robot arm kinematics."""

import numpy as np
from scipy.spatial.transform import Rotation

from articulated.shared.robot_arm import RobotArmKinematics


class TestIntegrateVelocity:
    """Tests for integrate_velocity."""

    def test_zero_velocity_preserves_orientation(self):
        """Zero angular velocity should not change orientation."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)
        omega = np.zeros(6)

        new_config = kin.integrate_velocity(config, omega, dt=0.01)

        # Relative rotation should be identity
        d1 = (config[0].inv() * new_config[0]).magnitude()
        d2 = (config[1].inv() * new_config[1]).magnitude()
        assert d1 < 1e-10
        assert d2 < 1e-10

    def test_output_is_valid_rotations(self):
        """Output should be a pair of valid Rotation objects."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)
        omega = rng.standard_normal(6)

        new_config = kin.integrate_velocity(config, omega, dt=0.01)

        assert isinstance(new_config[0], Rotation)
        assert isinstance(new_config[1], Rotation)
        # Rotation matrices should be orthogonal
        R1 = new_config[0].as_matrix()
        np.testing.assert_allclose(R1 @ R1.T, np.eye(3), atol=1e-10)

    def test_nonzero_velocity_changes_orientation(self):
        """Nonzero velocity should change the orientation."""
        kin = RobotArmKinematics()
        config = (Rotation.identity(), Rotation.identity())
        omega = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        new_config = kin.integrate_velocity(config, omega, dt=0.1)

        d1 = (config[0].inv() * new_config[0]).magnitude()
        d2 = (config[1].inv() * new_config[1]).magnitude()
        assert d1 > 0.05
        assert d2 > 0.05


class TestGeodesicDistance:
    """Tests for geodesic_distance."""

    def test_distance_to_self_is_zero(self):
        """Distance from a config to itself should be zero."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)

        d = kin.geodesic_distance(config, config)
        assert abs(d) < 1e-10

    def test_symmetry(self):
        """Distance should be symmetric: d(a, b) == d(b, a)."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)

        d12 = kin.geodesic_distance(c1, c2)
        d21 = kin.geodesic_distance(c2, c1)
        assert abs(d12 - d21) < 1e-10

    def test_triangle_inequality(self):
        """Triangle inequality: d(a, c) <= d(a, b) + d(b, c)."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)
        c3 = kin.sample_random_configuration(rng)

        d12 = kin.geodesic_distance(c1, c2)
        d23 = kin.geodesic_distance(c2, c3)
        d13 = kin.geodesic_distance(c1, c3)

        assert d13 <= d12 + d23 + 1e-10

    def test_nonnegative(self):
        """Distance should always be non-negative."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)

        assert kin.geodesic_distance(c1, c2) >= 0.0


class TestForwardKinematics:
    """Tests for forward_kinematics."""

    def test_identity_rotations(self):
        """Both joints at identity: end-effector at (L1+L2, 0, 0)."""
        kin = RobotArmKinematics(link_lengths=(1.0, 1.0))
        config = (Rotation.identity(), Rotation.identity())

        pos = kin.forward_kinematics(config)
        np.testing.assert_allclose(pos, [2.0, 0.0, 0.0], atol=1e-10)

    def test_custom_link_lengths(self):
        """Different link lengths with identity rotations."""
        kin = RobotArmKinematics(link_lengths=(2.0, 3.0))
        config = (Rotation.identity(), Rotation.identity())

        pos = kin.forward_kinematics(config)
        np.testing.assert_allclose(pos, [5.0, 0.0, 0.0], atol=1e-10)

    def test_output_shape(self):
        """Forward kinematics should return a 3D position."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)

        pos = kin.forward_kinematics(config)
        assert pos.shape == (3,)

    def test_end_effector_bounded(self):
        """End-effector should be within L1 + L2 of origin."""
        kin = RobotArmKinematics(link_lengths=(1.0, 1.0))
        rng = np.random.default_rng(42)

        for _ in range(10):
            config = kin.sample_random_configuration(rng)
            pos = kin.forward_kinematics(config)
            dist = np.linalg.norm(pos)
            assert dist <= 2.0 + 1e-10
