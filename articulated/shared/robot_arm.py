"""Robot arm kinematics and data utilities.

This module provides shared utilities for working with the 2-joint robotic arm
where each joint has full 3D rotational freedom (SO(3)).

Configuration space: Q = SO(3) x SO(3)

Both Team Estimation and Team RL will use these utilities.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class RobotArmKinematics:
    """Kinematics utilities for a 2-joint arm on SO(3) x SO(3).

    Each joint is parameterized by a full 3D rotation (SO(3)).
    The configuration is represented as two rotation matrices or quaternions.
    """

    def __init__(self, link_lengths: tuple[float, float] = (1.0, 1.0)):
        """Initialize robot arm.

        Args:
            link_lengths: Lengths of the two arm segments.
        """
        self.link_lengths = link_lengths

    def integrate_velocity(
        self,
        current_orientation: tuple[Rotation, Rotation],
        angular_velocity: np.ndarray,
        dt: float,
    ) -> tuple[Rotation, Rotation]:
        """Integrate angular velocities to update joint orientations.

        This is the core operation for path integration: given current joint
        orientations and angular velocities, compute the new orientations.

        TODO: Implement proper SO(3) integration using exponential map.

        Args:
            current_orientation: Current orientations of both joints.
            angular_velocity: Angular velocities [omega1, omega2] (6D total).
            dt: Time step.

        Returns:
            Updated orientations for both joints.
        """
        omega1 = angular_velocity[:3]
        omega2 = angular_velocity[3:]

        delta_R1 = Rotation.from_rotvec(omega1 * dt)
        delta_R2 = Rotation.from_rotvec(omega2 * dt)

        R1_new = current_orientation[0] * delta_R1
        R2_new = current_orientation[1] * delta_R2

        return (R1_new, R2_new)

    def forward_kinematics(
        self, joint_orientations: tuple[Rotation, Rotation]
    ) -> np.ndarray:
        """Compute end-effector position from joint orientations.

        TODO: Implement forward kinematics.

        Args:
            joint_orientations: Orientations of both joints.

        Returns:
            End-effector position in 3D space.
        """
        R1, R2 = joint_orientations
        L1, L2 = self.link_lengths

        p1 = L1 * R1.apply([1.0, 0.0, 0.0])
        p2 = L2 * (R1 * R2).apply([1.0, 0.0, 0.0])

        return p1 + p2

    def geodesic_distance(
        self,
        config1: tuple[Rotation, Rotation],
        config2: tuple[Rotation, Rotation],
    ) -> float:
        """Compute geodesic distance between two configurations on SO(3) x SO(3).

        TODO: Implement geodesic distance.
        Hint: Distance on SO(3) x SO(3) is sqrt(d1^2 + d2^2) where d1, d2
        are geodesic distances on each SO(3) factor.

        Args:
            config1: First configuration.
            config2: Second configuration.

        Returns:
            Geodesic distance.
        """
        d1 = (config1[0].inv() * config2[0]).magnitude()
        d2 = (config1[1].inv() * config2[1]).magnitude()

        return float(np.sqrt(d1**2 + d2**2))

    def sample_random_configuration(
        self, rng: np.random.Generator | None = None
    ) -> tuple[Rotation, Rotation]:
        """Sample a random configuration uniformly on SO(3) x SO(3).

        Args:
            rng: Random number generator.

        Returns:
            Random orientations for both joints.
        """
        if rng is None:
            rng = np.random.default_rng()

        R1 = Rotation.random(random_state=rng.integers(0, 2**31))
        R2 = Rotation.random(random_state=rng.integers(0, 2**31))

        return (R1, R2)
