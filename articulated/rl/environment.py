"""Environment wrappers for RL experiments.

Tasks:
1. Set up Reacher-v5 environment
2. Create wrapper that integrates embeddings from Team Estimation
3. Provide both raw and embedded observation modes
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch


class ReacherWithEmbedding(gym.Wrapper):
    """Wrapper for Reacher-v5 that can use learned embeddings.

    This wrapper allows switching between:
    - Raw observations (default Reacher-v5)
    - Embedded observations (using Team Estimation's model)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        use_embedding: bool = False,
        observation_mode: Optional[str] = None,
        history_length: int = 10,
        render_mode: Optional[str] = None,
    ):
        """Initialize the environment wrapper.

        Args:
            embedding_model: Trained StateEstimationModel from Team Estimation.
            use_embedding: Whether to use embeddings or raw observations.
            observation_mode: One of {"raw", "embed_only", "concat"}.
                If None, falls back to "embed_only" when use_embedding=True,
                otherwise "raw".
            history_length: Number of timesteps of history for embedding.
            render_mode: Gymnasium render mode.
        """
        env = gym.make("Reacher-v5", render_mode=render_mode)
        super().__init__(env)

        self.embedding_model = embedding_model
        self.observation_mode = self._resolve_observation_mode(
            observation_mode=observation_mode,
            use_embedding=use_embedding,
        )
        self.use_embedding = self.observation_mode != "raw"
        self.history_length = history_length

        # Observation history buffer
        self._obs_history: list[np.ndarray] = []
        # Per-episode initial position encoding for conditioned embeddings
        self._init_pos_so2: Optional[np.ndarray] = None

        # Update observation space for embedded/concatenated modes
        if self.use_embedding:
            if embedding_model is None:
                raise ValueError(
                    "embedding_model is required when observation_mode is "
                    "'embed_only' or 'concat'."
                )
            embedding_dim = embedding_model.hidden_size
            raw_obs_dim = int(np.prod(self.env.observation_space.shape))
            if self.observation_mode == "embed_only":
                obs_dim = embedding_dim
            elif self.observation_mode == "concat":
                obs_dim = raw_obs_dim + embedding_dim
            else:
                raise ValueError(
                    f"Unsupported observation_mode: {self.observation_mode}"
                )
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

    @staticmethod
    def _resolve_observation_mode(
        observation_mode: Optional[str], use_embedding: bool
    ) -> str:
        """Resolve observation mode while preserving legacy use_embedding behavior."""
        if observation_mode is None:
            return "embed_only" if use_embedding else "raw"

        mode = observation_mode.lower()
        valid_modes = {"raw", "embed_only", "concat"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown observation_mode '{observation_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )
        return mode

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_history = []
        self._init_pos_so2 = self._extract_init_pos_so2(obs)

        if self.observation_mode != "raw":
            obs = self._compose_observation(obs)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.observation_mode != "raw":
            obs = self._compose_observation(obs)

        return obs, reward, terminated, truncated, info

    def _compose_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Create embedded or concatenated observation from raw Reacher obs."""
        embedding = self._get_embedded_observation(raw_obs)
        if self.observation_mode == "embed_only":
            return embedding

        raw_obs = np.asarray(raw_obs, dtype=np.float32)
        if self.observation_mode == "concat":
            return np.concatenate([raw_obs, embedding], axis=0).astype(
                np.float32, copy=False
            )

        raise ValueError(f"Unsupported observation_mode: {self.observation_mode}")

    def _get_embedded_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Convert raw observation to embedding.

        TODO: Implement this method:
        1. Extract angular velocities from raw observation
        2. Maintain history buffer
        3. Pass through embedding model
        4. Return embedding

        Args:
            raw_obs: Raw observation from Reacher-v5.

        Returns:
            Embedded observation.
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model is required when use_embedding is True.")

        angular_velocities = self._extract_angular_velocities(raw_obs)
        angular_velocities = angular_velocities.astype(np.float32, copy=False)
        self._obs_history.append(angular_velocities)

        if len(self._obs_history) > self.history_length:
            self._obs_history = self._obs_history[-self.history_length :]

        vel_dim = angular_velocities.shape[0]
        if len(self._obs_history) < self.history_length:
            pad = np.zeros(
                (self.history_length - len(self._obs_history), vel_dim),
                dtype=np.float32,
            )
            history = np.vstack([pad, np.stack(self._obs_history)])
        else:
            history = np.stack(self._obs_history)

        history_tensor = torch.from_numpy(history).unsqueeze(0)
        init_tensor = torch.from_numpy(self._init_pos_so2).unsqueeze(0)
        if hasattr(self.embedding_model, "parameters"):
            try:
                device = next(self.embedding_model.parameters()).device
                history_tensor = history_tensor.to(device)
                init_tensor = init_tensor.to(device)
            except StopIteration:
                pass

        with torch.no_grad():
            embedding = self.embedding_model.get_embedding(
                history_tensor,
                init_pos=init_tensor,
            )

        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def _extract_angular_velocities(self, obs: np.ndarray) -> np.ndarray:
        """Extract angular velocities from Reacher observation.

        TODO: Implement based on Reacher-v5 observation structure.
        See: https://gymnasium.farama.org/environments/mujoco/reacher/

        Args:
            obs: Raw observation.

        Returns:
            Angular velocities.
        """
        if obs.shape[0] < 8:
            raise ValueError(
                "Expected Reacher-v5 observation with at least 8 elements."
            )
        return np.asarray(obs[6:8], dtype=np.float32)

    def _extract_init_pos_so2(self, obs: np.ndarray) -> np.ndarray:
        """Extract SO(2) init encoding from Reacher observation.

        Reacher-v5 observation starts with:
            [cos(theta1), cos(theta2), sin(theta1), sin(theta2), ...]
        while the SO(2) estimator expects:
            [cos(theta1), sin(theta1), cos(theta2), sin(theta2)].
        """
        if obs.shape[0] < 4:
            raise ValueError(
                "Expected Reacher-v5 observation with at least 4 elements."
            )
        cos_t1, cos_t2, sin_t1, sin_t2 = obs[:4]
        return np.asarray([cos_t1, sin_t1, cos_t2, sin_t2], dtype=np.float32)
