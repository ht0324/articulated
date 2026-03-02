"""RL agents for Reacher benchmark.

Tasks:
1. Set up RL training with stable-baselines3
2. Implement training loop
3. Compare raw vs embedded observations
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from articulated.rl.environment import ReacherWithEmbedding


class RLAgent:
    """RL agent for Reacher benchmark.

    Supports training with:
    - Raw observations (baseline)
    - Embedded observations (from Team Estimation)
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        use_embedding: bool = False,
        observation_mode: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        clip_range: float = 0.2,
        policy: str = "MlpPolicy",
        device: str = "auto",
        tensorboard_log: Optional[str] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        clip_obs: float = 10.0,
        history_length: int = 10,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the RL agent.

        Args:
            algorithm: RL algorithm ('ppo' or 'sac').
            use_embedding: Legacy flag controlling observation source.
                Kept for backward compatibility; ignored when
                observation_mode is provided.
            observation_mode: Observation mode for policy input:
                {"raw", "embed_only", "concat"}.
            embedding_model_path: Path to Team Estimation's checkpoint.
            learning_rate: Learning rate.
            n_steps: Number of steps per rollout (PPO only).
            batch_size: Mini-batch size (PPO/SAC).
            clip_range: PPO clipping range.
            policy: Policy network type for SB3.
            device: Torch device for SB3 ("cpu", "cuda", "auto").
            tensorboard_log: Optional TensorBoard log directory.
            normalize_observations: Whether to normalize observations (VecNormalize).
            normalize_rewards: Whether to normalize rewards (VecNormalize).
            clip_obs: Observation clipping range for VecNormalize.
            history_length: Timesteps of velocity history for embeddings.
            render_mode: Optional render mode for the env.
            seed: Random seed.
        """
        self.algorithm = algorithm.lower()
        self.observation_mode = self._resolve_observation_mode(
            observation_mode=observation_mode,
            use_embedding=use_embedding,
        )
        self.use_embedding = self.observation_mode != "raw"
        self.embedding_model_path = embedding_model_path
        self.learning_rate = (
            float(learning_rate) if isinstance(learning_rate, str) else learning_rate
        )
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.policy = policy
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.clip_obs = clip_obs
        self.history_length = history_length
        self.render_mode = render_mode
        self.seed = seed

        self.env: Optional[gym.Env | VecEnv] = None
        self.model: Optional[PPO | SAC] = None
        self.embedding_model = None
        self.vec_normalize: Optional[VecNormalize] = None

    @staticmethod
    def _resolve_observation_mode(
        observation_mode: Optional[str], use_embedding: bool
    ) -> str:
        """Resolve observation mode while preserving legacy configs."""
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

    def setup(self) -> None:
        """Set up the environment and model.

        Loads the embedding model if needed, creates the env, and initializes
        the SB3 algorithm.
        """
        if self.use_embedding:
            if not self.embedding_model_path:
                raise ValueError(
                    "embedding_model_path is required when use_embedding is True."
                )
            from articulated.estimation.model import StateEstimationModel

            self.embedding_model = StateEstimationModel.load_for_embedding(
                self.embedding_model_path
            )

        self.env, self.vec_normalize = self._make_env(
            render_mode=self.render_mode, training=True
        )

        if self.seed is not None:
            if isinstance(self.env, VecEnv):
                self.env.seed(self.seed)
            else:
                self.env.reset(seed=self.seed)
                if hasattr(self.env.action_space, "seed"):
                    self.env.action_space.seed(self.seed)

        if self.algorithm == "ppo":
            if self.n_steps % self.batch_size != 0:
                raise ValueError(
                    "For PPO, n_steps must be divisible by batch_size."
                )
            self.model = PPO(
                self.policy,
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                clip_range=self.clip_range,
                seed=self.seed,
                device=self.device,
                tensorboard_log=self.tensorboard_log,
                verbose=1,
            )
        elif self.algorithm == "sac":
            self.model = SAC(
                self.policy,
                self.env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                seed=self.seed,
                device=self.device,
                tensorboard_log=self.tensorboard_log,
                verbose=1,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_freq: int = 10_000,
        save_path: Optional[str] = None,
    ) -> dict:
        """Train the RL agent.

        TODO: Implement training loop:
        1. Set up evaluation callback
        2. Call model.learn()
        3. Save trained model

        Args:
            total_timesteps: Total training timesteps.
            eval_freq: Evaluation frequency.
            save_path: Path to save the trained model.

        Returns:
            Training statistics.
        """
        if self.model is None or self.env is None:
            raise RuntimeError("Call setup() before train().")

        eval_env = None
        eval_vec_norm = None
        callback = None
        if eval_freq and eval_freq > 0:
            eval_env, eval_vec_norm = self._make_env(render_mode=None, training=False)
            self._sync_vec_normalize(eval_vec_norm)
            callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=1,
            )

        tb_log_name = f"{self.algorithm}_{self.observation_mode}"
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
        )

        if save_path:
            self.model.save(save_path)
            if self.vec_normalize is not None:
                self.vec_normalize.save(f"{save_path}_vecnormalize.pkl")

        if eval_env is not None:
            eval_env.close()

        return {
            "total_timesteps": total_timesteps,
        }

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate the trained agent.

        TODO: Implement evaluation:
        1. Run n_episodes with deterministic policy
        2. Compute mean reward and episode length

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Call setup() and train() before evaluate().")

        eval_env, eval_vec_norm = self._make_env(render_mode=None, training=False)
        self._sync_vec_normalize(eval_vec_norm)
        rewards, lengths = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        eval_env.close()

        rewards_arr = np.asarray(rewards, dtype=np.float32)
        lengths_arr = np.asarray(lengths, dtype=np.float32)

        return {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "mean_length": float(lengths_arr.mean()),
            "std_length": float(lengths_arr.std()),
        }

    def _make_env(
        self, render_mode: Optional[str], training: bool
    ) -> tuple[gym.Env | VecEnv, Optional[VecNormalize]]:
        """Create a Reacher-v5 env, optionally wrapped/normalized."""

        def make_env() -> gym.Env:
            if self.observation_mode != "raw":
                return ReacherWithEmbedding(
                    embedding_model=self.embedding_model,
                    use_embedding=True,
                    observation_mode=self.observation_mode,
                    history_length=self.history_length,
                    render_mode=render_mode,
                )
            return gym.make("Reacher-v5", render_mode=render_mode)

        if self.normalize_observations or self.normalize_rewards:
            vec_env = DummyVecEnv([make_env])
            if self.seed is not None:
                vec_env.seed(self.seed)
            vec_norm = VecNormalize(
                vec_env,
                norm_obs=self.normalize_observations,
                norm_reward=self.normalize_rewards,
                clip_obs=self.clip_obs,
                training=training,
            )
            if not training:
                vec_norm.training = False
                vec_norm.norm_reward = False
            return VecMonitor(vec_norm), vec_norm

        env = Monitor(make_env())
        return env, None

    def _sync_vec_normalize(self, eval_vec_norm: Optional[VecNormalize]) -> None:
        """Sync normalization stats from training env to eval env."""
        if self.vec_normalize is None or eval_vec_norm is None:
            return
        eval_vec_norm.obs_rms = self.vec_normalize.obs_rms
        if hasattr(self.vec_normalize, "ret_rms"):
            eval_vec_norm.ret_rms = self.vec_normalize.ret_rms
        eval_vec_norm.training = False
        eval_vec_norm.norm_reward = False
