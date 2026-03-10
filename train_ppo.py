"""
train_ppo.py

Train a PPO agent on JetpackEnv (object-vector observations).

Install:
  pip install stable-baselines3 gymnasium numpy

Run:
  python train_ppo.py

Outputs:
  models/jetpack_ppo.zip
  TensorBoard logs: runs/jetpack_ppo/

View logs:
  tensorboard --logdir runs
"""

import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from jetpack_env import JetpackEnv


def make_env(rank: int, seed: int = 0):
    """Factory for vectorized environments."""
    def _init():
        env = JetpackEnv(k_threats=3, max_steps=60 * 60, seed=seed + rank)
        return Monitor(env)
    return _init


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # PPO learns better with multiple environments (more diverse experience).
    n_envs = 8
    env = SubprocVecEnv([make_env(i, seed=123) for i in range(n_envs)])

    # Choose device (GPU/CPU/MPS) based on availability.
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # IF RETRAINING: UNCOMMENT THIS AND COMMENT OUT THE APPOPRIATE LINES BELOW
    # model = PPO.load("models/jetpack_ppo.zip", env=env)
    # model.learn(total_timesteps=1_000_000)
    # model.save("models/jetpack_ppo")  # overwrites, but it's the updated model

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="runs/jetpack_ppo_v1",
        # Good starting hyperparameters for survival-style tasks:
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=100,
        device=device,
    )

    # Start here; increase if you want a stronger agent.
    total_timesteps = 3_000_000

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("models/jetpack_ppo_v1")

    env.close()
    print("Saved model to models/jetpack_ppo_v1.zip")


if __name__ == "__main__":
    main()