import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from jetpack_env import JetpackEnv


def make_env(rank: int, seed: int = 0):
    def _init():
        env = JetpackEnv(k_threats=3, seed=seed + rank)
        return Monitor(env)
    return _init


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    n_envs = 8
    env = SubprocVecEnv([make_env(i, seed=123) for i in range(n_envs)])

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Training on device: {device}")

    # ── To continue training an existing model ──────────────────────────────
    # model = PPO.load("models/jetpack_ppo_v3.zip", env=env, device=device)
    # model.learn(total_timesteps=50_000_000, reset_num_timesteps=False, progress_bar=True)
    # model.save("models/jetpack_ppo_v3")
    # return
    # ────────────────────────────────────────────────────────────────────────

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="runs/jetpack_ppo_v3",

        n_steps=2048,
        batch_size=512,
        n_epochs=4,

        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,

        ent_coef=0.02,


        learning_rate=3e-4,

        device=device,
    )

    total_timesteps = 80_000_000

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("models/jetpack_ppo_v3")

    env.close()
    print("Saved model to models/jetpack_ppo_v3.zip")


if __name__ == "__main__":
    main()