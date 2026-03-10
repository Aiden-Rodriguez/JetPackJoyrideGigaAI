"""
play_ppo.py

Run a trained PPO agent in the headless Gym env.

Run:
  python play_ppo.py

Note:
  This does not render. The fastest way to *see* the learned behavior is to
  add a small renderer to JetpackEnv or use a separate evaluation script that
  drives the pygame runner (jetpack.py) with model actions.
"""

from stable_baselines3 import PPO
from jetpack_env import JetpackEnv


def main():
    env = JetpackEnv(k_threats=3, max_steps=60 * 60, seed=999)
    model = PPO.load("models/jetpack_ppo")

    obs, info = env.reset()
    ep_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            print(f"Episode done. score={info.get('score', 0):.1f} reward={ep_reward:.1f}")
            obs, info = env.reset()
            ep_reward = 0.0


if __name__ == "__main__":
    main()