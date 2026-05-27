"""
Train a neural PPO agent for UAV selection.

Usage:
    source venv/bin/activate
    pip install -r requirements.txt
    python train_ppo.py
    python train_ppo.py --timesteps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import network_config as net
from uav_iov_env import UAVIoVEnv

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ppo_uav_iov"
LOG_DIR = Path("logs/ppo_uav_iov")


def make_env():
    return Monitor(UAVIoVEnv())


def train(total_timesteps: int = 200_000, n_envs: int = 4) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=0)
    eval_env = Monitor(UAVIoVEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    print(
        f"\nTraining PPO ({net.NUM_UAVS} UAVs, obs dim "
        f"{UAVIoVEnv().observation_space.shape[0]}) "
        f"for {total_timesteps:,} timesteps ({n_envs} parallel envs)...\n"
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(str(MODEL_PATH))
    vec_env.close()
    eval_env.close()

    print(f"\nModel saved to {MODEL_PATH}.zip")
    print("Next: python evaluate_ppo.py")
    return MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train PPO on UAV-IoV environment")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total environment steps for training (default: 200000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, n_envs=args.n_envs)


if __name__ == "__main__":
    main()
