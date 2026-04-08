"""
train.py — PPO Training Script
================================
Trains a Stable-Baselines3 PPO agent on GamingLaptopEnv for 100,000
timesteps and saves the policy as laptop_thermal_agent.zip.

Usage
-----
    python train.py [--timesteps N] [--seed S] [--log-dir PATH]

Requires
--------
    pip install gymnasium stable-baselines3[extra] torch numpy
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor

from laptop_env import GamingLaptopEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TIMESTEPS = 100_000
DEFAULT_SEED      = 42
MODEL_SAVE_PATH   = "laptop_thermal_agent"      # .zip appended automatically
LOG_DIR           = "./ppo_laptop_logs"
TENSORBOARD_DIR   = "./ppo_tensorboard"
N_ENVS            = 4             # Parallel environments for faster sampling
EVAL_FREQ         = 10_000        # Evaluate every N steps (per env)
N_EVAL_EPISODES   = 5


# ---------------------------------------------------------------------------
# Custom Callback: live console progress
# ---------------------------------------------------------------------------

class TrainingProgressCallback(BaseCallback):
    """Prints a one-line summary every `log_every` calls to `on_step`."""

    def __init__(self, log_every: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every   = log_every
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every == 0:
            elapsed = time.time() - self._start_time
            fps     = self.num_timesteps / max(elapsed, 1e-6)
            print(
                f"  [Train] timestep {self.num_timesteps:>7,d}  |  "
                f"elapsed {elapsed:6.1f}s  |  ~{fps:,.0f} steps/s"
            )
        return True  # continue training


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on GamingLaptopEnv"
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps (default: 100,000)")
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED,
                        help="Global random seed")
    parser.add_argument("--log-dir",   type=str, default=LOG_DIR,
                        help="Directory for Monitor CSV logs")
    parser.add_argument("--no-tb",     action="store_true",
                        help="Disable TensorBoard logging")
    return parser.parse_args()


def make_env(seed: int = 0, rank: int = 0):
    """Factory for a single monitored environment instance."""
    def _init():
        env = GamingLaptopEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args: argparse.Namespace) -> None:
    print("=" * 65)
    print("  AI-Enhanced Gaming Laptop Power & Thermal Manager")
    print("  Meta PyTorch OpenEnv Hackathon — PPO Training")
    print("=" * 65)

    # ── Device detection ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  PyTorch device : {device}")
    if device == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Parallel envs  : {N_ENVS}")
    print(f"  Total timesteps: {args.timesteps:,}")
    print()

    # ── Vectorised environment ─────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    vec_env = make_vec_env(
        lambda: GamingLaptopEnv(),
        n_envs=N_ENVS,
        seed=args.seed,
    )
    # Normalise observations & rewards for better PPO stability
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Separate evaluation environment (un-normalised for real reward values)
    eval_env = make_vec_env(
        lambda: GamingLaptopEnv(),
        n_envs=1,
        seed=args.seed + 999,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── PPO Hyperparameters ───────────────────────────────────────────
    # Tuned for a continuous-action, short-episode laptop control task.
    ppo_kwargs = dict(
        policy          = "MlpPolicy",
        env             = vec_env,
        device          = device,
        seed            = args.seed,
        verbose         = 0,                  # suppress SB3's own prints
        tensorboard_log = None if args.no_tb else TENSORBOARD_DIR,

        # Rollout & optimisation
        n_steps         = 512,               # Steps per env before update
        batch_size      = 128,               # Mini-batch size
        n_epochs        = 10,                # PPO update epochs per rollout
        gamma           = 0.995,             # High discount → care about future
        gae_lambda      = 0.97,              # GAE smoothing
        clip_range      = 0.20,              # PPO clip ε
        ent_coef        = 0.005,             # Light entropy bonus
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        learning_rate   = 3e-4,

        # Policy network: [256, 256] for both actor and critic
        policy_kwargs   = dict(
            net_arch    = dict(pi=[256, 256], vf=[256, 256]),
            activation_fn = torch.nn.ReLU,
        ),
    )

    model = PPO(**ppo_kwargs)

    print("  Policy architecture:")
    print(f"    Actor  : {ppo_kwargs['policy_kwargs']['net_arch']['pi']}")
    print(f"    Critic : {ppo_kwargs['policy_kwargs']['net_arch']['vf']}")
    print()

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(EVAL_FREQ // N_ENVS, 1),
        save_path   = args.log_dir,
        name_prefix = "ppo_laptop_ckpt",
        verbose     = 0,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = args.log_dir,
        log_path             = args.log_dir,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    progress_cb = TrainingProgressCallback(log_every=10_000)

    # ── Train ─────────────────────────────────────────────────────────
    print(f"  Training for {args.timesteps:,} timesteps …\n")
    t0 = time.time()

    model.learn(
        total_timesteps = args.timesteps,
        callback        = [checkpoint_cb, eval_cb, progress_cb],
        progress_bar    = False,
        reset_num_timesteps = True,
    )

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s  "
          f"({args.timesteps / elapsed:,.0f} steps/s)\n")

    # ── Save model ────────────────────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    # Save the VecNormalize statistics alongside the model so test.py can
    # restore normalisation without requiring the training environment.
    vec_env.save(f"{MODEL_SAVE_PATH}_vecnorm.pkl")

    print(f"  Model saved  -> {MODEL_SAVE_PATH}.zip")
    print(f"  VecNorm stats-> {MODEL_SAVE_PATH}_vecnorm.pkl")

    if not args.no_tb:
        print(f"  TensorBoard  -> tensorboard --logdir {TENSORBOARD_DIR}")

    print("\n  Done.")
    vec_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
