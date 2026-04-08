"""
test.py -- Evaluation / Demo Script
=====================================
Loads the saved PPO agent (laptop_thermal_agent.zip) and runs it
through 5 complete episodes of GamingLaptopEnv.

At each step it prints:
  - The action taken by the agent
  - The resulting component temperatures
  - Current FPS and battery level
  - The step reward

At the end of each episode it prints a summary table.

Usage
-----
    python test.py [--episodes N] [--no-vecnorm]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from laptop_env import GamingLaptopEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH       = "laptop_thermal_agent"          # .zip is implicit
VECNORM_PATH     = "laptop_thermal_agent_vecnorm.pkl"
DEFAULT_EPISODES = 5
THROTTLE_TEMP    = 92.0


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"


def _temp_colour(temp: float) -> str:
    if temp >= THROTTLE_TEMP:
        return RED
    elif temp >= 85.0:
        return YELLOW
    return GREEN


def _fps_colour(fps: float) -> str:
    if fps >= 100.0:
        return GREEN
    elif fps >= 50.0:
        return YELLOW
    return RED


def _reward_colour(r: float) -> str:
    if r >= 0.5:
        return GREEN
    elif r >= 0.0:
        return CYAN
    return RED


def colour(text: str, code: str) -> str:
    """Wrap text in an ANSI escape code (no-op on unsupported terminals)."""
    try:
        if sys.stdout.isatty() or os.environ.get("FORCE_COLOUR"):
            return f"{code}{text}{RESET}"
    except Exception:
        pass
    return str(text)


def print_header(episode: int) -> None:
    print()
    print(colour("=" * 90, BOLD))
    print(colour(f"  EPISODE {episode}", BOLD + CYAN))
    print(colour("=" * 90, BOLD))
    header = (
        f"{'Step':>5s}  "
        f"{'Fan':>5s}  "
        f"{'CpuPwr':>6s}  "
        f"{'GpuPwr':>6s}  "
        f"{'NpuPwr':>6s}  "
        f"{'AIoff':>5s}  "
        f"{'CPU_C':>6s}  "
        f"{'GPU_C':>6s}  "
        f"{'NPU_C':>6s}  "
        f"{'FPS':>7s}  "
        f"{'Batt%':>5s}  "
        f"{'Reward':>9s}"
    )
    print(colour(header, BOLD))
    print(colour("-" * 90, BOLD))


def print_step(
    step: int,
    action: np.ndarray,
    cpu_t: float,
    gpu_t: float,
    npu_t: float,
    fps: float,
    battery: float,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> None:
    fan_s, cpu_p, gpu_p, npu_p, ai_off = action

    # Throttle / done marker
    marker = ""
    if terminated:
        marker = colour(" [!!] TERMINATED", RED + BOLD)
    elif truncated:
        marker = colour(" [OK] TRUNCATED", GREEN)

    row = (
        f"{step:>5d}  "
        f"{fan_s:>5.2f}  "
        f"{cpu_p:>6.2f}  "
        f"{gpu_p:>6.2f}  "
        f"{npu_p:>6.2f}  "
        f"{ai_off:>5.2f}  "
        f"{colour(f'{cpu_t:6.1f}', _temp_colour(cpu_t))}  "
        f"{colour(f'{gpu_t:6.1f}', _temp_colour(gpu_t))}  "
        f"{colour(f'{npu_t:6.1f}', _temp_colour(npu_t))}  "
        f"{colour(f'{fps:7.1f}', _fps_colour(fps))}  "
        f"{battery:5.1f}  "
        f"{colour(f'{reward:+9.3f}', _reward_colour(reward))}"
        f"{marker}"
    )
    print(row)


def print_episode_summary(
    ep: int,
    steps: int,
    total_reward: float,
    peak_cpu: float,
    peak_gpu: float,
    peak_npu: float,
    avg_fps: float,
    final_battery: float,
    throttle_steps: int,
) -> None:
    print(colour("-" * 90, BOLD))
    print(colour(f"  Episode {ep} Summary", BOLD))
    print(f"    Total steps       : {steps}")
    print(f"    Cumulative reward : {colour(f'{total_reward:+.2f}', _reward_colour(total_reward / max(steps, 1)))}")
    print(f"    Avg reward/step   : {colour(f'{total_reward / max(steps, 1):+.3f}', _reward_colour(total_reward / max(steps, 1)))}")
    print(f"    Peak CPU temp     : {colour(f'{peak_cpu:.1f} C', _temp_colour(peak_cpu))}")
    print(f"    Peak GPU temp     : {colour(f'{peak_gpu:.1f} C', _temp_colour(peak_gpu))}")
    print(f"    Peak NPU temp     : {colour(f'{peak_npu:.1f} C', _temp_colour(peak_npu))}")
    print(f"    Avg FPS           : {colour(f'{avg_fps:.1f}', _fps_colour(avg_fps))}")
    print(f"    Final battery     : {final_battery:.1f}%")
    print(f"    Throttle events   : {colour(str(throttle_steps), RED if throttle_steps else GREEN)}")


def print_grand_summary(results: list[dict]) -> None:
    print()
    print(colour("=" * 90, BOLD))
    print(colour("  GRAND SUMMARY -- All Episodes", BOLD + MAGENTA))
    print(colour("=" * 90, BOLD))

    header = (
        f"{'Ep':>3s}  "
        f"{'Steps':>5s}  "
        f"{'TotalRew':>10s}  "
        f"{'Avg Rew':>9s}  "
        f"{'PkCPU':>6s}  "
        f"{'PkGPU':>6s}  "
        f"{'PkNPU':>6s}  "
        f"{'AvgFPS':>7s}  "
        f"{'Batt%':>5s}  "
        f"{'Throttle':>8s}"
    )
    print(colour(header, BOLD))
    print(colour("-" * 90, BOLD))

    for r in results:
        avg_rew      = r["total_reward"] / max(r["steps"], 1)
        ep_val       = r["episode"]
        steps_val    = r["steps"]
        total_r      = r["total_reward"]
        pk_cpu       = r["peak_cpu"]
        pk_gpu       = r["peak_gpu"]
        pk_npu       = r["peak_npu"]
        avg_fps_val  = r["avg_fps"]
        batt_val     = r["final_battery"]
        thr_val      = r["throttle_steps"]
        print(
            f"{ep_val:>3d}  "
            f"{steps_val:>5d}  "
            f"{colour(f'{total_r:>+10.2f}', _reward_colour(avg_rew))}  "
            f"{colour(f'{avg_rew:>+9.3f}', _reward_colour(avg_rew))}  "
            f"{colour(f'{pk_cpu:>6.1f}', _temp_colour(pk_cpu))}  "
            f"{colour(f'{pk_gpu:>6.1f}', _temp_colour(pk_gpu))}  "
            f"{colour(f'{pk_npu:>6.1f}', _temp_colour(pk_npu))}  "
            f"{colour(f'{avg_fps_val:>7.1f}', _fps_colour(avg_fps_val))}  "
            f"{batt_val:>5.1f}  "
            f"{colour(str(thr_val), RED if thr_val else GREEN):>8s}"
        )

    print(colour("=" * 90, BOLD))

    # Overall stats
    all_rew   = [r["total_reward"] for r in results]
    all_fps   = [r["avg_fps"]      for r in results]
    all_thr   = [r["throttle_steps"] for r in results]
    print(f"\n  Overall mean episode reward : {np.mean(all_rew):+.2f} +/- {np.std(all_rew):.2f}")
    print(f"  Overall mean episode FPS    : {np.mean(all_fps):.1f}")
    print(f"  Total throttle events       : {sum(all_thr)}")
    print()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def load_model(use_vecnorm: bool) -> tuple[PPO, object]:
    """
    Load the PPO model. If VecNormalize stats file is present and
    `use_vecnorm` is True, wrap the raw env with VecNormalize for
    consistent observation scaling.
    """
    if not Path(f"{MODEL_PATH}.zip").exists():
        sys.exit(
            f"\n  ERROR: model file '{MODEL_PATH}.zip' not found.\n"
            "  Run train.py first to generate it.\n"
        )

    print(f"  Loading model from '{MODEL_PATH}.zip' ...")

    if use_vecnorm and Path(VECNORM_PATH).exists():
        print(f"  Loading VecNormalize stats from '{VECNORM_PATH}' ...")
        raw_env = DummyVecEnv([lambda: GamingLaptopEnv()])
        vec_env = VecNormalize.load(VECNORM_PATH, raw_env)
        vec_env.training   = False   # freeze norm stats during eval
        vec_env.norm_reward = False  # report raw rewards
        model = PPO.load(MODEL_PATH, env=vec_env)
        return model, vec_env
    else:
        if use_vecnorm:
            print(f"  VecNorm file not found -- using raw environment.")
        # Use a plain (non-vectorised) env for direct gym interaction
        model = PPO.load(MODEL_PATH)
        env   = GamingLaptopEnv()
        return model, env


def run_vec_episode(model: PPO, vec_env, ep: int) -> dict:
    """Run one episode through a VecNormalize-wrapped environment."""
    obs    = vec_env.reset()
    print_header(ep)

    step          = 0
    total_reward  = 0.0
    peak_cpu = peak_gpu = peak_npu = 0.0
    fps_acc       = 0.0
    throttle_steps = 0
    final_battery = 100.0

    while True:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_arr, done_arr, info_arr = vec_env.step(action)

        info        = info_arr[0]
        reward_val  = float(reward_arr[0])
        done        = bool(done_arr[0])
        terminated  = done  # VecEnv auto-resets; treat done as terminated
        truncated   = False

        cpu_t   = info.get("cpu_temp", 0.0)
        gpu_t   = info.get("gpu_temp", 0.0)
        npu_t   = info.get("npu_temp", 0.0)
        fps_val = info.get("fps", 0.0)
        batt    = info.get("battery_pct", 0.0)

        total_reward  += reward_val
        peak_cpu       = max(peak_cpu, cpu_t)
        peak_gpu       = max(peak_gpu, gpu_t)
        peak_npu       = max(peak_npu, npu_t)
        fps_acc       += fps_val
        final_battery  = batt
        if max(cpu_t, gpu_t, npu_t) >= THROTTLE_TEMP:
            throttle_steps += 1

        act_disp = action[0]  # shape (1, 5) -> first env
        print_step(step, act_disp, cpu_t, gpu_t, npu_t, fps_val, batt,
                   reward_val, terminated, truncated)

        if done:
            break

    avg_fps = fps_acc / max(step, 1)
    print_episode_summary(ep, step, total_reward, peak_cpu, peak_gpu, peak_npu,
                          avg_fps, final_battery, throttle_steps)
    return {
        "episode": ep, "steps": step, "total_reward": total_reward,
        "peak_cpu": peak_cpu, "peak_gpu": peak_gpu, "peak_npu": peak_npu,
        "avg_fps": avg_fps, "final_battery": final_battery,
        "throttle_steps": throttle_steps,
    }


def run_raw_episode(model: PPO, env: GamingLaptopEnv, ep: int) -> dict:
    """Run one episode through a plain (non-vectorised) environment."""
    obs, _   = env.reset()
    print_header(ep)

    step          = 0
    total_reward  = 0.0
    peak_cpu = peak_gpu = peak_npu = 0.0
    fps_acc       = 0.0
    throttle_steps = 0
    final_battery = 100.0

    while True:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_val, terminated, truncated, info = env.step(action)

        cpu_t   = info.get("cpu_temp", 0.0)
        gpu_t   = info.get("gpu_temp", 0.0)
        npu_t   = info.get("npu_temp", 0.0)
        fps_val = info.get("fps", 0.0)
        batt    = info.get("battery_pct", 0.0)

        total_reward  += float(reward_val)
        peak_cpu       = max(peak_cpu, cpu_t)
        peak_gpu       = max(peak_gpu, gpu_t)
        peak_npu       = max(peak_npu, npu_t)
        fps_acc       += fps_val
        final_battery  = batt
        if max(cpu_t, gpu_t, npu_t) >= THROTTLE_TEMP:
            throttle_steps += 1

        print_step(step, action, cpu_t, gpu_t, npu_t, fps_val, batt,
                   float(reward_val), terminated, truncated)

        if terminated or truncated:
            break

    avg_fps = fps_acc / max(step, 1)
    print_episode_summary(ep, step, total_reward, peak_cpu, peak_gpu, peak_npu,
                          avg_fps, final_battery, throttle_steps)
    return {
        "episode": ep, "steps": step, "total_reward": total_reward,
        "peak_cpu": peak_cpu, "peak_gpu": peak_gpu, "peak_npu": peak_npu,
        "avg_fps": avg_fps, "final_battery": final_battery,
        "throttle_steps": throttle_steps,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained PPO agent")
    parser.add_argument("--episodes",  type=int,  default=DEFAULT_EPISODES,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--no-vecnorm", action="store_true",
                        help="Skip VecNormalize wrapping even if stats file exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print(colour("=" * 65, BOLD))
    print(colour("  AI Gaming Laptop Thermal Manager -- Agent Evaluation", BOLD + CYAN))
    print(colour("=" * 65, BOLD))

    use_vecnorm = not args.no_vecnorm
    model, env_or_vec = load_model(use_vecnorm=use_vecnorm)

    is_vec = isinstance(env_or_vec, VecNormalize)
    print(f"  Evaluation mode : {'VecNormalize' if is_vec else 'Raw Gym Env'}")
    print(f"  Episodes        : {args.episodes}")
    print()

    results = []
    for ep in range(1, args.episodes + 1):
        if is_vec:
            result = run_vec_episode(model, env_or_vec, ep)
        else:
            result = run_raw_episode(model, env_or_vec, ep)
        results.append(result)

    if hasattr(env_or_vec, "close"):
        env_or_vec.close()

    print_grand_summary(results)


if __name__ == "__main__":
    main()
