"""
demo.py — Oracle Demo Agent (no API key required)
===================================================
Runs the full OpenEnv inference pipeline on all 3 tasks using a
hard-coded oracle that always plays the optimal action sequence.

This proves the environment, grader, and STDOUT spec all work correctly
without requiring an OpenAI / HuggingFace API key.

Output format matches the official OpenEnv spec exactly:
    [START] task=... env=... model=...
    [STEP]  step=... action=... reward=... done=... error=...
    [END]   success=... steps=... score=... rewards=...

Usage
-----
    python demo.py
    python demo.py --task easy_fan_fix
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Optional

from laptop_env import Action, ServerSysadminEnv, MAX_STEPS

BENCHMARK  = "server-thermal-sysadmin"
MODEL_NAME = "oracle-v1 (no-API demo)"

# ── Optimal action sequences for every task ───────────────────────────────────

ORACLE: dict[str, list[dict]] = {
    "easy_fan_fix": [
        {"command": "set_fan_profile", "target": "rack-1", "value": "high"},
        {"command": "diagnose",        "target": "rack-1", "value": None},
        {"command": "set_fan_profile", "target": "rack-1", "value": "high"},
    ],
    "medium_rogue_process": [
        {"command": "list_processes",  "target": "rack-2", "value": None},
        {"command": "kill_process",    "target": "rack-2", "value": "9841"},
        {"command": "verify_thermals", "target": "rack-2", "value": None},
    ],
    "hard_db_migration": [
        {"command": "check_db_status",    "target": "rack-3", "value": None},
        {"command": "initiate_migration", "target": "rack-3", "value": "rack-4"},
        {"command": "verify_migration",   "target": "rack-4", "value": None},
        {"command": "shutdown_node",      "target": "rack-3", "value": None},
    ],
}

ALL_TASKS = list(ORACLE.keys())

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ── OpenEnv STDOUT helpers ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Pretty section helpers ────────────────────────────────────────────────────

def _section(label: str) -> None:
    bar = "=" * 68
    print()
    print(_c(bar, BOLD))
    print(_c(f"  {label}", BOLD + CYAN))
    print(_c(bar, BOLD))


def _divider() -> None:
    print(_c("-" * 68, DIM))


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    """Run one episode using the oracle action sequence."""
    env = ServerSysadminEnv()
    obs = env.reset(task_id)

    oracle_actions = ORACLE[task_id]

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    _section(f"TASK : {task_id.upper().replace('_', ' ')}")

    # Print initial observation
    print()
    print(_c(f"  INIT", BOLD + CYAN))
    for line in obs.system_log.splitlines():
        print(_c(f"         {line}", DIM))
    print()
    _divider()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step_idx, action_data in enumerate(oracle_actions, start=1):
            action = Action(**action_data)
            time.sleep(0.05)   # tiny delay so output feels live

            obs, reward, done, info = env.step(action)

            reward_val  = float(reward.score)
            action_str  = json.dumps(action.model_dump(exclude_none=False))

            rewards.append(reward_val)
            steps_taken = step_idx

            # ── Pretty step output ────────────────────────────────────────
            reward_col = GREEN if reward_val >= 1.0 else (YELLOW if reward_val > 0 else RED)
            done_col   = GREEN if done else DIM
            print(
                _c(f"  STEP {step_idx}", BOLD)
                + f" | cmd={_c(action.command, CYAN)}"
                + f"  target={action.target}"
                + (f"  value={_c(action.value, MAGENTA)}" if action.value else "")
            )
            print(
                f"       reward={_c(f'{reward_val:.2f}', reward_col)}"
                + f"  done={_c(str(done).lower(), done_col)}"
            )
            for line in obs.system_log.splitlines():
                print(_c(f"         {line}", DIM))
            print()

            # ── Machine-readable spec line ────────────────────────────────
            log_step(step=step_idx, action=action_str,
                     reward=reward_val, done=done, error=None)

            if done:
                score   = reward_val
                success = bool(score >= 1.0)
                break

    finally:
        try:
            env.close()
        except Exception:
            pass

        _divider()
        sym = _c("RESOLVED [OK]", GREEN + BOLD) if success else _c("FAILED [!!]", RED + BOLD)
        print(f"  Episode result : {sym}")
        print(f"  Final score    : {_c(f'{score:.2f}', GREEN if success else RED)}")
        print(f"  Steps used     : {steps_taken} / {MAX_STEPS}")
        _divider()

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps":   steps_taken,
        "score":   score,
        "rewards": rewards,
    }


# ── Grand summary ─────────────────────────────────────────────────────────────

def print_grand_summary(results: list[dict]) -> None:
    mean_score = sum(r["score"] for r in results) / len(results)
    _section("GRAND SUMMARY — All Tasks")
    print()
    print(_c(f"  {'Task':<30s}  {'Score':>6s}  {'Steps':>5s}  Status", BOLD))
    print(_c("  " + "-" * 56, DIM))
    for r in results:
        sym  = "[OK]  " if r["success"] else "[FAIL]"
        col  = GREEN if r["success"] else RED
        print(
            f"  {r['task_id']:<30s}  "
            + _c(f"{r['score']:>6.2f}", col)
            + f"  {r['steps']:>5d}  "
            + _c(sym, col)
        )
    print(_c("  " + "-" * 56, DIM))
    mean_col = GREEN if mean_score >= 0.99 else (YELLOW if mean_score > 0 else RED)
    print(
        f"  {'Mean score':<30s}  "
        + _c(f"{mean_score:>6.2f}", BOLD + mean_col)
    )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle demo agent for ServerSysadminEnv (no API key needed)"
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS + ["all"],
        default="all",
        help="Which task to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]

    print()
    print(_c("=" * 68, BOLD + MAGENTA))
    print(_c("  Server Thermal Sysadmin — Oracle Demo (no API key needed)", BOLD + MAGENTA))
    print(_c(f"  Benchmark : {BENCHMARK}", DIM))
    print(_c(f"  Tasks     : {', '.join(tasks_to_run)}", DIM))
    print(_c("=" * 68, BOLD + MAGENTA))

    results: list[dict] = []
    for task_id in tasks_to_run:
        result = run_episode(task_id)
        results.append(result)

    if len(results) > 1:
        print_grand_summary(results)


if __name__ == "__main__":
    main()
