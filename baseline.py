"""
baseline.py — GPT-4o Baseline Agent for ServerSysadminEnv
===========================================================
Runs an OpenAI GPT-4o agent against all three tasks in the
Automated IT Sysadmin Workspace OpenEnv environment.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python baseline.py [--task TASK_ID]

If --task is omitted the agent runs all three tasks sequentially
and prints a final summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from laptop_env import Action, Observation, Reward, ServerSysadminEnv, MAX_STEPS


# ═══════════════════════════════════════════════════════════════════════════
# ANSI colour helpers (gracefully degraded on non-TTY)
# ═══════════════════════════════════════════════════════════════════════════

_IS_TTY = sys.stdout.isatty() or os.environ.get("FORCE_COLOUR")

RESET  = "\033[0m"   if _IS_TTY else ""
BOLD   = "\033[1m"   if _IS_TTY else ""
RED    = "\033[91m"  if _IS_TTY else ""
GREEN  = "\033[92m"  if _IS_TTY else ""
YELLOW = "\033[93m"  if _IS_TTY else ""
CYAN   = "\033[96m"  if _IS_TTY else ""
MAGENTA= "\033[95m"  if _IS_TTY else ""
DIM    = "\033[2m"   if _IS_TTY else ""


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _score_colour(score: float) -> str:
    if score >= 1.0:
        return GREEN
    if score > 0.0:
        return YELLOW
    return RED


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert IT sysadmin AI agent responding to data-centre
    thermal emergencies. You receive real-time terminal logs and must
    issue the MINIMUM number of structured commands to resolve the
    incident before hardware meltdown.

    RESPONSE FORMAT — you MUST reply with a single valid JSON object
    that matches this schema exactly (no markdown, no prose, just JSON):

    {
      "command": "<string>",
      "target":  "<string>",
      "value":   "<string | null>"
    }

    AVAILABLE COMMANDS (by task context):
      diagnose           — Run hardware diagnostics on a node.
      set_fan_profile    — Set fan speed profile; value = "low"|"medium"|"high".
      list_processes     — List top CPU-consuming processes on a node.
      kill_process       — Kill a process; value = PID as string.
      verify_thermals    — Confirm temperature has returned to safe range.
      check_db_status    — Inspect the database engine status on a node.
      initiate_migration — Begin live DB migration; value = destination node name.
      verify_migration   — Confirm migration succeeded on the destination node.
      shutdown_node      — Safely power off a node (only after data is safe).

    RULES:
    1. Read every line of the terminal output carefully before deciding.
    2. Always act on the PRIMARY affected node unless the task requires
       a different target.
    3. Never shut down a database node before migrating its data.
    4. You have at most """ + str(MAX_STEPS) + """ steps per episode.
    5. Output ONLY the JSON — no explanations, no markdown code fences.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Agent loop
# ═══════════════════════════════════════════════════════════════════════════

def run_agent(
    task_id: str,
    client: OpenAI,
    *,
    model: str = "gpt-4o",
    verbose: bool = True,
) -> dict:
    """
    Run the GPT-4o agent on a single task until done or step budget exceeded.

    Parameters
    ----------
    task_id : str
        One of 'easy_fan_fix', 'medium_rogue_process', 'hard_db_migration'.
    client : OpenAI
        Initialised OpenAI client (API key already set).
    model : str
        OpenAI model identifier (default: gpt-4o).
    verbose : bool
        Print step-by-step interaction to stdout.

    Returns
    -------
    dict
        Summary with keys: task_id, resolved, score, steps, model.
    """
    env  = ServerSysadminEnv()
    obs: Observation = env.reset(task_id)

    if verbose:
        _print_task_header(task_id)
        _print_obs(obs, step=0)

    conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0
    step = 0

    while True:
        step += 1

        # ── Build user message from current observation ────────────────────
        user_msg = (
            f"STEP {step} | Server temp: {obs.active_server_temp:.1f}°C\n\n"
            f"{obs.terminal_output}"
        )
        conversation.append({"role": "user", "content": user_msg})

        # ── Call the LLM ───────────────────────────────────────────────────
        raw_action_json: str = _call_llm(
            client=client,
            model=model,
            messages=conversation,
        )

        if verbose:
            print(
                _c(f"\n  [GPT-4o → Step {step}] ", BOLD + CYAN)
                + _c(raw_action_json, DIM)
            )

        # ── Parse the JSON into an Action Pydantic model ───────────────────
        action = _parse_action(raw_action_json)

        if action is None:
            # Unparseable — inject error into conversation and retry
            err_msg = (
                "[PARSE ERROR] Your response was not valid JSON matching "
                "the Action schema. Reply with ONLY a JSON object containing "
                "'command', 'target', and optionally 'value'."
            )
            conversation.append({"role": "assistant", "content": raw_action_json})
            conversation.append({"role": "user",      "content": err_msg})
            if verbose:
                print(_c(f"  [WARNING] Failed to parse action. Retrying...", YELLOW))
            continue  # retry without consuming an env step

        # ── Step the environment ───────────────────────────────────────────
        obs, reward, done, info = env.step(action)
        final_score = reward.score

        # Record the assistant turn + environment response into conversation
        conversation.append({"role": "assistant", "content": raw_action_json})

        if verbose:
            _print_action(action, step)
            _print_reward(reward)
            _print_obs(obs, step=step)

        if done:
            break

    if verbose:
        _print_episode_summary(task_id, obs.is_resolved, final_score, step, model)

    return {
        "task_id":  task_id,
        "resolved": obs.is_resolved,
        "score":    final_score,
        "steps":    step,
        "model":    model,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LLM call + JSON parsing
# ═══════════════════════════════════════════════════════════════════════════

def _call_llm(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Call the OpenAI chat completions endpoint and return raw JSON string."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,       # deterministic for reproducible evaluation
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


def _parse_action(raw: str) -> Optional[Action]:
    """Try to parse a raw JSON string into an Action; return None on failure."""
    try:
        data = json.loads(raw)
        return Action(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_task_header(task_id: str) -> None:
    bar = "═" * 70
    print()
    print(_c(bar, BOLD))
    print(_c(f"  TASK : {task_id.upper().replace('_', ' ')}", BOLD + MAGENTA))
    print(_c(bar, BOLD))


def _print_obs(obs: Observation, step: int) -> None:
    prefix = "  INIT " if step == 0 else f"  OBS  "
    temp_col = RED if obs.active_server_temp >= 90 else (YELLOW if obs.active_server_temp >= 75 else GREEN)
    print()
    print(_c(f"{prefix}| temp={obs.active_server_temp:.1f}°C", temp_col))
    for line in obs.terminal_output.splitlines():
        print(_c(f"         {line}", DIM))


def _print_action(action: Action, step: int) -> None:
    val_str = f" value={action.value!r}" if action.value else ""
    print(
        _c(f"  ACT  ", BOLD + CYAN)
        + f"| cmd={action.command}  target={action.target}{val_str}"
    )


def _print_reward(reward: Reward) -> None:
    col = _score_colour(reward.score)
    print(
        _c(f"  REW  ", BOLD)
        + _c(f"| score={reward.score:.1f}", col)
        + f"  reason={reward.reason}"
    )


def _print_episode_summary(
    task_id: str, resolved: bool, score: float, steps: int, model: str
) -> None:
    col = GREEN if resolved else RED
    sym = "✓ RESOLVED" if resolved else "✗ FAILED"
    print()
    print(_c("─" * 70, BOLD))
    print(_c(f"  EPISODE SUMMARY : {task_id}", BOLD))
    print(f"    Status  : {_c(sym, col)}")
    print(f"    Score   : {_c(f'{score:.1f}', _score_colour(score))}")
    print(f"    Steps   : {steps} / {MAX_STEPS}")
    print(f"    Model   : {model}")
    print(_c("─" * 70, BOLD))


def _print_grand_summary(results: list[dict]) -> None:
    mean_score = sum(r["score"] for r in results) / len(results)
    print()
    print(_c("═" * 70, BOLD + MAGENTA))
    print(_c("  GRAND SUMMARY — All Tasks", BOLD + MAGENTA))
    print(_c("═" * 70, BOLD + MAGENTA))
    header = f"  {'Task':<30s}  {'Score':>6s}  {'Steps':>5s}  {'Status'}"
    print(_c(header, BOLD))
    print(_c("  " + "─" * 66, DIM))
    for r in results:
        sym = "✓" if r["resolved"] else "✗"
        col = GREEN if r["resolved"] else RED
        print(
            f"  {r['task_id']:<30s}  "
            + _c(f"{r['score']:>6.1f}", _score_colour(r["score"]))
            + f"  {r['steps']:>5d}  "
            + _c(sym, col)
        )
    print(_c("  " + "─" * 66, DIM))
    print(
        f"  {'Mean score':<30s}  "
        + _c(f"{mean_score:>6.2f}", _score_colour(mean_score))
    )
    print(_c("═" * 70, BOLD + MAGENTA))
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

ALL_TASKS = ["easy_fan_fix", "medium_rogue_process", "hard_db_migration"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-4o baseline agent for ServerSysadminEnv"
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS + ["all"],
        default="all",
        help="Which task(s) to run (default: all).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output; only show the final summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── API key check ──────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "\n  ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "  Export it before running:\n"
            "      export OPENAI_API_KEY=sk-...\n"
        )

    client = OpenAI(api_key=api_key)

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    verbose = not args.quiet

    print(_c("\n  Automated IT Sysadmin Workspace — GPT-4o Baseline", BOLD + CYAN))
    print(_c(f"  Model : {args.model}  |  Tasks : {', '.join(tasks_to_run)}", DIM))

    results: list[dict] = []
    for task_id in tasks_to_run:
        result = run_agent(
            task_id=task_id,
            client=client,
            model=args.model,
            verbose=verbose,
        )
        results.append(result)

    if len(results) > 1 or not verbose:
        _print_grand_summary(results)


if __name__ == "__main__":
    main()
