"""
run.py — Single-command launcher for Server + Inference Agent
==============================================================
Starts the FastAPI environment server as a background subprocess,
waits until it is healthy, runs the LLM inference agent against all
3 tasks, then cleans up the server process.

Usage
-----
    # With HuggingFace (default):
    set HF_TOKEN=hf_...
    python run.py

    # With OpenAI key directly:
    set HF_TOKEN=sk-proj-...
    set API_BASE_URL=https://api.openai.com/v1
    set MODEL_NAME=gpt-4o
    python run.py

    # Run a single task:
    set SYSADMIN_TASK=easy_fan_fix
    python run.py

    # Run oracle demo (no API key needed):
    python run.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_PORT    = int(os.getenv("SERVER_PORT", "7860"))
SERVER_URL     = f"http://localhost:{SERVER_PORT}"
SERVER_SCRIPT  = os.path.join(os.path.dirname(__file__), "server.py")
PYTHON_EXE     = sys.executable          # same Python that is running this file
MAX_WAIT_SECS  = 15                      # seconds to wait for server startup


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wait_for_server(url: str, timeout: int = MAX_WAIT_SECS) -> bool:
    """Poll GET /health until the server responds 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def _start_server() -> subprocess.Popen:
    """Launch server.py in a background subprocess."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [PYTHON_EXE, SERVER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def _kill_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate the server subprocess."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Demo mode (oracle, no API key) ────────────────────────────────────────────

def run_demo(task: str = "all") -> None:
    """Run the oracle demo agent as a clean subprocess — no API key required."""
    demo_script = os.path.join(os.path.dirname(__file__), "demo.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [PYTHON_EXE, demo_script]
    if task != "all":
        cmd += ["--task", task]
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


# ── Live agent mode ───────────────────────────────────────────────────────────

async def run_live() -> None:
    """Start server + run the async inference agent."""
    from inference import (             # noqa: PLC0415
        ALL_TASKS, BENCHMARK, MODEL_NAME, SYSADMIN_TASK,
        SysadminEnvClient, run_episode, print_grand_summary,
    )
    from openai import OpenAI           # noqa: PLC0415

    api_key = (
        os.getenv("HF_TOKEN")
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        sys.exit(
            "\n  ERROR: No API key found.\n"
            "  Set one of:  HF_TOKEN  |  OPENAI_API_KEY\n"
            "  Example:\n"
            "      set HF_TOKEN=hf_...\n"
            "      python run.py\n"
        )

    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

    print(f"\n  [run.py] Starting environment server on port {SERVER_PORT} ...",
          flush=True)

    server_proc = _start_server()
    try:
        if not _wait_for_server(SERVER_URL, timeout=MAX_WAIT_SECS):
            # Print whatever the server wrote before failing
            out, _ = server_proc.communicate(timeout=2)
            print(out.decode(errors="replace") if out else "(no output)")
            sys.exit(
                f"\n  ERROR: Server did not start within {MAX_WAIT_SECS}s.\n"
                "  Check server.py for import errors or port conflicts.\n"
            )

        print(f"  [run.py] Server is ready at {SERVER_URL}", flush=True)
        print(f"  [run.py] Model : {model}", flush=True)
        print(f"  [run.py] Tasks : {SYSADMIN_TASK}\n", flush=True)

        llm_client     = OpenAI(base_url=api_base, api_key=api_key)
        tasks_to_run   = ALL_TASKS if SYSADMIN_TASK == "all" else [SYSADMIN_TASK]

        results = []
        for task_id in tasks_to_run:
            env_client = SysadminEnvClient(SERVER_URL)
            result = await run_episode(
                llm_client=llm_client,
                env_client=env_client,
                task_id=task_id,
            )
            results.append(result)

        if len(results) > 1:
            print_grand_summary(results)

    finally:
        print("\n  [run.py] Shutting down environment server ...", flush=True)
        _kill_server(server_proc)
        print("  [run.py] Done.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch server + inference agent in one command"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run oracle demo agent (no API key needed, no server required).",
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["easy_fan_fix", "medium_rogue_process", "hard_db_migration", "all"],
        help="Which task(s) to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if args.demo:
        run_demo(task=args.task)
    else:
        if args.task != "all":
            os.environ["SYSADMIN_TASK"] = args.task
        asyncio.run(run_live())


if __name__ == "__main__":
    main()
