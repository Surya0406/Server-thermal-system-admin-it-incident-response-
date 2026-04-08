"""
server.py — ServerSysadminEnv REST API
=======================================
FastAPI server that wraps ServerSysadminEnv and exposes it as a
stateful REST API for the OpenEnv benchmark harness.

Endpoints
---------
POST /reset          { "task_id": "easy_fan_fix" }   → Observation
POST /step           Action JSON                       → StepResult
GET  /state                                           → Observation
GET  /health                                          → {"status": "ok"}

The /reset endpoint also accepts an empty body {} (for the
OpenEnv validation ping that sends POST /reset with body {}).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from laptop_env import (
    Action,
    Observation,
    Reward,
    ServerSysadminEnv,
    MAX_STEPS,
)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Server Thermal Sysadmin — OpenEnv",
    description=(
        "Automated IT Sysadmin Workspace.\n"
        "An LLM agent receives terminal logs about overheating servers "
        "and must issue JSON commands to resolve the incidents."
    ),
    version="1.0.0",
)

# One shared environment instance (stateful per-session)
_env = ServerSysadminEnv()


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """POST /reset body — task_id is optional so the validator ping {} works."""
    task_id: Optional[str] = Field(
        default="easy_fan_fix",
        description="One of: easy_fan_fix | medium_rogue_process | hard_db_migration",
    )


class StepResult(BaseModel):
    """Full return value of POST /step."""
    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe — always returns 200 while the server is up."""
    return {"status": "ok", "env": "server-thermal-sysadmin", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
async def reset(body: ResetRequest = ResetRequest()) -> Observation:
    """
    Initialise (or re-initialise) the environment for the given task.
    Accepts an empty body {} — defaults to 'easy_fan_fix' for the
    OpenEnv validation ping.
    """
    task_id = body.task_id or "easy_fan_fix"
    try:
        obs = _env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResult)
async def step(action: Action) -> StepResult:
    """
    Advance the environment by one step using the provided Action.
    Returns the next Observation, Reward, done flag, and info dict.
    """
    obs, reward, done, info = _env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=Observation)
async def state() -> Observation:
    """Return the current Observation without advancing the environment."""
    return _env.state()


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint — basic info about the environment."""
    return {
        "name":        "server-thermal-sysadmin",
        "version":     "1.0.0",
        "tasks":       ["easy_fan_fix", "medium_rogue_process", "hard_db_migration"],
        "max_steps":   MAX_STEPS,
        "endpoints": {
            "reset":  "POST /reset   body: {task_id: str}",
            "step":   "POST /step    body: Action JSON",
            "state":  "GET  /state",
            "health": "GET  /health",
            "docs":   "GET  /docs",
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,          # HuggingFace Spaces default port
        log_level="info",
        reload=False,
    )
