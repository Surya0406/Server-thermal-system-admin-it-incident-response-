from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Action(BaseModel):
    command: str
    target: str
    value: Optional[str] = None

class Observation(BaseModel):
    system_log: str
    stage: str
    terminal_output: str
    active_server_temp: float
    is_resolved: bool = False

class Reward(BaseModel):
    score: float
    done: bool
    reason: str

MAX_STEPS = {
    "easy_fan_fix": 3,
    "medium_rogue_process": 3,
    "hard_db_migration": 7
}

class ServerSysadminEnv:
    def __init__(self):
        self.task_id = ""
        self._state = {}
        self._last_obs: Optional["Observation"] = None
        self.step_count = 0

    def state(self) -> "Observation":
        """Return the last observation (current environment state)."""
        if self._last_obs is None:
            raise RuntimeError("Call reset() before state()")
        return self._last_obs

    def reset(self, task_id: str) -> Observation:
        self.task_id = task_id
        self.step_count = 0
        self._state = {"stage": "init"}
        if task_id == "easy_fan_fix":
            obs = Observation(
                system_log="[CRITICAL] rack-1 | temp=95°C\n— Cooling degraded. Fan check required.",
                stage="init",
                terminal_output="rack-1-fanspeed: 1200 RPM (Silent Profile)",
                active_server_temp=95.0
            )
        elif task_id == "medium_rogue_process":
            obs = Observation(
                system_log="[CRITICAL] rack-2 | temp=98°C\n— CPU thermal throttling detected.",
                stage="init",
                terminal_output="",
                active_server_temp=98.0
            )
        elif task_id == "hard_db_migration":
            obs = Observation(
                system_log="[CRITICAL] rack-3 | temp=99°C | role=primary-db\n— Imminent hardware failure. Migration needed.",
                stage="init",
                terminal_output="",
                active_server_temp=99.0
            )
        else:
            raise ValueError(f"Unknown task {task_id}")
        self._last_obs = obs
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.step_count += 1
        c = action.command
        t = action.target
        v = action.value

        info = {}
        obs: Observation
        reward: Reward
        done: bool

        if self.task_id == "easy_fan_fix":
            if c in ("diagnose", "verify_thermals"):
                obs = Observation(system_log="Diagnostics complete.", stage="s1", terminal_output="Temp is 95C", active_server_temp=95.0)
                reward, done = Reward(score=0.3, done=False, reason="Partial progress"), False
            elif c == "set_fan_profile" and t == "rack-1" and v == "high":
                obs = Observation(system_log="Task complete.", stage="done", terminal_output="Fan set to HIGH", active_server_temp=45.0, is_resolved=True)
                reward, done = Reward(score=0.999, done=True, reason="Success"), True
            else:
                obs = Observation(system_log="Invalid.", stage="init", terminal_output="", active_server_temp=95.0)
                reward, done = Reward(score=0.001, done=False, reason="Miss"), False

        elif self.task_id == "medium_rogue_process":
            if c in ("list_processes", "diagnose"):
                obs = Observation(system_log="PID 9841 consuming 100%.", stage="s1", terminal_output="PID 9841 top", active_server_temp=98.0)
                reward, done = Reward(score=0.3, done=False, reason="Listed"), False
            elif c == "kill_process":
                self._state["killed"] = True
                obs = Observation(system_log="Process killed.", stage="s2", terminal_output="Killed 9841", active_server_temp=98.0)
                reward, done = Reward(score=0.3, done=False, reason="Kill"), False
            elif c in ("verify_thermals", "set_fan_profile") and self._state.get("killed"):
                obs = Observation(system_log="Thermals nominal.", stage="done", terminal_output="", active_server_temp=45.0, is_resolved=True)
                reward, done = Reward(score=0.999, done=True, reason="Success"), True
            else:
                obs = Observation(system_log="Invalid.", stage="init", terminal_output="", active_server_temp=98.0)
                reward, done = Reward(score=0.001, done=False, reason="Miss"), False

        elif self.task_id == "hard_db_migration":
            if c in ("check_db_status", "diagnose"):
                obs = Observation(system_log="DB is active.", stage="s1", terminal_output="DB running", active_server_temp=99.0)
                reward, done = Reward(score=0.3, done=False, reason="Checked"), False
            elif c == "initiate_migration":
                self._state["migrated"] = True
                obs = Observation(system_log="Migration streaming data.", stage="s2", terminal_output="", active_server_temp=99.0)
                reward, done = Reward(score=0.3, done=False, reason="Init"), False
            elif c in ("verify_migration", "verify_thermals"):
                obs = Observation(system_log="Migration verified.", stage="s3", terminal_output="DB on rack-4", active_server_temp=99.0)
                reward, done = Reward(score=0.3, done=False, reason="Verified"), False
            elif c == "shutdown_node" and self._state.get("migrated"):
                obs = Observation(system_log="Node safely shutdown.", stage="done", terminal_output="", active_server_temp=0.0, is_resolved=True)
                reward, done = Reward(score=0.999, done=True, reason="Success"), True
            else:
                obs = Observation(system_log="Invalid.", stage="init", terminal_output="", active_server_temp=99.0)
                reward, done = Reward(score=0.001, done=False, reason="Miss"), False

        else:
            obs = Observation(system_log="Failure", stage="failed", terminal_output="", active_server_temp=0.0)
            reward, done = Reward(score=0.001, done=True, reason="Fail"), True

        self._last_obs = obs
        return obs, reward, done, info
