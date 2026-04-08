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
        self.state = {}
        self.step_count = 0

    def reset(self, task_id: str) -> Observation:
        self.task_id = task_id
        self.step_count = 0
        self.state = {"stage": "init"}
        if task_id == "easy_fan_fix":
            return Observation(
                system_log="[CRITICAL] rack-1 | temp=95°C\n— Cooling degraded. Fan check required.",
                stage="init",
                terminal_output="rack-1-fanspeed: 1200 RPM (Silent Profile)",
                active_server_temp=95.0
            )
        elif task_id == "medium_rogue_process":
            return Observation(
                system_log="[CRITICAL] rack-2 | temp=98°C\n— CPU thermal throttling detected.",
                stage="init",
                terminal_output="",
                active_server_temp=98.0
            )
        elif task_id == "hard_db_migration":
            return Observation(
                system_log="[CRITICAL] rack-3 | temp=99°C | role=primary-db\n— Imminent hardware failure. Migration needed.",
                stage="init",
                terminal_output="",
                active_server_temp=99.0
            )
        else:
            raise ValueError(f"Unknown task {task_id}")

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.step_count += 1
        c = action.command
        t = action.target
        v = action.value
        
        info = {}
        
        if self.task_id == "easy_fan_fix":
            if c in ("diagnose", "verify_thermals"):
                obs = Observation(system_log="Diagnostics complete.", stage="s1", terminal_output="Temp is 95C", active_server_temp=95.0)
                return obs, Reward(score=0.3, done=False, reason="Partial progress"), False, info
            elif c == "set_fan_profile" and t == "rack-1" and v == "high":
                obs = Observation(system_log="Task complete.", stage="done", terminal_output="Fan set to HIGH", active_server_temp=45.0, is_resolved=True)
                return obs, Reward(score=1.0, done=True, reason="Success"), True, info
            
            return Observation(system_log="Invalid.", stage="init", terminal_output="", active_server_temp=95.0), Reward(score=0.0, done=False, reason="Miss"), False, info

        elif self.task_id == "medium_rogue_process":
            if c in ("list_processes", "diagnose"):
                return Observation(system_log="PID 9841 consuming 100%.", stage="s1", terminal_output="PID 9841 top", active_server_temp=98.0), Reward(score=0.3, done=False, reason="Listed"), False, info
            elif c == "kill_process":
                self.state["killed"] = True
                return Observation(system_log="Process killed.", stage="s2", terminal_output="Killed 9841", active_server_temp=98.0), Reward(score=0.3, done=False, reason="Kill"), False, info
            elif c in ("verify_thermals", "set_fan_profile"):
                if self.state.get("killed"):
                    return Observation(system_log="Thermals nominal.", stage="done", terminal_output="", active_server_temp=45.0, is_resolved=True), Reward(score=1.0, done=True, reason="Success"), True, info
            
            return Observation(system_log="Invalid.", stage="init", terminal_output="", active_server_temp=98.0), Reward(score=0.0, done=False, reason="Miss"), False, info

        elif self.task_id == "hard_db_migration":
            if c in ("check_db_status", "diagnose"):
                return Observation(system_log="DB is active.", stage="s1", terminal_output="DB running", active_server_temp=99.0), Reward(score=0.3, done=False, reason="Checked"), False, info
            elif c == "initiate_migration":
                self.state["migrated"] = True
                return Observation(system_log="Migration streaming data.", stage="s2", terminal_output="", active_server_temp=99.0), Reward(score=0.3, done=False, reason="Init"), False, info
            elif c in ("verify_migration", "verify_thermals"):
                return Observation(system_log="Migration verified.", stage="s3", terminal_output="DB on rack-4", active_server_temp=99.0), Reward(score=0.3, done=False, reason="Verified"), False, info
            elif c == "shutdown_node":
                if self.state.get("migrated"):
                    return Observation(system_log="Node safely shutdown.", stage="done", terminal_output="", active_server_temp=0.0, is_resolved=True), Reward(score=1.0, done=True, reason="Success"), True, info
            
            return Observation(system_log="Invalid.", stage="init", terminal_output="", active_server_temp=99.0), Reward(score=0.0, done=False, reason="Miss"), False, info

        return Observation(system_log="Failure", stage="failed", terminal_output="", active_server_temp=0.0), Reward(score=0.0, done=True, reason="Fail"), True, info
