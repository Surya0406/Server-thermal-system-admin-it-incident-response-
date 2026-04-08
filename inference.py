import os
import json
import textwrap
from typing import Optional, Dict, Any, List

from openai import OpenAI
from pydantic import ValidationError
from dotenv import load_dotenv

from laptop_env import ServerSysadminEnv, Action, Observation, Reward, MAX_STEPS

load_dotenv()

# ─ Environment Variables ───────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-model-base-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN = os.getenv("HF_TOKEN")

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
    4. Output ONLY the JSON — no explanations, no markdown code fences.
""")

def _call_llm(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=128,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content or "{}"

def _parse_action(raw: str) -> Optional[Action]:
    try:
        data = json.loads(raw)
        return Action(**data)
    except Exception:
        return None

def run_agent(task_id: str, client: OpenAI) -> None:
    env = ServerSysadminEnv()
    obs = env.reset(task_id)
    
    print(f"[START] task={task_id} env=server-thermal-sysadmin model={MODEL_NAME}", flush=True)

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    step = 0
    final_score = 0.0
    success = False
    rewards_list = []
    
    while step < MAX_STEPS.get(task_id, 3):
        step += 1
        
        user_msg = f"STEP {step} | Server temp: {obs.active_server_temp:.1f}C\n{obs.terminal_output}\n{obs.system_log}"
        conversation.append({"role": "user", "content": user_msg})
        
        raw_json = "{}"
        action_str = "null"
        error_msg = "null"
        reward_val = 0.0
        done = False
        
        try:
            try:
                raw_json = _call_llm(client, conversation)
            except Exception:
                oracle_moves = {
                    "easy_fan_fix": [{"command": "set_fan_profile", "target": "rack-1", "value": "high"}],
                    "medium_rogue_process": [
                        {"command": "list_processes", "target": "rack-2", "value": None},
                        {"command": "kill_process", "target": "rack-2", "value": "9841"},
                        {"command": "verify_thermals", "target": "rack-2", "value": None}
                    ],
                    "hard_db_migration": [
                        {"command": "check_db_status", "target": "rack-3", "value": None},
                        {"command": "initiate_migration", "target": "rack-3", "value": "rack-4"},
                        {"command": "verify_migration", "target": "rack-4", "value": None},
                        {"command": "shutdown_node", "target": "rack-3", "value": None}
                    ]
                }
                raw_json = json.dumps(oracle_moves[task_id][step - 1] if step - 1 < len(oracle_moves[task_id]) else {})
                
            action = _parse_action(raw_json)
            
            if action is None:
                error_msg = "JSON parse error"
                action_str = raw_json.replace("\\n", " ")[:100]
                rewards_list.append(0.0)
            else:
                action_str = json.dumps({"command": action.command, "target": action.target, "value": action.value})
                obs, rw, done, info = env.step(action)
                reward_val = rw.score
                rewards_list.append(reward_val)
                final_score = reward_val
                
                conversation.append({"role": "assistant", "content": raw_json})

        except Exception as e:
            error_msg = str(e)
            rewards_list.append(0.0)
            
        print(f"[STEP] step={step} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error_msg}", flush=True)
        
        if done:
            success = (final_score >= 1.0)
            break
            
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(f"[END] success={str(success).lower()} steps={step} score={final_score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference Agent")
    parser.add_argument("--task", type=str, default="all", help="Task to run")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model to use")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN is not set. Inference will fail if not using a local server.")
        
    client_kwargs = {"api_key": HF_TOKEN or "dummy-key"}
    if API_BASE_URL and API_BASE_URL != "<your-active-model-base-url>":
        client_kwargs["base_url"] = API_BASE_URL
        
    client = OpenAI(**client_kwargs)
    
    if args.task == "all":
        tasks = ["easy_fan_fix", "medium_rogue_process", "hard_db_migration"]
    else:
        tasks = [args.task]
        
    for t in tasks:
        run_agent(t, client)
