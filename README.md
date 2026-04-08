# 🖥️ Server Thermal Sysadmin — Automated IT Incident Response

> **Meta PyTorch OpenEnv Hackathon Submission**
> An agentic, text-based LLM environment where an AI must diagnose and
> resolve data-centre thermal emergencies before hardware meltdown.

---

## 🚀 Motivation

Data centres generate thousands of thermal alerts every day. Human
on-call engineers must triage these incidents in minutes—often in the

This OpenEnv environment encodes three archetypal on-call runbooks as
deterministic, graded tasks. A GPT-4o baseline agent receives real-time
terminal-style logs and must issue the correct sequence of structured
JSON commands to resolve each incident within a fixed step budget.

By framing IT incident response as an RL / agentic environment we enable:
- **Automated evaluation** of LLM sysadmin capabilities.
- **Safe synthetic training** without touching live infrastructure.
- **Reproducible benchmarking** via deterministic graders (score 0.0–1.0).

---

## 📐 Environment Specification

### Action Space

Every agent action is a JSON object matching this schema:

```json
{
  "command": "string",       // sysadmin command (see table below)
  "target":  "string",       // server node, e.g. "rack-1"
  "value":   "string | null" // optional command parameter
}
```

| `command`            | `value`          | Description                                      |
|----------------------|------------------|--------------------------------------------------|
| `diagnose`           | —                | Run full hardware diagnostics on the target node. |
| `set_fan_profile`    | `low/medium/high`| Set the fan controller speed profile.             |
| `list_processes`     | —                | List top CPU-consuming processes.                 |
| `kill_process`       | `<PID>`          | Send SIGKILL to the specified process ID.         |
| `verify_thermals`    | —                | Confirm temperature has returned below threshold. |
| `check_db_status`    | —                | Inspect the database engine status.               |
| `initiate_migration` | `<dest-node>`    | Begin live database migration to a standby node.  |
| `verify_migration`   | —                | Verify migration succeeded on the destination.    |
| `shutdown_node`      | —                | Power off the target node gracefully.             |

### Observation Space

Returned after every `reset()` and `step()` call:

| Field                 | Type    | Description                                               |
|-----------------------|---------|-----------------------------------------------------------|
| `terminal_output`     | `str`   | Terminal log produced by the last action.                 |
| `active_server_temp`  | `float` | Current temperature (°C) of the primary incident server. |
| `is_resolved`         | `bool`  | `True` when the task has been fully resolved.             |

### Reward Space

| Field    | Type    | Range     | Description                                           |
|----------|---------|-----------|-------------------------------------------------------|
| `score`  | `float` | 0.0 – 1.0 | Normalised task completion score.                     |
| `reason` | `str`   | —         | Human-readable explanation of the awarded score.      |

**Scoring rubric:**

| Outcome                           | Score |
|-----------------------------------|-------|
| Correct stage completed (partial) | 0.3   |
| All stages completed correctly    | 1.0   |
| Wrong / invalid command           | 0.0   |
| Step budget (7 steps) exceeded    | 0.0   |

---

## 🎯 Tasks

### 🟢 Easy — `easy_fan_fix` (2 steps)

**Scenario:** `rack-1` is at 95 °C. A firmware update reset the fan
controller to the `low` profile, silently cutting cooling capacity.

**Optimal sequence:**
1. `diagnose` → `rack-1` — reveals the stuck fan config.
2. `set_fan_profile` → `rack-1`, `value=high` — restores cooling,
   temperature drops to 72 °C. ✓

**Why it's easy:** Direct cause → straightforward fix with no
dependencies between nodes.

---

### 🟡 Medium — `medium_rogue_process` (3 steps)

**Scenario:** `rack-2` is at 98 °C. CPU is at 100 % load, causing
thermal throttling. A rogue cryptominer process (PID 9841) is hiding
among legitimate services.

**Optimal sequence:**
1. `list_processes` → `rack-2` — identifies PID 9841 as the anomaly.
2. `kill_process` → `rack-2`, `value=9841` — terminates the miner.
3. `verify_thermals` → `rack-2` — confirms temperature at 71 °C. ✓

**Why it's medium:** The agent must discover the PID from the log
output and use it as a value in the next command—requiring two-step
information extraction.

---

### 🔴 Hard — `hard_db_migration` (4 steps)

**Scenario:** `rack-3` is at 99 °C and hosts the **primary production
database**. Shutting it down directly would cause catastrophic data
loss. The standby node `rack-4` has zero replication lag and is ready
for promotion.

**Optimal sequence:**
1. `check_db_status` → `rack-3` — confirms primary DB is active and
   `rack-4` is in sync.
2. `initiate_migration` → `rack-3`, `value=rack-4` — promotes
   `rack-4` to primary, demotes `rack-3`.
3. `verify_migration` → `rack-4` — confirms all connections
   redirected and data integrity OK.
4. `shutdown_node` → `rack-3` — safely powers off the overheating
   node. ✓

**Why it's hard:** Multi-node reasoning with a strict safety ordering
constraint — premature shutdown is never accepted regardless of the
step count.

---

## 🐳 Docker — Build & Run

### Prerequisites
- Docker 20.10+
- An OpenAI API key with access to `gpt-4o`

### Build the image

```bash
docker build -t server-thermal-sysadmin .
```

### Run all three tasks

```bash
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  server-thermal-sysadmin
```

### Run a single task

```bash
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  server-thermal-sysadmin \
  python baseline.py --task easy_fan_fix
```

### Available CLI flags

| Flag              | Default    | Description                          |
|-------------------|------------|--------------------------------------|
| `--task`          | `all`      | Task to run (`easy_fan_fix` / `medium_rogue_process` / `hard_db_migration` / `all`) |
| `--model`         | `gpt-4o`   | OpenAI model identifier              |
| `--quiet`         | `False`    | Suppress step-by-step output         |

---

## 🧩 Project Structure

```
.
├── openenv.yaml      # OpenEnv metadata specification
├── laptop_env.py     # ServerSysadminEnv (Pydantic models + grader logic)
├── baseline.py       # GPT-4o agent inference script
├── requirements.txt  # Python dependencies
├── Dockerfile        # Hugging Face Space containerisation
└── README.md         # This file
```

---

## 🛠️ Local Development (no Docker)

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the environment sanity check (no API key needed)
python laptop_env.py

# Run the full baseline agent
export OPENAI_API_KEY=sk-...
python baseline.py --task all
```

---

## 📊 Expected Baseline Performance

| Task                    | Steps | Expected Score |
|-------------------------|-------|----------------|
| `easy_fan_fix`          | 2     | 1.0            |
| `medium_rogue_process`  | 3     | 1.0            |
| `hard_db_migration`     | 4     | 1.0            |
| **Mean**                | —     | **1.0**        |

GPT-4o with `temperature=0` reliably solves all three tasks in the
optimal number of steps given the rich terminal-log context.

---

## 📄 License

MIT License — see individual file headers.

---

*Built for the Meta PyTorch OpenEnv Hackathon.*
