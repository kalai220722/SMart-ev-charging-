import sys
import os

# ABSOLUTE TOP PRIORITY: Path Injection
# Ensure the root directory and server directory are in sys.path.
# We resolve the absolute real path to ensure the container can find graders.
try:
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
    
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    if current_file_dir not in sys.path:
        sys.path.insert(0, current_file_dir)
except Exception as e:
    sys.stderr.write(f"[WARNING] Path injection failure: {e}\n")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Safe Station Environment.
"""

from fastapi import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    create_app = None

# Now import root modules - path injection ensures these are found
try:
    from models import SafeStationAction, SafeStationObservation
    from safe_station_environment import SafeStationEnvironment
    import graders
except ImportError:
    # Final manual fallback if the first path injection is bypassed by the loader
    sys.path.append(os.getcwd())
    import models
    import safe_station_environment
    import graders
    SafeStationAction = models.SafeStationAction
    SafeStationObservation = models.SafeStationObservation
    SafeStationEnvironment = safe_station_environment.SafeStationEnvironment

import yaml

# Manual Metadata Loading (Bulletproof Discovery)
def load_metadata():
    try:
        yaml_path = os.path.join(root_dir, "metaenv.yaml")
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        sys.stderr.write(f"[ERROR] Metadata load failed: {e}\n")
        return {}

metadata = load_metadata()
tasks = metadata.get("tasks", [])

app = create_app(
    SafeStationEnvironment,
    SafeStationAction,
    SafeStationObservation,
    env_name="safe_station",
    max_concurrent_envs=1,
) if create_app else None

if app:
    app.title = metadata.get("name", "safe_station")
    app.version = metadata.get("version", "0.1.0")

# Static files mounting
static_dir = os.path.join(root_dir, "static")
if os.path.exists(static_dir) and app:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Singleton interactive session for web UI
active_env = SafeStationEnvironment()
active_env.reset()

# --- REWARD CALCULATION HELPER ---
def compute_reward_components(action_id, grid_price, car_present_before, car_need_before, bess_before):
    base = 0.0
    peak_bonus = 0.0
    topup_bonus = 0.0
    grid_cost = 0.0
    time_pen = 0.0
    energy_grid = 0.0
    energy_batt = 0.0
    charge_amt = 0.0

    if action_id == 0 and car_present_before:
        energy_grid = 20.0
        charge_amt = 20.0
    elif action_id == 1 and car_present_before:
        energy_batt = 20.0
        charge_amt = 20.0
    elif action_id == 2 and not car_present_before:
        space_left = max(0.0, 1000.0 - bess_before)
        energy_grid = min(30.0, space_left)
        if grid_price < 5.0 and energy_grid > 0:
            topup_bonus = energy_grid * 6.0
    elif action_id == 3 and car_present_before:
        energy_grid = 20.0
        energy_batt = 20.0
        charge_amt = 40.0

    grid_cost = energy_grid * grid_price

    car_completed = False
    if car_present_before and charge_amt > 0:
        if energy_batt > 0 and grid_price > 10.0:
            peak_bonus = 50.0
        if (car_need_before - charge_amt) <= 0:
            base = 200.0
            car_completed = True

    if car_present_before and (car_need_before - charge_amt > 0):
        time_pen = 5.0

    return {
        "base": base,
        "peak_bonus": peak_bonus,
        "topup_bonus": topup_bonus,
        "grid_cost": grid_cost,
        "time_pen": time_pen,
        "car_completed": car_completed,
        "energy_grid": energy_grid,
        "energy_batt": energy_batt
    }

# --- ENRICHED API ENDPOINTS ---

@app.get("/tasks")
async def get_tasks():
    """Explicitly defined endpoint for the validator."""
    return tasks

@app.get("/metadata")
async def get_metadata_override():
    """Explicitly defined endpoint for the validator."""
    return {
        "name": metadata.get("name", "safe_station"),
        "version": metadata.get("version", "0.1.0"),
        "tasks": tasks
    }

@app.get("/api/state")
async def get_current_station_state():
    """Returns current active station environment state for dashboard telemetry."""
    return {
        "hour": active_env.hour,
        "grid_price": active_env.grid_price,
        "station_battery_level": active_env.station_battery_level,
        "car_present": active_env.car_present,
        "car_battery_need": active_env.car_battery_need,
        "step_count": active_env._state.step_count,
        "is_peak": 18 <= active_env.hour <= 22,
        "is_off_peak": 0 <= active_env.hour <= 6,
        "done": active_env._state.step_count >= 96 or active_env.station_battery_level <= 0
    }

@app.post("/api/reset")
async def reset_station_env(req: Request):
    """Resets the station environment, optionally loading specific task configuration."""
    body = {}
    try:
        body = await req.json()
    except Exception:
        pass

    obs = active_env.reset()
    if body and "h" in body:
        active_env.hour = int(body["h"])
        if hasattr(active_env, '_get_grid_price'):
            active_env.grid_price = active_env._get_grid_price(active_env.hour)
        active_env.station_battery_level = float(body.get("b", 1000.0))
        active_env.car_present = int(body.get("c", 0))
        active_env.car_battery_need = float(body.get("n", 0.0))
        obs = active_env._get_observation()

    obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
    obs_dict["step_count"] = active_env._state.step_count
    return obs_dict

@app.post("/api/step")
async def step_station_action(req: Request):
    """Executes a step in the station environment with rich mathematical breakdown."""
    payload = await req.json()
    action_val = payload.get("action")
    if isinstance(action_val, dict):
        action_val = action_val.get("action", 0)
    elif action_val is None:
        action_val = 0
    action_val = int(action_val)

    # Capture state before step
    c_bef = active_env.car_present
    n_bef = active_env.car_battery_need
    b_bef = active_env.station_battery_level
    gp = active_env.grid_price

    obs = active_env.step(SafeStationAction(action=action_val))
    breakdown = compute_reward_components(action_val, gp, c_bef, n_bef, b_bef)

    obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
    return {
        "observation": obs_dict,
        "reward": obs.reward,
        "done": obs.done,
        "breakdown": breakdown
    }

@app.post("/api/benchmark")
async def run_benchmark():
    """Runs the 3 benchmark tasks from metaenv.yaml and returns official grader results."""
    TASKS_CFG = [
        {"id": "easy_start", "initial_state": {"h": 10, "b": 500, "c": 1, "n": 40}, "grader": graders.grade_easy},
        {"id": "medium_ops", "initial_state": {"h": 19, "b": 800, "c": 1, "n": 30}, "grader": graders.grade_medium},
        {"id": "hard_constraints", "initial_state": {"h": 3, "b": 100, "c": 0, "n": 0}, "grader": graders.grade_hard}
    ]

    logs = []
    task_scores = {}

    def heuristic_action(car_present, grid_price, bess_level):
        if car_present:
            if bess_level > 200.0:
                return 1
            return 0
        else:
            if grid_price < 5.0 and bess_level < 1000.0:
                return 2
            return 0

    for task in TASKS_CFG:
        task_id = task["id"]
        cfg = task["initial_state"]
        logs.append(f"[START] task={task_id}")

        test_env = SafeStationEnvironment()
        test_env.reset()
        test_env.hour = cfg["h"]
        if hasattr(test_env, '_get_grid_price'):
            test_env.grid_price = test_env._get_grid_price(test_env.hour)
        test_env.station_battery_level = float(cfg["b"])
        test_env.car_present = cfg["c"]
        test_env.car_battery_need = float(cfg["n"])

        total_reward = 0.0
        for s in range(1, 6):
            aid = heuristic_action(test_env.car_present, test_env.grid_price, test_env.station_battery_level)
            c_bef = test_env.car_present
            n_bef = test_env.car_battery_need
            b_bef = test_env.station_battery_level
            gp = test_env.grid_price

            test_env.step(SafeStationAction(action=aid))
            comp = compute_reward_components(aid, gp, c_bef, n_bef, b_bef)
            step_reward = comp["base"] + comp["peak_bonus"] + comp["topup_bonus"] - comp["grid_cost"] - comp["time_pen"]
            total_reward += step_reward
            logs.append(f"[STEP] step={s} action={aid} reward={step_reward:+.2f}")

        # Compute grade with official graders
        score = task["grader"]({"total_reward": total_reward})
        task_scores[task_id] = score
        logs.append(f"[END] task={task_id} score={score:.2f} total_reward={total_reward:+.2f}")

    avg_score = sum(task_scores.values()) / len(task_scores)
    logs.append(f"[FINAL] Average Score across all tasks: {avg_score:.2f}")

    return {
        "average_score": avg_score,
        "task_scores": task_scores,
        "logs": logs
    }

# FastAPI Route Precedence Hack: Move discovery and custom routes to front
if app:
    discovery_paths = ["/tasks", "/metadata", "/api/state", "/api/step", "/api/reset", "/api/benchmark"]
    for i, route in enumerate(list(app.routes)):
        if hasattr(route, "path") and route.path in discovery_paths:
            r = app.routes.pop(i)
            app.routes.insert(0, r)

@app.get("/")
async def root(request: Request):
    """Serves the interactive web dashboard for browser requests; JSON for API clients."""
    accept = request.headers.get("accept", "")
    index_file = os.path.join(root_dir, "static", "index.html")
    if "text/html" in accept and os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "environment": "Safe Station RL v1.0",
        "dashboard": "/dashboard",
        "message": "Visit / for interactive UI or /docs for API explorer."
    }

@app.get("/dashboard")
async def dashboard():
    """Explicitly serves the HTML dashboard."""
    index_file = os.path.join(root_dir, "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Safe Station Frontend is loading...</h1>")

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    import uvicorn
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
