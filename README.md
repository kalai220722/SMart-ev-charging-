---
title: Safe Station Environment Server
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - ev-management
---

# Safe Station Environment ⚡

An advanced Reinforcement Learning environment and digital twin for managing a hard-constrained EV Smart-Station equipped with a Battery Energy Storage System (BESS) and dynamic electricity tariffs. Developed for the **Meta PyTorch OpenEnv Hackathon 2026**.

---

## 🖥️ Interactive Frontend Dashboard

Safe Station now features **two complete frontend interfaces**:

### 1. Modern Web Application (FastAPI + HTML5 + Tailwind + Chart.js)
- **Live Station Digital Twin**: Real-time animated SVG power flows showing energy direction between Central Grid, Station BESS, and EV Vehicle Bay.
- **Dynamic Action Command Pad**: 4 tactical action buttons with real-time tariff preview and arbitrage hints.
- **Autonomous AI Simulation Engine**: Auto-run heuristic agent with speed controls (1x, 2x, 5x, Max).
- **Meta Hackathon Benchmark Arena**: One-click execution of Easy, Medium, and Hard task evaluations with official `graders.py` scoring.
- **Telemetry Charts**: Cumulative profit trajectory, BESS level vs. tariff timeline, and live step log.

To launch:
```bash
python safe_station/run_frontend.py
```
Or with Uvicorn:
```bash
python -m uvicorn safe_station.server.app:app --port 8000
```
Open **`http://localhost:8000`** in any web browser.

---

### 2. Streamlit Dashboard
If you prefer Streamlit, launch the interactive Streamlit dashboard:
```bash
streamlit run safe_station/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Environment Specifications

- **Goal**: Maximize profit by smart charging EVs while minimizing grid costs.
- **State Space**:
  - `hour`: Time of day (0 to 23).
  - `grid_price`: Dynamic electricity tariff (₹4.0 off-peak 0:00-06:00, ₹15.0 peak 18:00-22:00, ₹8.0 normal).
  - `station_battery_level`: BESS storage (0.0 to 100.0 kWh).
  - `car_present`: Binary flag (1 if EV is in charging bay, 0 otherwise).
  - `car_battery_need`: Energy needed by vehicle (0.0 to 100.0 kWh).
- **Action Space**:
  - `0`: Grid Charge (20 units from Grid to EV; cost = `20 * grid_price`).
  - `1`: Battery Charge (20 units from BESS to EV; `0` grid cost + ₹50 peak bonus).
  - `2`: Top-Up BESS (Refill up to 30 units into BESS; earns `energy * 6.0` off-peak bonus).
  - `3`: Hybrid Charge (40 units total: 20 Grid + 20 BESS for rapid customer turnover).

---

## 🏆 Reward System

- **+200.0**: Successfully charging a car to completion.
- **+50.0**: Peak-Shaving bonus (using battery when grid price > 10.0).
- **+Profit**: Reward for topping up BESS during cheap off-peak hours (`price < 5.0`).
- **-Cost**: Direct electricity tariff penalty for grid consumption (`-energy * price`).
- **-5.0**: Customer wait penalty per 15-minute interval while car is incomplete.
- **-500.0**: Critical failure penalty if station BESS hits `0 kWh`.

---

## 🔌 OpenEnv REST API Endpoints

- `GET /`: Serves the interactive web dashboard (HTML) or API status (JSON).
- `GET /api/state`: Current telemetry, tariffs, and recommendations.
- `POST /api/step`: Executes action with granular mathematical reward breakdown.
- `POST /api/reset`: Resets episode or sets specific task presets.
- `POST /api/benchmark`: Runs full evaluation across the 3 tasks and returns scores.
- `POST /step`: OpenEnv standard step endpoint (`{"action": {"action": 0..3}}`).
- `POST /reset`: OpenEnv standard environment reset.
- `GET /tasks`: OpenEnv task metadata.
- `GET /metadata`: Environment version and schema.
- `GET /health`: Server health check.
- `GET /docs`: Interactive Swagger API documentation.
