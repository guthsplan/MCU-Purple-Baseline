# MCU-Purple-Baseline

Purple agent baseline for **MCU AgentBeats (MineStudio)** using the **A2A (Agent-to-Agent) protocol**.

This repository provides a minimal but fully functional **Purple policy server** that:
- Implements an A2A-compliant HTTP server
- Receives `init` / `obs` messages from Green
- Responds with `ack` / `action` payloads
- Uses **Rocket-1 (MineStudio)** as the default policy
- Is compatible with the **MCU evaluator and conformance tests**

---

## Features

- ✅ A2A-compliant Agent Card and message handling
- ✅ Rocket-1 policy (ViT-based, pretrained via Hugging Face)
- ✅ Proper session / context_id management
- ✅ Deterministic action sampling (baseline-safe)
- ✅ Robust image decoding (base64 → RGB)
- ✅ Always returns JSON-completed tasks (evaluator-safe)

---

## Requirements

- Python **>= 3.10**
- OS: **Linux / WSL** (recommended)
- GPU optional (CPU works for baseline)

---

## Installation

### 1. Clone repository
```bash
git clone https://github.com/<YOUR_ID>/MCU-Purple-Baseline.git
cd MCU-Purple-Baseline
```
### 2. Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### Run the Purple Agent
```bash
python -m src.server.app --agent rocket1
```
**Default settings**
- Host: 127.0.0.1
- Port: 9019
- Agent: rocket1
**You can override options:**
```bash
python -m src.server.app --host 127.0.0.1 --port 9019 --agent rocket1
```

## Verify Agent Card
Once running, the agent card should be available at:
```bash
http://localhost:9008/.well-known/agent-card.json
```
**This endpoint is required for:**
- A2A client discovery
- MCU evaluator
- Conformance tests

## Message Protocol (Summary) 
### init
```json
{
  "type": "init",
  "text": "build a house"
}
```
### Observation
```json
{
  "type": "obs",
  "step": 0,
  "obs": "<base64-encoded image>"
}
```
### Action (response)
```json
{
  "type": "action",
  "buttons": [0, 1, 0, ...],  // length = 20
  "camera": [dx, dy]         // length = 2
}
```
## Project Structure
```
MCU-Purple-Baseline/
├── src/
│   ├── agent/
│   │   ├── base.py
│   │   ├── noop.py
│   │   └── rocket1/
│   │       ├── agent.py
│   │       ├── model.py
│   │       └── preprocess.py
│   ├── protocol/
│   │   └── models.py
│   └── server/
│       ├── app.py
│       ├── executor.py
│       └── session_manager.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```
## Notes

- This repository contains only the Purple agent.

- Green agent, MineStudio environment, and evaluator are external.

- Do not commit model weights, virtual environments, or caches.

## License

This project is provided as a baseline reference for the MCU AgentBeats benchmark.

## 📄 License

MIT License