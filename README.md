# MCU-Purple-Baseline

Purple agent baseline for **MCU AgentBeats (MineStudio)** using the **A2A (Agent-to-Agent) protocol**.

This repository provides a minimal but fully functional **Purple policy server** that:
- Runs an A2A-compliant HTTP server (Agent Card + message endpoint)
- Receives `init` / `obs` messages from Green
- Responds with JSON `ack` / `action` payloads (evaluator-safe)
- Supports multiple policies (Rocket-1 / VPT / STEVE-1 / NoOp, optional LLM)
- Is compatible with the MCU evaluator and the included conformance tests

---

## Features

- A2A-compliant Agent Card (`/.well-known/agent-card.json`)
- Robust message parsing (TextPart JSON → typed payload)
- Robust observation decoding (base64 JPEG/PNG → RGB numpy)
- Per-`context_id` session/state management (recurrent memory, TTL GC)
- Action normalization to MineRL/VPT standard:
  - `buttons`: length 20, int {0,1}
  - `camera`: length 2, float

---

## Requirements

- Python **>= 3.10** (Recommended: 3.11 for Purple; Green can be 3.10)
- OS: **Linux / WSL** recommended
- GPU optional (CPU works for baseline; some models may be slow on CPU)
  
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
**Available agents**
- rocket1 (default pretrained Rocket-1 from Hugging Face via MineStudio)
- vpt (MineStudio VPTPolicy)
- steve1 (MineStudio SteveOnePolicy)
- noop (sanity-check baseline)
- llm (experimental; requires OPENAI_API_KEY and compatible prompt/client wiring)
Example:
```bash
python -m src.server.app --agent vpt
python -m src.server.app --agent steve1
python -m src.server.app --agent noop
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
Expected response:
```json
{
  "type": "ack",
  "success": true,
  "message": "Initialization success with task: build a house"
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
Expected response:
```json
{
  "type": "action",
  "buttons": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "camera": [0.0, 0.0]
}
```
Action contract:
- buttons: length 20, each 0/1
- camera: length 2 floats [dx, dy]
  

## Project Structure
```
MCU-Purple-Baseline/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── noop.py
│   │   ├── registry.py
│   │   ├── rocket1/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   ├── vpt/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   ├── steve1/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   └── llm/              
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── model.py
│   │       ├── preprocess.py
│   │       ├── prompt.py
│   │       ├── client.py
│   │       └── action_map.py
│   ├── protocol/
│   │   ├── __init__.py
│   │   └── models.py
│   └── server/
│       ├── __init__.py
│       ├── app.py
│       ├── executor.py
│       └── session_manager.py
├── tests/
│   ├── conftest.py
│   ├── test_agent_card.py
│   └── test_init_obs_action.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore

## Notes

- This repository contains only the Purple agent server.

- Green agent, MineStudio environment, and evaluator are external.

- Do not commit model weights, virtual environments, caches, or large outputs.

- For public deployments behind NAT/containers, use --card-url to advertise a reachable URL in the Agent Card:

## License

This project is provided as a baseline reference for the MCU AgentBeats benchmark.

## 📄 License

MIT License