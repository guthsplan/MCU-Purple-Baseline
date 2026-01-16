# MCU-Purple-Baseline

A minimal but fully functional **Purple agent baseline** for **MCU AgentBeats (MineStudio)** using the **A2A (Agent-to-Agent) protocol**.

This repository provides a reference implementation of an A2A-compliant policy server that responds to task instructions and observations from the Green evaluator agent.

## 🎯 Overview

The Purple Agent is responsible for:
- **Task Initialization**: Receive task instructions (e.g., "craft oak planks") and prepare internal state
- **Perception**: Decode observation images (128×128 RGB) from the environment
- **Decision Making**: Execute policies (Rocket-1, VPT, STEVE-1, or custom) to generate actions
- **Action Output**: Return standardized action formats (buttons + camera controls)

All communication follows the **A2A protocol** with JSON message contracts, enabling seamless integration with:
- 🟢 Green Agent (MCU evaluator)
- 🎮 MineStudio environment
- ✅ Conformance tests

### Key Features

✅ **A2A-compliant** - Agent Card + standard message endpoints  
✅ **Multiple Policies** - Rocket-1, VPT, STEVE-1, NoOp, LLM (experimental)  
✅ **Robust Observation Handling** - Base64 decoding, image validation, preprocessing  
✅ **Session Management** - Per-context state tracking with TTL garbage collection  
✅ **Evaluator-Safe** - Standardized action format (23 buttons + 2-D camera)  

---

## 🚀 Quick Start

### Prerequisites

- Python **>= 3.10** (Recommended: 3.11)
- OS: **Linux / WSL** recommended
- GPU optional (CPU works; some models may be slow)

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<YOUR_ID>/MCU-Purple-Baseline.git
cd MCU-Purple-Baseline

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Purple Agent Server

**Default (Rocket-1 policy):**
```bash
python -m src.server.app --agent rocket1
```

**Custom configuration:**
```bash
python -m src.server.app --host 0.0.0.0 --port 9019 --agent steve1
```

**Available policies:**
- `rocket1` (default) - Pretrained Rocket-1 via Hugging Face
- `vpt` - MineStudio VPT policy
- `steve1` - MineStudio STEVE-1 policy  
- `noop` - No-op baseline (returns [0, 0, ..., 0, 0])
- `llm` - Experimental LLM-based policy (requires `OPENAI_API_KEY`)

### 3. Verify Agent Card

Check that the server is running and discoverable:

```bash
curl http://localhost:9019/.well-known/agent-card.json
```

Expected response: JSON agent card with metadata (required for Green agent discovery)

---

## Message Protocol

### Context Management

Each task is identified by a `context_id` assigned by the Green agent.

- `init` is always called first for a given `context_id`
- All subsequent `obs` messages reuse the same `context_id`
- The Purple Agent must maintain per-context state (e.g. RNN memory)
- State must not leak across different `context_id`s
- When a task finishes, the context may be discarded

Your Purple Agent must implement the following A2A message handlers:

### 1. Initialization
**Request:**
```json
{
  "text": "craft oak planks from oak logs"
}
```
**Response:**
```json
{
  "success": true,
  "message": "Ready"
}
```

The Purple Agent should:
- Parse the task instruction from `text`
- Initialize internal state/policies for the task
- Return `success: true` when ready, or `false` if initialization fails
- Optionally include a descriptive `message`
- Initialization is guaranteed to be called exactly once per `context_id`
before any observation messages are sent.


### 2. Observation → Action

### Observation Image Contract

- `obs` is a base64-encoded RGB image (JPEG or PNG)
- After decoding, the image MUST satisfy:
  - Shape: (H, W, 3)
  - Dtype: `uint8` or `float32`
  - Color order: RGB (NOT BGR)

The Purple Agent is responsible for decoding, validating,
and converting the image before passing it to the policy.

**Request (Observation):**
```json
{
  "step": 42,
  "obs": "<base64_encoded_128x128_image>"
}
```

**Response (Action):**
```json
{
  "buttons": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "camera": [0, 60]
}
```
### Internal vs External Action Format

Some policies (e.g. Rocket-1, STEVE-1) internally operate on
token-based or compressed action representations.

However, the Purple Agent MUST always return actions in the
environment-compatible format expected by the Green agent:

- `buttons`: length 23
- `camera`: length 2

Any internal token or latent representation must be converted
before responding to the Green agent.

### Action Space Specification
- **buttons**: Array of 23 integers (0 or 1)
  - Index meanings: Forward, Back, Left, Right, Jump, Sneak, Sprint, Attack, Use, and more
  - Each element must be 0 (inactive) or 1 (active)
  
- **camera**: Array of 2 integers [yaw, pitch]
  - **yaw**: Rotation around vertical axis, typically [-180, 180]
  - **pitch**: Vertical view angle, typically [-90, 90]
  - Represents delta changes from current camera state

### Implementation Notes
- The `step` field indicates the current environment step number (0-indexed)
- The `obs` is always a **128x128 RGB image** in base64-encoded format (JPEG/PNG)
- Response must be returned **promptly** to avoid timeout
- Actions are applied every game tick (20 ticks/second in Minecraft)
  
### Message Flow (Simplified)

Green → Purple:
1. init(context_id, task_text)
2. obs(context_id, step=0, image)
3. obs(context_id, step=1, image)
4. ...

Purple → Green:
1. ack(success)
2. action(buttons, camera)
3. action(buttons, camera)
4. ...

## 📁Project Structure
```

MCU-Purple-Baseline/
├── README.md
├── Dockerfile.rocket1
├── Dockerfile.steve1
├── Dockerfile.vpt
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── src/
│   ├── action/
│   │   ├── __init__.py
│   │   ├── action_space.py
│   │   └── pipeline.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── download.py
│   │   ├── noop.py
│   │   ├── registry.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── action_map.py
│   │   │   ├── agent.py
│   │   │   ├── client.py
│   │   │   ├── model.py
│   │   │   ├── preprocess.py
│   │   │   └── prompt.py
│   │   ├── rocket1/
│   │   │   ├── __init__.py
│   │   │   ├── action_formatter.py
│   │   │   ├── agent.py
│   │   │   ├── input_validator.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   ├── steve1/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   └── vpt/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── model.py
│   │       └── preprocess.py
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
└── .github/
    └── workflows/
        └── test-and-publish.yml

```
## Notes

- This repository contains only the Purple agent server.

- Green agent, MineStudio environment, and evaluator are external.

- Do not commit model weights, virtual environments, caches, or large outputs.

- For public deployments behind NAT/containers, use --card-url to advertise a reachable URL in the Agent Card:

## Out of Scope

The Purple Agent does NOT:
- Launch the Minecraft simulator
- Compute rewards or scores
- Manage episode termination
- Perform evaluation or video scoring

These responsibilities belong to the Green agent and evaluator.

## License

This project is provided as a baseline reference for the MCU AgentBeats benchmark.

## 📄 License

MIT License