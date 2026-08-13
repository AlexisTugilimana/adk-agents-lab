# ADK Agent — course exercises and a deployable agent backend

This repository is two things, built in sequence:

1. **A from-scratch implementation of the [Kaggle × Google 5-Day AI Agents Intensive](https://www.kaggle.com/learn-guide/5-day-agents).**
   Instead of running the course notebooks, each day was re-implemented as a **real terminal chat client wired to a Google ADK agent**, growing one capability per day (tools → MCP → sessions/compaction/memory → observability). The code lives in [course/](course/).
   Day 5 (deployment on Vertex AI Agent Engine) has no course folder on purpose — it is delivered by part 2.

2. **An agent application that consolidates everything learned into a single, deployable backend.** Source in [src/agent/](src/agent/), tests in [tests/](tests/).
   It is a **backend only — there is no client**. The point of the exercise was to build the whole scaffold by hand: ports and adapters, a single composition root, HITL authority on the server side, telemetry, session provisioning, and a deploy path — and to prove that **the same code runs in-process and inside the Agent Engine container**.

The architectural claim worth checking: `bindings.py` is the *only* module that reads the profile. Everything downstream — the `App`, the plugins, the HITL policy, the assembly — is byte-identical local and cloud.

---

## 1. Part 1 — the course, as a real client

Each script is a self-contained terminal chat (argparse CLI, streaming event loop, graceful shutdown) that adds one concept to the previous day.

| Day | Script | Adds |
|---|---|---|
| 1 | [chat-1-simple-chat.py](course/day-1/chat-1-simple-chat.py) | `Runner`, the async event loop, `google_search` tool |
| 1 | [chat-1-team-chat.py](course/day-1/chat-1-team-chat.py) | Multi-agent team (`AgentTool`), coordinator/critic pattern, artifacts, authored events |
| 2 | [chat-2b-with-mcp.py](course/day-2/chat-2b-with-mcp.py) | `McpToolset`, stdio + streamable-HTTP transports, server registry, discovery, subprocess lifecycle |
| 2 | [chat-2b-with-compaction.py](course/day-2/chat-2b-with-compaction.py) | Tool confirmation (long-running calls), first compaction pass |
| 3 | [chat-3a-with-session.py](course/day-3/chat-3a-with-session.py) | `SessionService`: in-memory vs `DatabaseSessionService` (SQLite), resumable sessions, session audit CLI |
| 3 | [chat-3a-with-compaction.py](course/day-3/chat-3a-with-compaction.py) | Context compaction: `off` / `interval` / `token` / `hybrid`, custom summarizer model and prompt |
| 3 | [chat-3b-with-memory.py](course/day-3/chat-3b-with-memory.py) | Long-term memory: in-memory keyword store, `load_memory` / `preload_memory` retrieval, save triggers (every-turn / end / interval) |
| 4 | [chat-4a-with-observability.py](course/day-4/chat-4a-with-observability.py) | Cross-cutting telemetry as ADK **plugins**: metrics aggregation, deny-by-default HITL policy plugin |
| 5 | — | Deployment to Agent Engine → implemented as part 2 ([src/agent/deploy.py](src/agent/deploy.py)) |

Run any of them directly (they read credentials from the environment):

```bash
uv sync --extra local --extra dev

# simplest
uv run --env-file .env.local python course/day-1/chat-1-simple-chat.py

# the most complete one (day 4)
uv run --env-file .env.local python course/day-4/chat-4a-with-observability.py \
  --session-backend sqlite --compaction hybrid --memory memory --memory-retrieval preload

# useful flags on day 3/4 scripts
#   --list-tools            connect, print each MCP server's tools, exit
#   --audit                 read-only report of the SQLite session store, exit
#   --delete-session        delete --session-id from the SQLite store, exit
#   --debug / --quiet       raw event logging / clean preview
```

MCP servers are spawned with `npx`, so **Node.js must be on `PATH`** for the day-2+ scripts.

---

## 2. Part 2 — the agent app

### 2.1 Design

Clean/hexagonal layering with constructor injection. One pure core, a ring of ports, adapters behind them, and exactly **one** profile switch.

```
config.py, policy.py            PURE. No ADK, no I/O.
        ↑
ports.py                        ABCs (behaviour) + callable aliases (builders).
        ↑
plugins.py, tools/, sessions.py, telemetry.py
                                ADAPTERS. Import ADK freely, implement a port.
        ↑
app_factory.py                  The ONLY place the ADK `App` is built.
assemble.py                     The ONLY construction site for `AdkApp`. Zero profile conditionals.
bindings.py                     The switch. The only reader of `Profile`.
local_runtime.py / deploy.py    Entrypoints. Profile-blind.
```

The three invariants that produce local↔cloud parity *by construction* rather than by discipline:

1. **One switch** — [`build_agent_bindings`](src/agent/bindings.py) resolves a profile into concrete adapters.
2. **One construction site** — [`assemble_adk_app`](src/agent/assemble.py) builds the `AdkApp`; every difference arrives as a *value* (a `None` session builder, a `False` tracing flag, a different instrumentor).
3. **One App factory** — [`build_app`](src/agent/app_factory.py) is the only creator of the ADK `App` (agent + plugins + resumability + compaction).

| Concern | Port | Local adapter | Cloud adapter |
|---|---|---|---|
| Tools | `ToolProvisioning` | `CompositeTools([Native, Mcp])` | `CompositeTools([Native])` |
| Sessions | `SessionProvisioning` | `LocalSessions` → in-memory or SQLite builder | `ManagedSessions` → `None` (platform-owned) |
| Memory | `MemoryProvisioning` | `NoMemory` | `NoMemory` |
| Telemetry | `ServerTelemetry` | human stderr logs, tracing off | JSON stdout logs + GenAI instrumentor, tracing on |
| Credentials | `CredentialsCheck` | `AdcCredentials` (may `SystemExit`) | `ServiceAccountCredentials` (never raises) |
| Clock | `Clock` | `SystemClock` | `SystemClock` |

### 2.2 What is implemented

- **HITL, server-side and non-bypassable.** [`policy.py`](src/agent/policy.py) is the pure decision core; [`ConfirmationPolicyPlugin`](src/agent/plugins.py) is registered on the `App`, so a client can *answer* a gate but never *remove* one. Decision order is load-bearing: destructive-verb floor → ambiguous name → unknown tool (fail-closed) → deny-by-default unless explicitly bypassed. Pauses surface as an `adk_request_confirmation` long-running call; `ResumabilityConfig(is_resumable=True)` is always on.
- **Tools.** A native gated demo tool (`sensitive_echo`) ships in both profiles so HITL is provable on the deployed engine without MCP. MCP providers add discovery-derived gating; `CompositeTools` merges gating specs and flags cross-provider name collisions as ambiguous → gated.
- **MCP (local).** Declarative `SERVERS` registry, stdio and streamable-HTTP transports, preflight, discovery, and guaranteed subprocess teardown ([tools/mcp.py](src/agent/tools/mcp.py)).
- **Sessions.** In-memory or SQLite (`DatabaseSessionService` + `aiosqlite`, URL auto-normalized to the async driver) locally; managed Agent Engine sessions in cloud. Builders are zero-arg callables — `AdkApp` calls them itself. `LocalSessions.teardown()` disposes what it created.
- **Compaction.** `off` / `interval` / `token` / `hybrid`, optional dedicated summarizer model and prompt, validated at construction ([config.py](src/agent/config.py), [app_factory.py](src/agent/app_factory.py)).
- **Observability.** `MetricsPlugin` emits one aggregate line per invocation (latency, tool calls, tool/model errors, token usage) carrying `session_id` **and** `invocation_id` — correlation across the client/agent boundary is by these ids, not by a distributed trace. `telemetry.py` decides *how* those lines render (human vs JSON) and installs the Cloud Trace instrumentor in the container.
- **Deployment.** [`deploy.py`](src/agent/deploy.py) packages `src/agent` and the **base-only** [deployment/requirements.txt](deployment/requirements.txt) and calls `agent_engines.create`. MCP and SQLite are `[local]` extras and are never importable on the cloud path — "MCP does not deploy" is a packaging fact, not a convention.

---

## 3. Setup

### Prerequisites

- Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/)
- `gcloud auth application-default login` with a Vertex AI–enabled project
- Node.js (`npx`) — only for MCP (course day 2+, `mcp`-marked tests)

### Dependency sync

```bash
uv sync                                  # base = the deployable set (no MCP, no SQLite)
uv sync --extra local                    # + MCP toolset stack + aiosqlite
uv sync --extra dev                      # + pytest, pytest-asyncio, psutil
uv sync --extra local --extra dev        # the normal development install
```

`deployment/requirements.txt` mirrors the `[project].dependencies` block of [pyproject.toml](pyproject.toml). Keep them in sync manually — it is the literal answer to "what runs in the container".

### Environment

Copy [.env.example](.env.example) to `.env.local` (git-ignored) and fill it in:

| Variable | Used by | Meaning |
|---|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | all | Must be truthy (`1`/`true`/`yes`); `AdcCredentials` rejects otherwise |
| `GOOGLE_CLOUD_PROJECT` | all | GCP project |
| `GOOGLE_CLOUD_LOCATION` | all | **Model** region (e.g. `us-central1`) |
| `GOOGLE_CLOUD_ENGINE_LOCATION` | deploy, deployed tests | Agent Engine region |
| `GOOGLE_CLOUD_STAGING_BUCKET` | deploy | `gs://…` staging bucket |
| `AGENT_SESSION_BACKEND` | local | `memory` (default) or `sqlite` |
| `AGENT_DB_URL` | local | SQLAlchemy URL, default `sqlite+aiosqlite:///mcp_chat_session_db` |
| `AGENT_ENABLE_MCP` | local | Attach the MCP provider (`agent smoke` forces it off) |
| `AGENT_RESOURCE_NAME` | `deployed` tests | Resource name of a live Agent Engine deployment |
| `GITHUB_MCP_TOKEN` / `GITHUB_TOKEN` / `GITHUB_PERSONAL_ACCESS_TOKEN` | MCP | Bearer token for the GitHub MCP server (read-only) |

---

## 4. Running the agent app

The package exposes one console script, `agent` ([src/agent/\_\_main\_\_.py](src/agent/__main__.py)).

```bash
# One in-process turn against the real model. Native-only, in-memory sessions.
# A gated tool parks the turn — there is no approver here, so the pause IS the pass condition.
uv run --env-file .env.local agent smoke
uv run --env-file .env.local agent smoke --prompt "Please echo 'hello world' using your echo tool."

# Same thing, module form (default prompt)
uv run --env-file .env.local python -m agent.local_runtime

# Deploy the CLOUD profile to Agent Engine. MUST run from the repo root.
# Prints the Agent Engine resource name on stdout.
uv run --env-file .env.local agent deploy

# Capture it for the deployed test suite
export AGENT_RESOURCE_NAME="$(uv run --env-file .env.local agent deploy | tail -1)"
```

Embedding the agent in your own local driver:

```python
from agent.config import AgentConfig, Profile
from agent.local_runtime import local_adk_app

cfg = AgentConfig.from_env(Profile.LOCAL)
async with local_adk_app(cfg) as app:          # tools.setup() → yield → guaranteed teardown
    async for event in app.async_stream_query(message="hi", user_id="me"):
        ...
```

`local_adk_app` is the public API of this repo; treat its signature as versioned.

---

## 5. Tests

Markers are declared in [pyproject.toml](pyproject.toml):

| Marker | Cost / requirements |
|---|---|
| `unit` | Fast, free, no network, no credentials |
| `integration` | Calls a real model — needs ADC + env, **costs money** |
| `deployed` | Also needs `AGENT_RESOURCE_NAME` pointing at a live engine |
| `mcp` | Also needs the `[local]` extra and Node.js (`npx`) on `PATH` |

> `addopts = "-m integration"` is the default, so a bare `uv run pytest` runs the **integration** suite. A `-m` on the command line overrides it.

```bash
# --- unit: no credentials, no network ---
uv run pytest -m unit

# --- integration, local only (no deployment, no MCP) ---
uv run --env-file .env.local pytest -m "integration and not deployed and not mcp"

# --- MCP lifecycle (spawns npx subprocesses, asserts no orphans) ---
uv run --env-file .env.local pytest -m "integration and mcp"

# --- deployed engine (HITL parity + managed sessions in the container) ---
uv run --env-file .env.local pytest -m "deployed"

# --- default (everything marked integration, incl. deployed + mcp) ---
uv run --env-file .env.local pytest

# --- everything, unit included ---
uv run --env-file .env.local pytest -m "unit or integration"

# --- targeted runs ---
uv run pytest tests/test_policy.py -m unit
uv run --env-file .env.local pytest tests/integration/test_hitl_local.py -m integration
```

### What is covered

The suite verifies **behaviour and wiring, not answer quality** — there is no evaluation suite in this repo (see [Limitations](#6-limitations)).

**Unit** ([tests/](tests/)) — pure and wiring, zero I/O. Every fake in [tests/fakes.py](tests/fakes.py) *inherits* its port ABC, so interface drift fails at construction:
- [test_policy.py](tests/test_policy.py) — destructive floor beats bypass, unknown gated, ambiguous gated, bypass honoured.
- [test_tools_gating.py](tests/test_tools_gating.py) — `gating_from_discovery` against a canned discovery dict (no MCP connection), native provider gating, `CompositeTools` merge collisions.
- [test_metrics.py](tests/test_metrics.py) — injected `FakeClock` + capturing logger → deterministic latency, both correlation ids on the line.
- [test_assemble.py](tests/test_assemble.py) — local-shaped fakes give a session builder and `enable_tracing=False`; cloud-shaped fakes flip both **through the same function, no branch touched**.

**Integration** ([tests/integration/](tests/integration/)) — the HITL flows (`park → approve / reject / cross-session / unknown id`) live once in [scenarios.py](tests/integration/scenarios.py) and are executed by both [test_hitl_local.py](tests/integration/test_hitl_local.py) and [test_hitl_deployed.py](tests/integration/test_hitl_deployed.py), so local and deployed behaviour cannot drift. Also covered: session persistence across a new app instance, managed-session history in the cloud, the metrics/observability contract, MCP spawn-discover-teardown with a `psutil` orphan-process check, and a no-tool turn as a regression guard against over-gating.

---

## 6. Limitations

Known and deliberate — this is a learning scaffold, not a production platform.

**Framework**
- Fully coupled to Google ADK / Vertex AI. `ports.py` exposes ADK types (`ToolUnion`, `BaseSessionService`, `BaseMemoryService`) and `App` / `AdkApp` are used directly. Only `policy.py` and `config.py` would survive a framework change untouched.
- Single agent. No multi-agent orchestration in the app (the coordinator/critic team exists only in the day-1 course script).
- **No A2A** (agent-to-agent protocol).

**Sessions**
- Local persistence is SQLite only; cloud relies entirely on the managed Agent Engine session service. No parity test of storage semantics between the two.
- **No session CLI** — the day-3/4 course scripts had `--audit` / `--delete-session`; that tooling was not ported to the app. Inspection is manual.

**MCP**
- Local only. There is no cloud MCP handling: the `mcp` package is a `[local]` extra and the cloud branch cannot import it.
- The `SERVERS` registry is **hardcoded in [tools/mcp.py](src/agent/tools/mcp.py)** — no config file, no env-driven server selection, no per-user server sets.

**Memory**
- Not wired in the app: `NoMemory` under both profiles. The in-memory keyword store, `load`/`preload` retrieval and save triggers exist only in [chat-3b-with-memory.py](course/day-3/chat-3b-with-memory.py).
- No Vertex AI Memory Bank on the cloud side. The seams are reserved (`memory_tools`, `after_agent_callback` in `build_app`) but unused.
- **No RAG** — no retrieval, no vector store, no grounding corpus.

**Capabilities**
- **No artifact management** — no `ArtifactService` plugin or port (artifacts appear only in the day-1 team chat).
- HITL is **gated by default for everything**: the only relaxation surface is `bypass_confirmation` in an MCP server spec, and it can never override the destructive floor. The native demo tool is permanently gated. Expect friction if you add many benign tools.
- Compaction, model choice and instructions are **programmatic only** — `AgentConfig.from_env` does not read them, and the CLI does not expose them.
- No client. Approvals are driven by the test suite; resume mechanics (matching function-response ids, idempotent resume) are deliberately out of scope for this repo.

**Testing & evaluation**
- **No evaluation suite. Agent quality is never measured.** The tests assert *mechanics* — that a gated tool parks the turn, that an approval resumes it, that sessions persist, that MCP subprocesses die — never that an answer is correct, relevant or well-formed. There is no eval set, no golden trajectories, no LLM-as-judge, no `adk eval` / `AgentEvaluator` wiring, and therefore no regression signal on prompt, model or instruction changes.
- Integration assertions are intentionally loose about model behaviour (e.g. "some text was produced", with re-prompt retries when the model declines to call the tool), so a quality regression passes green.

**Security & operations**
- No security controls: no PII detection or redaction, no agent identity / per-caller authorization, no tool sandboxing, no rate limiting, no prompt-injection defence beyond the confirmation gate.
- No CI pipeline.
- Deployed tests require a **pre-existing** deployment (`AGENT_RESOURCE_NAME`); there is no ephemeral deploy-test-teardown cycle, and every integration run bills real model calls.
