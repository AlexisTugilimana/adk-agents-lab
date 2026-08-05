"""Local entrypoint: run the agent in-process (counterpart to deploy.py's cloud 
path).

`local_adk_app(cfg)` is the real execution path - an async context manager that 
builds the bindings, run `tools.setup()`, assembles the read `AdkApp`, yields it, 
and guarantees `tools.teardown()` on exit. Any local caller (the smoke or the client) 
uses it to obtain a live agent.

`run_smoke(cfg, prompt)` is a minimal driver: it enters `local_adk_app`, sends one 
turn, print the streamed events, and exits. It's a liveness/smoke run against the 
real stack. Run it with:

    uv run --env-file .env.local agent smoke --prompt <prompt>                  
        # via the [project.scripts] entry or
    uv run python -m --env-file .env.local agent.local_runtime --prompt <prompt> 
        # runs the module's __main__ directly
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import AsyncIterator

from vertexai.agent_engines import AdkApp

from agent.assemble import assemble_adk_app
from agent.bindings import build_agent_bindings
from agent.config import AgentConfig, Profile, SessionBackend


