"""Local entrypoint: run the agent in-process (counterpart to deploy.py's cloud 
path).

`local_adk_app(cfg)` is the real execution path - an async context manager that 
builds the bindings, run `tools.setup()`, assembles the read `AdkApp`, yields it, 
and guarantees `tools.teardown()` on exit. Any local caller (the smoke or the client) 
uses it to obtain a live agent.

`run_smoke(cfg, prompt)` is a minimal driver: it enters `local_adk_app`, sends one 
turn, print the streamed events, and exits. It's a liveness/smoke run against the 
real stack. Run it with:

    # CLI - supports a custom prompt:
    uv run --env-file .env.local agent smoke --prompt "<prompt>"
    
    # Module form - default prompt is used:
    uv run --env-file .env.local python -m agent.local_runtime
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import AsyncGenerator

from vertexai.agent_engines import AdkApp

from agent.assemble import assemble_adk_app
from agent.bindings import build_agent_bindings
from agent.config import AgentConfig, Profile, SessionBackend

_DEFAULT_SMOKE_PROMPT = "Please echo 'hello world' using your echo tool."
_LOCAL = Profile( "local" )
log = logging.getLogger( "agent.local_runtime" )

@asynccontextmanager
async def local_adk_app( cfg: AgentConfig ) -> AsyncGenerator[ AdkApp ]:
    """Yield an in-process `AdkApp` with its tool lifecycle managed."""
    
    bindings = build_agent_bindings( cfg )
    await bindings.tools.setup()
    try:
        yield assemble_adk_app( cfg, bindings, adk_app_factory = AdkApp )
    finally:
        await bindings.tools.teardown()
    
async def run_smoke( cfg: AgentConfig, prompt: str = _DEFAULT_SMOKE_PROMPT ) -> None:
    """Run one turn in-process and print the streamed events."""
    
    async with local_adk_app( cfg ) as app:
        print( f"[smoke] sending: {prompt}\n" )
        async for event in app.async_stream_query(
            message = prompt, user_id = "smoke-user"
        ):
            print( event )
        
def _default_smoke_config() -> AgentConfig:
    """A native-only, in-memory local config for the smoke. Project/location/Vertex 
    come from the environment variables."""
    
    return replace(
        AgentConfig.from_env( _LOCAL ),
        session_backend = SessionBackend.MEMORY,
        enable_mcp = False
    )

if __name__ == "__main__":
    
    logging.basicConfig( level = logging.INFO )
    asyncio.run( run_smoke( _default_smoke_config() ) )