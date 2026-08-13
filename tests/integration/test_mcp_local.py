"""MCP lifecycle.

`test_tools_gating` is a unit test. This one proves the servers can actually spawn, 
that discovery returns real tool, and that teardown closes the process it started.

Requires [local] extra and Node.js on PATH:

    uv sync --extra local
"""
from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest
import psutil

from tests.integration.apps import local_agent, local_config
from tests.integration.driver import NO_TOOL_PROMPT, drain

from agent.tools.mcp import SERVERS, McpToolProvider

def _require_node() -> None:
    if shutil.which( "npx" ) is None:
        pytest.fail(
            "npx not on PATH; MCP stdio servers cannot be launched"
        )

@asynccontextmanager
async def _provider() -> AsyncGenerator[ McpToolProvider, None ]:
    """Start the MCP provider."""
    
    provider = McpToolProvider( SERVERS )
    try:
        await provider.setup()
    except Exception as exc:
        await provider.teardown()
        # This could be skipped instead of failed loudly.
        pytest.fail( f"MCP servers could not be started in this environment: {exc}" )
    try:
        yield provider
    except ConnectionError as exc:
        # If the connection cannot be set with the MCP servers. In this case, also 
        # fail loudly. Could also be skipped.
        pytest.fail( f"MCP session could not be established in this environment: {exc}" )
    finally:
        await provider.teardown()

async def test_setup_discovers_tools_and_gating_is_derived_from_discovery():
    """Live discovery must yield tools, and gating must be computed from what came back."""
    
    _require_node()
    from agent.policy import compile_policy, is_destructive
    
    async with _provider() as provider:
        tools = provider.tools()
        if not tools:
            pytest.fail(
                "no MCP servers reachable in this environment (0 tools discovered)"
            )
        spec = provider.gating()
        if not spec.tool_to_server:
            pytest.fail(
                "no MCP tools discovered in this environment (servers selected but "
                "returned nothing)"
            )
        
        policy = compile_policy( spec )
        
        # Unknown tool name, regardless the tools discovered
        assert policy.requires_confirmation( "not_a_real_tool" ) is True
        
        for runtime_name, raw in spec.runtime_to_raw.items():
            if is_destructive( raw ):
                assert policy.requires_confirmation( raw ) is True, (
                    f"destructive floor did not gate {runtime_name!r}"
                )

async def test_teardown_leaves_no_orphan_processes():
    """After teardown the spawned server processes must be gone.
    
    The test is limited to check whether the number of child processes is lowered 
    after teardown.
    """
    _require_node()
    
    me = psutil.Process()
    before = len( me.children( recursive = True ) )
    
    provider = McpToolProvider( SERVERS )
    try:
        await provider.setup()
    except Exception as exc:
        # We are testing teardown so we cannot count on it on exception
        await provider.teardown()
        pytest.fail(
            f"MCP servers could not be started in this environment: {exc}"
        )
    
    during = len( me.children( recursive = True ) )
    if during <= before:
        # Same remark as before
        await provider.teardown()
        pytest.fail(
            f"no MCP server subprocesses spawned in this environment; "
            "nothing to verify cleanup for"
        )
    
    await provider.teardown()
    
    # Give the os the time to reap.
    await asyncio.sleep( 1.0 )
    after = len( me.children( recursive = True ) )
    assert after <= before, (
        f"orphased child processes after teardown; before={before} after={after}"
    )

async def test_local_app_with_mcp_enabled_assembles_and_answer():
    """End-to-end with MCP on: the composite provider feeds a real turn.
    
    Proves the servers spawn, discovers runs, CompositeTools merges native + MCP 
    gatings without error, and the merged tool declaration are accepted by the model.
    
    Does not prove any gating decision or MCP server running.
    """
    
    _require_node()
    try:
        async with local_agent( local_config( enable_mcp = True ) ) as app:
            events = await drain( app, NO_TOOL_PROMPT )
            assert events, "expected events from an MCP-enabled app"
    except Exception as exc:
        pytest.fail(
            f"error occurs during servers setup or inference in local app: {exc}."
        )