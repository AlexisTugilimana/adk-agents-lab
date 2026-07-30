"""MCP tool provider: registry, transports, lifecycle, and pure gating.

Layer: adapter. This whole package depends on **LOCAL EXTRA** - it requires 
`mcp` package and  must never been imported under the cloud profile. It is 
imported lazily under the local branch only.
"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

from google.adk.agents.llm_agent import ToolUnion
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StreamableHTTPConnectionParams
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters

from agent.policy import GatingSpec
from agent.ports import ToolProvisioning

log = logging.getLogger( "agent.mcp" )

ConnectionParams = Union[ StdioConnectionParams, StreamableHTTPConnectionParams ]

# ==============================================================================
# Transport
# ==============================================================================
class Transport( ABC ):
    """How to reach one MCP server."""
    
    @abstractmethod
    def connection_params( self, timeout: float ) -> ConnectionParams:
        """Build the ADK connection parameters for this transport."""
    
    @abstractmethod
    def preflight( self ) -> list[ str ]:
        """Return a list of human-readable problems. Empty == ready."""
    
    @abstractmethod
    def describe( self ) -> str:
        """One-line description of the server."""

@dataclass
class StdioTransport( Transport ):
    """Server launched as a local subprocess speaking over stdio."""
    
    command: str
    args: list[ str ] = field( default_factory = list )
    
    def connection_params( self, timeout: float ) -> StdioConnectionParams:
        return StdioConnectionParams(
            server_params = StdioServerParameters( command = self.command, args = self.args ),
            timeout = timeout
        )
    
    def preflight( self ) -> list[ str ]:
        if shutil.which( self.command ) is None:
            hint = "(install Node.js for npx/node)" if self.command in { "npx", "node" } else ""
            return [ f"command {self.command!r} not found on PATH{hint}" ]
        return []

    def describe( self ) -> str:
        return f"stdio: {self.command}: {' '.join( self.args )}".strip()

@dataclass
class StreamableHTTPTransport( Transport ):
    """Remote server reached over streamable HTTP."""
    
    url: str
    headers: dict[ str, str | None ] = field( default_factory = dict )
    
    def connection_params( self, timeout: float ) -> StreamableHTTPConnectionParams:
        return StreamableHTTPConnectionParams(
            url = self.url,
            headers = self._resolve_headers(),
            timeout = timeout
        )
    
    def _resolve_headers( self ) -> dict[ str, str ]:
        return { key: value for key, value in self.headers.items() if value is not None }

    def preflight( self ) -> list[ str ]:
        missing = [ key for key, value in self.headers.items() if value is None ]
        if missing:
            return [ f"unresolved header(s): {' '.join( missing )}" ]
        return []
    
    def describe(self) -> str:
        return f"http: {self.url}"
    
# ==============================================================================
# Server registry
# ==============================================================================
@dataclass
class McpServerSpec:
    """Declarative description of one MCP server.
    
    Confirmation is deny-by-default. `bypass_confirmation` is an opt-in allowlist 
    of raw tool names on this server that may skip the gate; it can never skip the 
    destructive floor.
    """
    
    name: str
    transport: Transport
    tool_filter: list[ str ] | None
    timeout: float = 30.0
    enabled: bool = True
    bypass_confirmation: frozenset[ str ] = field( default_factory = frozenset )

def _github_transport() -> StreamableHTTPTransport:
    """Github transport, resolving the bearer token from the environment."""
    token = (
        os.environ.get( "GITHUB_MCP_TOKEN" )
        or os.environ.get( "GITHUB_TOKEN" )
        or os.environ.get( "GITHUB_PERSONAL_ACCESS_TOKEN" )
    )
    return StreamableHTTPTransport(
        url = "https://api.githubcopilot.com/mcp/",
        headers = {
            "Authorization": f"Bearer {token}" if token else None,
            "X-MCP-Toolsets": "all",
            "X-MCP-Readonly": "true",
        }
    )

# The available MCP servers with their specs. `tool_filter` scopes which tools are 
# exposed.
SERVERS: list[ McpServerSpec ] = [
    McpServerSpec(
        name = "everything",
        transport = StdioTransport(
            command = "npx", args = [ "y", "@modelcontextprotocol/server-everything" ]
        ),
        tool_filter = [ "echo", "get-sum", "get-env" ],
        timeout = 30.0,
        bypass_confirmation = frozenset( { "echo" } )
    ),
    McpServerSpec(
        name = "github",
        transport = _github_transport(),
        tool_filter = None,
        timeout = 30.0
    )
]
    
# ==============================================================================
# Toolset lifecycle
# ==============================================================================
def build_toolsets( specs: list[ McpServerSpec ] ) -> list[ McpToolset ]:
    """Construct one McpToolset per spec."""
    
    toolsets: list[ McpToolset ] = []
    for spec in specs:
        toolset = McpToolset(
            connection_params = spec.transport.connection_params( spec.timeout ),
            tool_filter = spec.tool_filter,
            tool_name_prefix = spec.name
        )
        toolsets.append( toolset )
        log.info(
            f"toolset.built name={spec.name} via={spec.transport.describe()} filter={spec.tool_filter}"
        )
    return toolsets

async def discover_tools( toolsets: list[ McpToolset ] ) -> dict[ int, list[ str ] ]:
    """Ask each server what tools it actually exposes."""
    
    discovered: dict[ int, list[ str ] ] = {}
    for index, toolset in enumerate( toolsets ):
        try:
            tools = await toolset.get_tools()
            names = [ tool.name for tool in tools ]
        except Exception:
            log.exception( f"toolset.discover.error index={index}" )
            names = []
        discovered[ index ] = names
        log.info( f"toolset.discovered index={index} tools={names}" )
    return discovered

async def close_toolsets( toolsets: list[ McpToolset ] ) -> None:
    """Shut every server's subprocess down cleanly."""
    
    for index, toolset in enumerate( toolsets ):
        try:
            await toolset.close()
            log.info( f"toolset.closed index={index}" )
        except Exception:
            log.exception( f"toolset.closed.error index={index}" )

def select_available_servers( specs: list[ McpServerSpec ] ) -> list[ McpServerSpec ]:
    """Split specs into available vs skipped via `enabled` + transport preflight."""
    
    available: list[ McpServerSpec ] = []
    skipped: list[ tuple[ McpServerSpec, str ] ] = []
    
    for spec in specs:
        if not spec.enabled:
            skipped.append( ( spec, "disabled" ) )
            continue
        problems = spec.transport.preflight()
        if problems:
            skipped.append( ( spec, "; ".join( problems ) ) )
        else:
            available.append( spec )
    
    for spec, reason in skipped:
        log.warning( f"server.skipped name={spec} reason={reason}" )
    
    if not available:
        raise SystemExit(
            "No MCP servers available. Check reason above."
        )
    return available

# ==============================================================================
# Pure gating
# ==============================================================================
def gating_from_discovery(
    specs: list[ McpServerSpec ], discovered: dict[ int, list[ str ] ]
) -> GatingSpec:
    """Compile (specs, discovered) into GatingSpec."""
    
    tool_to_server: dict[ str, str ] = {}
    runtime_to_raw: dict[ str, str ] = {}
    ambiguous: set[ str ] = set()
    bypass: set[ tuple[ str, str ] ] = set()
    
    for position, spec in enumerate( specs ):
        for raw in discovered.get( position, [] ):
            runtime_name = f"{spec.name}_{raw}"

            owner = tool_to_server.get( runtime_name )
            if owner is not None and owner != spec.name:
                ambiguous.add( runtime_name )
                log.warning(
                    f"confirmation.ambiguous tool={runtime_name} "
                    f"servers={owner, spec.name} -> forcing confirmation"
                )
                continue
            
            tool_to_server[ runtime_name ] = spec.name
            runtime_to_raw[ runtime_name ] = raw
            
            if raw in spec.bypass_confirmation:
                bypass.add( ( spec.name, runtime_name ) )
    
    return GatingSpec(
        bypass = frozenset( bypass ),
        tool_to_server = tool_to_server,
        runtime_to_raw = runtime_to_raw,
        ambiguous = frozenset( ambiguous )
    )
    
# ==============================================================================
# Provider
# ==============================================================================
class McpToolProvider( ToolProvisioning ):
    """Owns the MCP toolset lifecycle and provisioning.
    
    `setup` builds toolset and discovers tools; `teardown` closes them and must 
    run even if a turn raises. `gating` is only valid after setup.
    """
    
    def __init__( self, specs: list[ McpServerSpec ] ) -> None:
        self._specs = specs
        self._available: list[ McpServerSpec ] = []
        self._toolsets: list[ McpToolset ] = []
        self._discovered: dict[ int, list[ str ] ] | None = None
    
    async def setup( self ) -> None:
        """Preflight first so missing runtime is clean and early skip."""
        
        self._available = select_available_servers( self._specs )
        self._toolsets = build_toolsets( self._specs )
        self._discovered = await discover_tools( self._toolsets )
    
    async def teardown(self) -> None:
        await close_toolsets( self._toolsets )
        self._toolsets = []
    
    def tools( self ) -> list[ ToolUnion ]:
        return list( self._toolsets )

    def gating( self ) -> GatingSpec:
        if self._discovered is None:
            raise RuntimeError(
                "McpToolProvider.gating() called before setup(); discovered has not run."
            )
        return gating_from_discovery( self._specs, self._discovered )
        
    