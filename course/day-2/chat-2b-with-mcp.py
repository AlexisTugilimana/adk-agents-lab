"""
==============================================================================
chat-2b-with-mcp.py
==============================================================================

A small useful ADK agent: a terminal chat whose tools are supplied entirely by 
external MCP servers. The goal is to explore the following:

    1. McpToolset   - How ADK consumes an MCP server as a set of tools.
    2. Transport    - MCP servers can be reached over different transports. 
                      This version supports: stdio and streamable HTTP.
    2. Lifecycle    - MCP stdio servers are spawned subprocesses. They have to 
                      be discovered at startup and closed at shutdown.
    3. Discovery    - Tools are advertised by the server at the connect time and 
                      discovered dynamically.

Servers are described declaratively in the SERVERS registry. Earch server pairs 
a `Transport` (how to reach it) with server-level metadata (name, tool filter, 
timeout).

The bundled servers are:

    - everything: reference learning server (stdio, npx)
    - kaggle    : datasets / notebooks / comps (stdio, npx mcp-remote)
    - github    : PR / issue analysis (streamable HTTP, remote)

Authentication
--------------
Gemini is authenticated via Vertex AI and gcloud:

    - gcloud auth application-default login
    - export GOOGLE_CLOUD_PROJECT="your-project-id"
    - export GOOGLE_CLOUD_LOCATION="your-location"
    - export GOOGLE_GENAI_USE_VERTEXAI=TRUE

MCP server auth is per-server and independent of the model auth:

    - github: export GITHUB_MCP_TOKEN="ghp_..."
    - kaggle: interactive OAuth handled by mcp-remote on first connection.
    
Run it with: 

    uv run --with "google-adk-[mcp]" chat-2b-with-mcp.py
    uv run --with "google-adk-[mcp]" chat-2b-with-mcp.py --debug
    uv run --with "google-adk-[mcp]" chat-2b-with-mcpt.py --list-tools   # discover and exit
"""
import os
import time
import shutil
import asyncio
import logging
import sys
import argparse
import base64
from abc import ABC, abstractmethod
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
from mcp import StdioServerParameters
from google.adk.events import Event
from google.genai import types

# ==============================================================================
# Constants
# ==============================================================================
AGENT_NAME = "mcp_agent"
MODEL = "gemini-3.5-flash"
APP_NAME = "mcp-chat"
USER_ID = "user-0"
SESSION_ID = "session-0"
UNKNOWN_RESPONSE_NAME = "<unknown>"
ARTIFACT_DIR = "mcp-artifacts"

INSTRUCTIONS = (
    "You are a helpful assistant whose abilities come from connected MCP "
    "tools. Inspect the tools available to you and use them when they fit the "
    "user's request; otherwise answer directly. Be concise, and when you use "
    "a tool, briefly say what it returned."
)

# ==============================================================================
# Classes
# ==============================================================================
@dataclass
class ImageBlob:
    mime_type: str
    data: bytes
    
@dataclass
class ToolCall:
    name: str
    args: dict = field( default_factory = dict )

@dataclass
class ToolResult:
    name: str
    images: list[ ImageBlob ] = field( default_factory = list )

@dataclass
class Answer:
    text: str

Update = Union[ ToolCall, ToolResult, Answer ]

@dataclass
class TurnMetrics:
    """Per-turn operational telemetry"""
    
    invocation_id: str | None = None
    events: int = 0
    tool_calls: int = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    
# ==============================================================================
# Logger
# ==============================================================================
log = logging.getLogger( "mcp-chat" )

# ==============================================================================
# Transport
# ==============================================================================
# A transport answers 2 questions about the server:
#
#   1. How do I build the ADK connection params to reach it?    -> connection_params()
#   2. Is everything I need in place to reach it?               -> preflight()
ConnectionParams = Union[
    StdioConnectionParams,
    StreamableHTTPConnectionParams
]

class Transport( ABC ):
    """How to reach one MCP server."""
    
    @abstractmethod
    def connection_params( self, timeout: float ) -> ConnectionParams:
        """Build the ADK connection parameters for this transport."""
    
    @abstractmethod
    def preflight( self ) -> list[ str ]:
        """Return a list of human-readable problems. Empty list == ready."""
    
    @abstractmethod
    def describe( self ) -> str:
        """One-line description of the server."""
        
@dataclass
class StdioTransport( Transport ):
    """Launched server as a local subprocess speaking MCP over"""
    
    command: str
    args: list[ str ] = field( default_factory = list )
    
    def connection_params( self, timeout: float ) -> StdioConnectionParams:
        return StdioConnectionParams(
            server_params = StdioServerParameters(
                command = self.command,
                args = self.args
            ),
            timeout = timeout
        )
    
    def preflight( self ) -> list[ str ]:
        if shutil.which( self.command ) is None:
            hint = "(install Node.js for npx/node)" if { "npx", "node" } else ""
            return [ f"command {self.command!r} not found on PATH{hint}" ]
        return []
    
    def describe( self ) -> str:
        return f"stdio: {self.command} {' '.join( self.args )}".strip()
    
@dataclass
class StreamableHTTPTransport( Transport ):
    """Remote server reached over streamable HTTP."""
    
    url: str
    headers: dict[ str, str | None ] = field( default_factory = dict )
    
    def connection_params( self, timeout: float ) -> StreamableHTTPConnectionParams:
        return StreamableHTTPConnectionParams(
            url = self.url,
            headers = self._resolved_headers(),
            timeout = timeout
        )
    
    def _resolved_headers( self ) -> dict[ str, str ]:
        return { key: value for key, value in self.headers.items() if value is not None }

    def preflight( self ) -> list[ str ]:
        missing = [ key for key, value in self.headers.items() if value is None ]
        if missing:
            return [ f"unresolved header(s): {", ".join( missing )}" ]
        return []

    def describe( self ) -> str:
        return f"http: {self.url}"
    
# ==============================================================================
# Server registry
# ==============================================================================
@dataclass
class MCPServerSpec:
    """Declarative description of one stdio MCP server."""
    
    name: str
    transport: Transport
    tool_filter: list[ str ] | None = None
    timeout: float = 30.0
    enabled: bool = True
    
def _github_transport() -> StreamableHTTPTransport:
    """Specific Github transport implementation resolving bearer token env variable."""
    
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
            "X-MCP-Readonly": "true"
        }
    )
    
# List of available mcp servers with their specs. `tool_filter` is scope the 
# available tools to be used by the mcp.
SERVERS: list[ MCPServerSpec ] = [
    MCPServerSpec(
        name = "everything",
        transport = StdioTransport(
            command = "npx",
            args = [ "-y", "@modelcontextprotocol/server-everything" ]
        ),
        tool_filter = [ "echo", "get-sum", "get-tiny-image", "get-env" ],
        timeout = 30.0
    ),
    MCPServerSpec(
        name = "github",
        transport = _github_transport(),
        tool_filter = None,
        timeout = 30.0
    ),
]

# ==============================================================================
# MCP toolsets
# ==============================================================================
def build_toolsets( specs: list[ MCPServerSpec ] ) -> list[ McpToolset ]:
    """Contruct one McpToolset per spec.
    
    Constructing the toolsets does not yet spawn the subprocess; the connection is 
    established lazily on the first tool use/discovery.
    """
    
    toolsets: list[ McpToolset ] = []
    for spec in specs:
        toolset = McpToolset(
            connection_params = spec.transport.connection_params( spec.timeout ),
            tool_filter = spec.tool_filter
        )
        toolsets.append( toolset )
        log.info( f"toolset.built name={spec.name} via={spec.transport.describe()!r} filter={spec.tool_filter}" )
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
            log.exception( f"toolset.close.error index={index}" )
            
# ==============================================================================
# Event loop
# ==============================================================================
def extract_images( response: object ) -> list[ ImageBlob ]:
    """Extract images from MCP function response.
    
    The Everything server's get-tiny-image returns a content 
    like {"type": "image", "data": "<base64>", "mimeType": 
    image/png" }."""
    
    images: list[ ImageBlob ] = []
    content = response.get( "content", [] ) if isinstance( response, dict ) else []
    
    for item in content:
        if isinstance( item, dict ) and item.get( "type" ) == "image" and item.get( "data" ):
            try:
                images.append(
                    ImageBlob(
                        mime_type = item.get( "mimeType", "image/png" ),
                        data = base64.b64decode( item[ "data" ] )
                    )
                )
            except Exception:
                log.exception( "image.decode.error" )
    
    return images

def observe( event: Event ) -> list[ Update ]:
    """Translate one raw ADK event into zero or more semantic updates."""
    
    updates: list[ Update ] = []
    
    # Tool calls and responses
    for call in event.get_function_calls():
        updates.append( ToolCall( call.name or UNKNOWN_RESPONSE_NAME, dict( call.args or {} ) ) )
        
    # Tool results (including images if present)
    for resp in event.get_function_responses():
        images = extract_images( getattr( resp, "response", None ) )
        updates.append( ToolResult( resp.name or UNKNOWN_RESPONSE_NAME, images ) )
    
    # Final response
    if event.is_final_response():
        parts = ( event.content.parts or [] ) if event.content else []
        text = "".join( ( p.text or "" ) for p in parts if getattr( p, "text", None ) )
        if text: updates.append( Answer( text ) )
    
    return updates

async def stream_turn( 
    runner: Runner, *, user_id: str, session_id: str, text: str 
) -> AsyncIterator[ Update ]:
    """Run one turn and yield updates for presentation.
    
    Two separate output channels:
        1. `yield update`:  presentation data for the user.
        2. `log.*(...)`:    operational telemetry for observability.
    """
    
    message = types.Content( role = "user", parts = [ types.Part( text = text ) ] )
    metrics = TurnMetrics()
    
    log.info( f"turn.start user={user_id} session={session_id}" )
    start = time.perf_counter()
    
    try:
        async for event in runner.run_async(
            user_id = user_id, session_id = session_id, new_message = message 
        ):
            metrics.events += 1
            metrics.invocation_id = metrics.invocation_id or getattr( event, "invocation_id", None )
            
            calls = event.get_function_calls()
            metrics.tool_calls += len( calls )
            
            usage = getattr( event, "usage_metadata", None )
            if usage:
                metrics.prompt_tokens = getattr( usage, "prompt_token_count", None ) or metrics.prompt_tokens
                metrics.output_tokens = getattr( usage, "candidates_token_count", None ) or metrics.output_tokens
                metrics.total_tokens = getattr( usage, "total_token_count", None ) or metrics.total_tokens
            
            log.debug(
                f"event inv={metrics.invocation_id} author={event.author} "
                f"final={event.is_final_response()} tools={[c.name for c in calls] or None}"
            )
            
            for update in observe( event ):
                yield update
                
    except Exception:
        log.exception( f"turn.error inv={metrics.invocation_id} user={user_id} session={session_id}" )
        raise
    finally:
        elapsed_ms = ( time.perf_counter() - start ) * 1000
        log.info(
            f"turn.end inv={metrics.invocation_id} tool_calls={metrics.tool_calls} "
            f"latency={elapsed_ms} tokens(prompt={metrics.prompt_tokens} output={metrics.output_tokens} "
            f"total={metrics.total_tokens})"
        )

def present( update: Update ) -> None:
    """Single presentation layer that writes to the terminal."""
    
    if isinstance( update, ToolCall ):
        detail = ", ".join( f"{k}={v!r}" for k, v in update.args.items() )
        print( f"   \U0001F527 {update.name}({detail})" )
    elif isinstance( update, ToolResult ):
        print( f"   \u2713 {update.name} returned" )
        for blob in update.images:
            path = save_image( update.name, blob )
            print( f"   \U0001F5BC image saved -> {path} ({blob.mime_type}, {len( blob.data )} bytes)")
    elif isinstance( update, Answer ):
        print( f"agent \u25b8 {update.text}\n" )
        
def save_image( tool_name: str, blob: ImageBlob ) -> str:
    """Write an image blow under ARTIFACT_DIR and return its path."""
    
    os.makedirs( ARTIFACT_DIR, exist_ok = True )
    ext = blob.mime_type.split( "/" )[ -1 ] if "/" in blob.mime_type else "png"
    stamp = datetime.now().strftime( "%Y%m%d-%H%M%S-%f" )
    path = os.path.join( ARTIFACT_DIR, f"{tool_name}-{stamp}.{ext}" )
    with open( path, "wb" ) as handle:
        handle.write( blob.data )
    return path

# ==============================================================================
# MCP lifecycle and shared helpers
# ==============================================================================
@asynccontextmanager
async def mcp_toolsets( specs: list[ MCPServerSpec ] ) -> AsyncIterator[ list[ McpToolset ] ]:
    """Own the toolset lifecyle for any command: build on entry and guarantee
    teardown on exit.
    
    Should belong to the toolsets part of the code.
    """
    
    toolsets = build_toolsets( specs )
    try:
        yield toolsets
    finally:
        await close_toolsets( toolsets )
        
def print_servers( specs: list[ MCPServerSpec ], discovered: dict[ int, list[ str ] ] ) -> None:
    """Render discovered results."""
    
    print( "Connected MCP servers:" )
    for index, spec in enumerate( specs ):
        names = ", ".join( discovered.get( index, [] ) ) or "(no tools)"
        print( f"   \u2022 {spec.name}: {names}" )
    print()
    
async def chat_loop( runner: Runner, *, user_id: str, session_id: str ) -> None:
    """Simple REPL: read a line, stream the turn, present the update, repeat."""
    
    print( "MCP chat - type a request, or 'exit' to quit.\n" )
    print( "Try: 'echo Hello world', 'add 21 and 21', 'show me a tiny image'.\n" )
    
    while True:
        try:
            text = ( await asyncio.to_thread( input, "you \u25b8 " ) ).strip()
        except ( EOFError, KeyboardInterrupt ):
            print( "\nBye." )
            return
    
        if not text: continue
        
        if text.lower() in { "exit", "quit" }:
            print( "\nBye." )
            return

        # Consume the update stream and hand each one to the presenter.
        try:
            async for update in stream_turn(
                runner, user_id = user_id, session_id = session_id, text = text
            ):
                present( update )
        except Exception as exc:
            # Keep the chat alive if a turn fails
            print( f"agent \u25b8 (error: {exc})\n" )
            continue
        
# ==============================================================================
# Agent
# ==============================================================================
def build_agent( toolsets: list[ McpToolset ] ) -> Agent:
    return Agent(
        name = AGENT_NAME,
        model = MODEL,
        instruction = INSTRUCTIONS,
        tools = list( toolsets )
    )
    
# ==============================================================================
# Cmd: list-tools (connects to MCP, never touches the model)
# ==============================================================================
async def list_tools() -> None:
    """Connect, report each server's advertised tools, and exit."""
    
    specs = select_available_servers( SERVERS )
    async with mcp_toolsets( specs ) as toolsets:
        discovered = await discover_tools( toolsets )
        print_servers( specs, discovered )
        
# ==============================================================================
# Cmd: chat (connect to MCP and drives the model)
# ==============================================================================
async def chat() -> None:
    """Run the interactive chat. Needs first to check credentials and MCP 
    runtimes.
    
    """
    print( f"[auth] using {check_auth()}" )
    specs = select_available_servers( SERVERS )
    
    async with mcp_toolsets( specs ) as toolsets:
        discovered = await discover_tools( toolsets )
        print_servers( specs, discovered )
        
        session_service = InMemorySessionService()
        agent = build_agent( toolsets )
        runner = Runner( app_name = APP_NAME, agent = agent, session_service = session_service )
        log.info( f"Startup app={APP_NAME} agent={agent.name} model={MODEL}" )
        
        user_id, session_id = USER_ID, SESSION_ID
        await session_service.create_session(
            app_name = APP_NAME, user_id = user_id, session_id = session_id
        )
        log.info( f"session.ready user={user_id} session={session_id}" )
        
        await chat_loop( runner, user_id = user_id, session_id = session_id )

# ==============================================================================
# Preflight (environment + credentials)
# ==============================================================================
def check_auth() -> str:
    """Validate credentials and return a short human label of what we'll use."""
    
    use_vertex = os.environ.get( "GOOGLE_GENAI_USE_VERTEXAI", "").lower() in { "1", "true", "yes" }

    if use_vertex:
        if not os.environ.get( "GOOGLE_CLOUD_PROJECT" ):
            raise SystemExit(
                "GOOGLE_CLOUD_PROJECT is not set.\n"
                "   export GOOGLE_CLOUD_PROJECT=your-project-id\n"
                "   export GOOGLE_CLOUD_LOCATION=your-location\n"
                "and make sure your run: gcloud auth application default login"
            )
        if not os.environ.get( "GOOGLE_CLOUD_LOCATION" ):
            raise SystemExit(
                "GOOGLE_CLOUD_LOCATION is not set.\n"
                "   export GOOGLE_CLOUD_PROJECT=your-project-id\n"
                "   export GOOGLE_CLOUD_LOCATION=your-location\n"
                "and make sure your run: gcloud auth application default login"
            )
        return (
            f"Vertex AI (project={os.environ['GOOGLE_CLOUD_PROJECT']}), "
            f"location={os.environ['GOOGLE_CLOUD_LOCATION']}"
        )
        
    raise SystemExit(
        "No credentials found. Run the following commands to get your credentials:\n"
        "   gcloud auth application default login\n"
        "   export GOOGLE_GENAI_USE_VERTEXAI=TRUE\n"
        "   export GOOGLE_CLOUD_PROJECT=your-project-id\n"
        "   export GOOGLE_CLOUD_LOCATION=your-location\n"
    )

def select_available_servers( specs: list[ MCPServerSpec ] ) -> list[ MCPServerSpec ]:
    """Split servers spec into available vs skipped via:
        - enabled option
        - preflight check
    """
    
    available: list[ MCPServerSpec ] = []
    skipped: list[ tuple[ MCPServerSpec, str ] ] = []
    
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
        log.warning( f"server.skipped name={spec.name} reason={reason}" )
        print( f"   \u26a0 skipping {spec.name}: {reason}" )
    
    if not available:
        raise SystemExit(
            "No MCP servers are available. Check the reasons above "
            "(missing Node.js for npx, missing GITHUB_MCP_TOKEN, etc.)."
        )
        
    return available
        
# ==============================================================================
# Logger
# ==============================================================================
def configure_logging( debug: bool ) -> None:
    """Send operational logs to STDERR. DEBUG adds a line per event; INFO keeps 
    it to startup, session, and per-turn summaries.
    """
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt = "%H:%M:%S",
        stream = sys.stderr
    )
    
    log.setLevel( logging.DEBUG if debug else logging.INFO )
    
    
# ==============================================================================
# Entry point
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser( description = "Minimal ADK search chat backed by MCP servers." )
    parser.add_argument(
        "--debug", action = "store_true", help = "Verbose logging: one line per raw event in the loop."
    )
    parser.add_argument(
        "--list-tools", action = "store_true", help = "Connect, princ each server's advertised tools, then exit."
    )
    args = parser.parse_args()
    
    configure_logging( args.debug )
    cmd = list_tools if args.list_tools else chat
    asyncio.run( cmd() )
    
if __name__ == "__main__":
    main()