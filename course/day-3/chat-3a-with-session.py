"""
==============================================================================
chat-3a-with-session.py
==============================================================================

A small useful ADK agent: a terminal chat whose tools are supplied entirely by 
external MCP servers. The goal is to explore the following:

    1. McpToolset   - How ADK consumes an MCP server as a set of tools.
    2. Transport    - MCP servers can be reached over different transports. 
                      This version supports: stdio and streamable HTTP.
    3. Lifecycle    - MCP stdio servers are spawned subprocesses. They have to 
                      be discovered at startup and closed at shutdown.
    4. Discovery    - Tools are advertised by the server at the connect time and 
                      discovered dynamically.
    5. Confirmation - Long-running operations: pause selected MCP tools and ask 
                      the human for approval.
    6. Sessions     - where conversation history (events + state) lives. Two
                      interchangeable backends can be selected:
                      * memory -> InMemorySessionService (fast, ephemeral)
                      * sqlite -> DatabaseSessionService (persistent, resumable)

Sessions (short-term memory)
----------------------------
ADK's `SessionService` is the storage layer that remembers conversation as an 
order of list of `Events` plus a `State` scratchpad, and the `Runner` replays 
that history on every turn. This file lets you picke the backend:

    --session-backend memory    (default) history lives in RAM; gone on exit.
    --session-backend sqlite              history persists to a SQlite file.

With SQlite, rerunning with the same session-id resumes the prior conversation 
instead of starting fresh.

Auditing persistent sessions
----------------------------
`--audit` opens the SQlite database read-only and prints a report:
which apps/users/sessions exist, how many event each hold, per-session timeline, 
and the shared user:/app: state.

MCP
---
Servers are described declaratively in the SERVERS registry. Earch server pairs 
a `Transport` (how to reach it) with server-level metadata (name, tool filter, 
timeout). It also provides a `requires_confirmation` boolean that says whether 
the server's tools must be approved by a humain before they execute.

The bundled servers are:

    - everything: reference learning server (stdio, npx)
    - kaggle    : datasets / notebooks / comps (stdio, npx mcp-remote)
    - github    : PR / issue analysis (streamable HTTP, remote)

Long-running operation
----------------------
A humain-in-the-loop strategy (HIL) is implemented by calling 
`tool_context.request_confirmation()` in a `before_tool_callback` such that it 
intercepts in a general fashion tools before their execution. It checks whether 
a tool is gated or not and, if so, call `request_confirmation()` on the tool's 
behalf. The MCP server is therefore never touched.

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
    
Install (one time). The `mcp` extra pulls the MCP toolset stack; the `db` extra 
pulls SQLALchemy for DatabaseSessionService; `aiosqlite` is the async SQlite 
driver ADK's engine requires:

    uv add "google-adk[mcp,db]" aiosqlite
    
Run it with: 

    # In-memory (default)
    uv run chat-mcp-confirm.py
    
    # Persistent sqlite session
    uv run chat-mcp-confirm.py --session-backend sqlite --session-id alice-001
    
    # Audit
    uv run chat-mcp-confirm.py --audit --session-backend sqlite --session-id alice-001
    
    # Utility flags
    uv run chat-mcp-confirm.py --debut      # verbose logging
    uv run chat-mcp-confirm.py --list-tools # discover tools and exit
"""
# ==============================================================================
# Quiet ADK's [EXPERIMENTAL] feature user's warning when not in debug mode
# ==============================================================================
# ADK announces every opt-in experimental features via `warnings.warn(...)`. Some 
# of them fires at import time, other during execution.
import sys
import os
import time
import shutil
import asyncio
import logging
import sys
import argparse
import base64
import sqlite3
import json
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Union, Callable, Awaitable, Iterator

# ==============================================================================
# Route warnings into logging
# ==============================================================================
# ADK announces every opt-in experimental features via `warnings.warn(...)`, some 
# of which fire at the import time. They are deliberitaly kept visible on every turn 
# for observability. `capture_warnings(True)` funnels them through the logging system 
# as `py.warnings` records instead of stderr text.
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# # Set the handler to avoid silently loose the warnings. Baseline set to WARNING.
# logging.basicConfig( level = logging.WARNING, format = LOG_FORMAT, datefmt = DATE_FORMAT, stream = sys.stderr )
# logging.captureWarnings( True )

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, BaseSessionService, DatabaseSessionService, Session
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams
from google.adk.tools import BaseTool, ToolContext
from google.adk.apps import App
from mcp import StdioServerParameters
from google.adk.events import Event
from google.adk.apps.app import App, ResumabilityConfig
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

DEFAULT_DB_URL = "sqlite+aiosqlite:///mcp_chat_session_db"

CONFIRMATION_FUNCTION_NAME = "adk_request_confirmation"
CONFIRMATION_HINT = "This tool requires your approval before it runs."

INSTRUCTIONS = (
    "You are a helpful assistant whose abilities come from connected MCP "
    "tools. Inspect the tools available to you and use them when they fit the "
    "user's request; otherwise answer directly. Be concise, and when you use "
    "a tool, briefly say what it returned. Some tools require human approval "
    "before they run; if an approval is rejected, report that plainly and do "
    "not retry the same action."
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
    
@dataclass
class ApprovalRequest:
    """The agent has paused: a gated tool wants to run and needs a humain yes/no."""
    
    tool_name: str
    hint: str
    payload: dict
    approval_id: str
    invocation_id: str
    
@dataclass
class ApprovalDecision:
    """Human's verdict on an ApprovalRequest."""
    
    tool_name: str
    approved: bool
    
Update = Union[ ToolCall, ToolResult, Answer, ApprovalRequest, ApprovalDecision ]

@dataclass
class TurnMetrics:
    """Per-turn operational telemetry"""
    
    invocation_id: str | None = None
    events: int = 0
    tool_calls: int = 0
    approvals: int = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    
# Approver signature injected into the turn.
Approver = Callable[ [ ApprovalRequest ], Awaitable[ bool ] ]

# Before callback signature
BeforeToolCallback = Callable[ [ BaseTool, dict, ToolContext ], dict | None ]

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
    requires_confirmation: bool = True
    
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
        timeout = 30.0,
        requires_confirmation = True
    ),
    MCPServerSpec(
        name = "github",
        transport = _github_transport(),
        tool_filter = None,
        timeout = 30.0,
        requires_confirmation = True
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
# Confirmation wiring
# ==============================================================================
def gated_tool_names( specs: list[ MCPServerSpec ], discovered: dict[ int, list[ str ] ] ) -> set[ str ]:
    """Collect tool names that require user's approval."""
    
    gated: set[ str ] = set()
    for position, spec in enumerate( specs ):
        if not spec.requires_confirmation:
            continue
        gated.update( discovered.get( position, [] ) )
    if gated:
        log.info( f"confirmation.gated_tools={sorted( gated )}" )
    return gated

def make_confirmation_callback( gated: set[ str ] ) -> BeforeToolCallback:
    """Build the `before_tool_callback` that enforces user's approval on gated tools."""
    
    def before_tool_callback( tool: BaseTool, args: dict, tool_context: ToolContext ) -> dict | None:
        if tool.name not in gated:
            return None
    
        confirmation = tool_context.tool_confirmation
        if confirmation is None:
            tool_context.request_confirmation(
                hint = CONFIRMATION_HINT,
                payload = { "tool": tool.name, "args": args or [] }
            )
            log.info( f"confirmation.requested tool={tool.name!r}" )
            return { "status": "pending", "message": f"Awaiting approval for {tool.name}." }

        if confirmation.confirmed:
            log.info( f"confirmation.approved tool={tool.name}" )
            return None
        
        log.info( f"confirmation.rejected tool={tool.name!r}" )
        return { "status": "rejected", "message": f"{tool.name} was not approved; skipped." }
    
    return before_tool_callback

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

def parse_confirmation_args( args: dict ) -> tuple[ str, dict ]:
    """Returns hint and payload from arguments of the function returned 
    by the agent.
    
    """
    
    DEFAULT_HINT = "Approve this actions?"
    
    if not isinstance( args, dict ): 
        return ( DEFAULT_HINT, {} )
    
    block = (
        args.get( "toolConfirmation" )
        or args.get( "tool_confirmation" )
        or args
    )
    block = block if isinstance( block, dict ) else {}
    
    hint = block.get( "hint" ) or args.get( "hint" ) or DEFAULT_HINT
    payload = block.get( "payload" ) or args.get( "payload" ) or {}
    payload = payload if isinstance( payload, dict ) else { "value": payload }
    return ( hint, payload )

def extract_approval( event: Event ) -> ApprovalRequest | None:
    """If the event is a pause (with adk_request_confirmation), return the 
    ApprovalRequest instance requires to present it, and later resume it.
    """
    
    for call in event.get_function_calls():
        if call.name == CONFIRMATION_FUNCTION_NAME:
            hint, payload = parse_confirmation_args( call.args or {} )
            return ApprovalRequest(
                tool_name = payload.get( "tool", "" ) or UNKNOWN_RESPONSE_NAME,
                hint = hint,
                payload = payload,
                approval_id = call.id or "",
                invocation_id = event.invocation_id
            )
    
    return None
    
def observe( event: Event ) -> list[ Update ]:
    """Translate one raw ADK event into zero or more semantic updates."""
    
    updates: list[ Update ] = []
    
    # Tool calls and responses (skip confirmation dialog)
    for call in event.get_function_calls():
        if call.name == CONFIRMATION_FUNCTION_NAME:
            continue
        updates.append( ToolCall( call.name or UNKNOWN_RESPONSE_NAME, dict( call.args or {} ) ) )
        
    # Tool results (including images if present but skip confirmation return)
    for resp in event.get_function_responses():
        if resp.name == CONFIRMATION_FUNCTION_NAME:
            continue
        images = extract_images( getattr( resp, "response", None ) )
        updates.append( ToolResult( resp.name or UNKNOWN_RESPONSE_NAME, images ) )
    
    # Final response
    if event.is_final_response():
        parts = ( event.content.parts or [] ) if event.content else []
        text = "".join( ( p.text or "" ) for p in parts if getattr( p, "text", None ) )
        if text: updates.append( Answer( text ) )
    
    return updates

def create_approval_response( request: ApprovalRequest, approved: bool ) -> types.Content:
    """Format the human decision as FunctionResponse ADK.
    
    The id + name must match the original confirmation request.
    """
    
    confirmation_request = types.FunctionResponse(
        id = request.approval_id,
        name = CONFIRMATION_FUNCTION_NAME,
        response = { "confirmed": approved }
    )
    
    return types.Content(
        role = "user",
        parts = [ types.Part( function_response = confirmation_request ) ]
    )
    
# ==============================================================================
# Turn streaming
# ==============================================================================
async def stream_turn( 
    runner: Runner, *, 
    user_id: str, 
    session_id: str, 
    text: str, 
    approver: Approver
) -> AsyncIterator[ Update ]:
    """Run one turn and yield updates for presentation.
    
    The turn is a loop because it may pause one or more times for approval:
    
        run -> detect adk_request_confirmation? 
               -> no --> Done
               -> yes --> ask human --> resume -> run again

    Resuming requires passing the same `invocation_id` so the runner continues the 
    paused execution instead of starting a new one.
    
    Two separate output channels:
        1. `yield update`:  presentation data for the user.
        2. `log.*(...)`:    operational telemetry for observability.
    """
    
    metrics = TurnMetrics()
    log.info( f"turn.start user={user_id} session={session_id}" )
    start = time.perf_counter()
    
    message = types.Content( role = "user", parts = [ types.Part( text = text ) ] )
    
    # First turn send the user's message so the resume invocation ID is None.
    # After that, if a user's confirmation is needed, the same invocation ID will be used.
    resume_invocation_id: str | None = None
    
    try:
        while True:
            
            pending: ApprovalRequest | None = None
            
            run_kwargs = {
                "user_id": user_id,
                "session_id": session_id,
                "new_message": message
            }
            if resume_invocation_id is not None:
                run_kwargs[ "invocation_id" ] = resume_invocation_id
                
            async for event in runner.run_async( **run_kwargs ):
                metrics.events += 1
                metrics.invocation_id = metrics.invocation_id or getattr( event, "invocation_id", None )
                
                calls = event.get_function_calls()
                metrics.tool_calls += len( [ c.name for c in calls if c.name != CONFIRMATION_FUNCTION_NAME ] )
                
                usage = getattr( event, "usage_metadata", None )
                if usage:
                    metrics.prompt_tokens = getattr( usage, "prompt_token_count", None ) or metrics.prompt_tokens
                    metrics.output_tokens = getattr( usage, "candidates_token_count", None ) or metrics.output_tokens
                    metrics.total_tokens = getattr( usage, "total_token_count", None ) or metrics.total_tokens
                
                log.debug(
                    f"event inv={metrics.invocation_id} author={event.author} "
                    f"final={event.is_final_response()} tools={[c.name for c in calls] or None}"
                )
                
                # Remember last pause seen to avoid overwriting pending
                pending = extract_approval( event ) or pending
                
                # Yield event updates but skip confirmation dialog and return (handled separately after)
                for update in observe( event ):
                    yield update
                
            # The turn is complete without user's confirmation
            if pending is None:
                break
            
            metrics.approvals += 1
            yield pending
            approved = await approver( pending )
            yield ApprovalDecision( pending.tool_name, approved )
            
            message = create_approval_response( pending, approved )
            resume_invocation_id = pending.invocation_id
            
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

# ==============================================================================
# Presentation
# ==============================================================================
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
    elif isinstance( update, ApprovalRequest ):
        print( f"   \u23F8 approval required for {update.tool_name}" )
        print( f"       {update.hint}" )
        if update.payload:
            print( f"   args:{update.payload['args']}" )
    elif isinstance( update, ApprovalDecision ):
        mark = "\u2705 approved" if update.approved else "\u274c rejected"
        print( f"   {mark}: {update.tool_name}" )
    elif isinstance( update, Answer ):
        print( f"agent \u25b8 {update.text}\n" )
        
async def terminal_approver( request: ApprovalRequest ) -> bool:
    """Gather human decision from the terminal.
    
    Focus on the input itself.
    Default is NO."""
    
    prompt = f"     approve {request.tool_name}? [y/N] \u25b8 "
    try:
        answer = ( await asyncio.to_thread( input, prompt ) ).strip().lower()
    except ( EOFError, KeyboardInterrupt ):
        print()
        return False
    return answer in { "y", "yes" }

# TODO: move this to the right place
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
    
async def chat_loop(
    runner: Runner, *, 
    user_id: str, 
    session_id: str,
    approver: Approver
) -> None:
    """Simple REPL: read a line, stream the turn, present the update, repeat."""
    
    print( "MCP chat - type a request, or 'exit' to quit.\n" )
    print( "Try: 'echo Hello world', 'add 21 and 21', 'show me a tiny image'.\n" )
    print( "Gated demo: 'echo hello world' -> pauses for approval if everythng server is gated.\n" )
    
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
                runner, 
                user_id = user_id, 
                session_id = session_id, 
                text = text,
                approver = approver
            ):
                present( update )
        except Exception as exc:
            # Keep the chat alive if a turn fails
            print( f"agent \u25b8 (error: {exc})\n" )
            continue

# ==============================================================================
# Session backends
# ==============================================================================
class SessionBackend( str, Enum ):
    """Selectable session storage backends."""
    
    MEMORY = "memory"
    SQLITE = "sqlite"

@dataclass
class SessionConfig:
    """Everything needed to open the right session for a single turn."""
    
    backend: SessionBackend
    db_url: str
    app_name: str
    user_id: str
    session_id: str
    
def normalize_sqlite_url( url: str ) -> str:
    """Be sure the SQLite URL uses the async `aiosqlite` driver that ADK requires."""
    
    if url.startswith( "sqlite://" ) and "+aiosqlite" not in url:
        fixed = url.replace( "sqlite://", "sqlite+aiosqlite://", 1 )
        log.warning( f"session.db_url.normalized from={url!r} to={fixed!r} (async driver required)" )
        return fixed
    return url

def build_session_service( config: SessionConfig ) -> tuple[ BaseSessionService, str ]:
    """Construct the session service for the choosen backend.
    
    Returns the service + a short human label for logging/printing. This is the 
    only place that references concrete session classes.
    """
    
    if config.backend is SessionBackend.MEMORY:
        return InMemorySessionService(), "in-memory (ephemeral; history is lost on exit)"
    
    url = normalize_sqlite_url( config.db_url )
    service = DatabaseSessionService( db_url = url )
    path = sqlite_path_from_url( url ) or url
    return service, f"sqlite -> {path} (persistent; resume with the same --session-id)"

async def open_session(
    service: BaseSessionService, *,
    app_name: str,
    user_id: str,
    session_id: str
) -> tuple[ Session, bool ]:
    """Resume an existing session or create a new one. Return (session, create)."""
    
    existing = await service.get_session(
        app_name = app_name, user_id = user_id, session_id = session_id
    )
    
    if existing is not None:
        return ( existing, False )
    
    created = await service.create_session(
        app_name = app_name, user_id = user_id, session_id = session_id
    )
    return ( created, True )

# ==============================================================================
# Session audit (read-only inspection of persistent SQLite store)
# ==============================================================================
def sqlite_path_from_url( url: str ) -> str | None:
    """Extract the filesystem path from a `sqlite[+driver]:///path` URL.
    
    Returns None for non-sqlite database URLs and for the special in-memory database.
    """
    
    if "sqlite" not in url:
        return None
    
    tail = url.split( ":///", 1 )[ -1 ] if ":///" in url else ""
    if not tail or tail == ":memory:":
        return None
    return tail
        
@dataclass
class SessionSummary:
    """One row of the session overview."""
    
    app_name: str
    user_id: str
    session_id: str
    events: int
    first_ts: str | None
    last_ts: str | None
    
def _safe_json( raw: object ) -> dict | None:
    """Parse JSON column value into a dict."""
    
    if not isinstance( raw, ( str, bytes, bytearray ) ):
        return None
    
    try:
        parsed = json.loads( raw )
    except ( ValueError, TypeError ):
        return None
    return parsed if isinstance( parsed, dict ) else None

def _kv( args: dict ) -> str:
    """Compact, redaction-friendly rendering of tool arguments in the timeline."""
    
    if not isinstance( args, dict ):
        return ""
    return ", ".join( f"{key}={str( value )[:40]!r}" for key, value in args.items() )

def summarize_content( content: dict | None ) -> tuple[ str, str ]:
    """Classify one event's content into (kind, snippet) for a compact timeline."""
    
    if content is None:
        return ( "turn-end", "" )
    if not isinstance( content, dict ):
        return ( "empty", "" )
    
    for part in content.get( "parts" ) or []:
        if not isinstance( part, dict ):
            continue
        
        if part.get( "text" ):
            return ( "text", " ".join( part[ "text" ].split() if isinstance( part[ "text" ], str ) else "" )[ :120 ] )
        
        call = part.get( "function_call" ) or part.get( "functionCall" )
        if call:
            name = call.get( "name", "?" )
            if name == CONFIRMATION_FUNCTION_NAME:
                return ( "confirm-request", name )
            return ( "tool_call", f"{name}({_kv( call.get( "args" ) or {} )})")
        
        response = part.get( "function_response" ) or part.get( "functionResponse" )
        if response:
            name = response.get( "name", "?" )
            kind = "confirm_reply" if name == CONFIRMATION_FUNCTION_NAME else "tool_result"
            return ( kind, name )
        
    return ( "other", "" )

class SessionAuditor:
    """Read-only reporter over an ADK SQLite session database."""
    
    def __init__( self, db_path: str ) -> None:
        self.db_path = db_path
        
    @contextmanager
    def _connect( self ) -> Iterator[ sqlite3.Connection ]:
        
        # Read-only mode
        uri = f"file:{Path( self.db_path ).as_posix()}?mode=ro"
        connection = sqlite3.connect( uri, uri = True )
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()
        
    def _tables( self, connection: sqlite3.Connection ) -> set[ str ]:
        rows = connection.execute( "SELECT name from sqlite_master WHERE type='table'" )
        return { row[ 0 ] for row in rows }
    
    def _columns( self, connection: sqlite3.Connection, table: str ) -> list[ str ]:
        try:
            return [ row[ 1 ] for row in connection.execute( f'PRAGMA table_info("{table}")' ) ] 
        except sqlite3.Error:
            return []
        
    def _event_records( self, connection: sqlite3.Connection ) -> list[ dict ]:
        """Normalize every event row into a uniform dict regardless of schema."""
        
        columns = set( self._columns( connection, "events") )
        if not columns:
            return []
        
        rows = connection.execute( "SELECT * FROM events" ).fetchall()
        records: list[ dict ] = []
        for row in rows:
            keys = row.keys()
            record = {
                "app_name": row[ "app_name" ] if "app_name" in keys else "",
                "user_id": row[ "user_id" ] if "user_id" in keys else "",
                "session_id": row[ "session_id" ] if "session_id" in keys else "",
                "timestamp": str( row[ "timestamp" ] ) if "timestamp" in keys else ""
            }
            
            
            if "event_data" in keys:  # Current ADK: one JSON blob
                data = _safe_json( row[ "event_data" ] ) or {}
                record[ "author" ] = data.get( "author" )
                record[ "content" ] = data.get( "content" )
            else:                       # legacy ADK / notebook schema
                record[ "author" ] = row[ "author" ] if "author" in keys else None
                record[ "content" ] = _safe_json( row[ "content" ] ) if "content" in keys else None
            
            records.append( record )
        
        records.sort( key = lambda r: r[ "timestamp" ] or "" )
        return records
                
    def summaries( self ) -> list[ SessionSummary ]:
        """One SessionSummary per (app, user, session) with event counts + span."""
        
        with self._connect() as connection:
            if "sessions" not in self._tables( connection ):
                return []
            
            events = self._event_records( connection )
            by_key: dict[ tuple[ str, str, str ], list[ dict ] ] = {}
            for record in events:
                key = ( record[ "app_name" ], record[ "user_id" ], record[ "session_id" ] )
                by_key.setdefault( key, [] ).append( record )

            summaries: list[ SessionSummary ] = []
            for row in connection.execute( "SELECT app_name, user_id, id FROM sessions" ):
                key = ( row[ "app_name" ], row[ "user_id" ], row[ "id"] )
                bucket = by_key.get( key, [] )
                timestamps = [ record[ "timestamp" ] for record in bucket if record[ "timestamp" ] ]
                summaries.append(
                    SessionSummary(
                        app_name = row[ "app_name" ],
                        user_id = row[ "user_id" ],
                        session_id = row[ "id" ],
                        events = len( bucket ),
                        first_ts = min( timestamps ) if timestamps else None,
                        last_ts = max( timestamps ) if timestamps else None
                    )
                )
                
            summaries.sort( key = lambda s: ( s.last_ts or "", s.session_id ), reverse = True )
            return summaries
        
    def timeline( self, session_id: str ) -> list[ dict ]:
        """Chronological events for one session."""
        
        with self._connect() as connection:
            events = [ r for r in self._event_records( connection ) if r[ "session_id" ] == session_id ]
        
        timeline: list[ dict ] = []
        
        for record in events:
            kind, snippet = summarize_content( record[ "content" ] )
            timeline.append(
                {
                    "timestamp": record[ "timestamp" ],
                    "author": record[ "author" ] or "?",
                    "kind": kind,
                    "snippet": snippet
                }
            )
        
        return timeline
    
    def shared_state( self ) -> tuple[ list[ dict ], list[ dict ] ]:
        """Return (app_states, user_states) rows."""
        
        def read( connection: sqlite3.Connection, table: str ) -> list[ dict ]:
            rows = connection.execute( f'SELECT * FROM "{table}"' ).fetchall()
            out: list[ dict ] = []
            for row in rows:
                keys = row.keys()
                out.append(
                    {
                        "app_name": row[ "app_name" ] if "app_name" in keys else "",
                        "user_id": row[ "user_id" ] if "user_id" in keys else "",
                        "state": ( _safe_json( row[ "state" ] ) or {} ) if "state" in keys else {}
                    }
                )
            return out
        
        with self._connect() as connection:
            tables = self._tables( connection )
            apps = read( connection, "app_states" ) if "app_states" in tables else []
            users = read( connection, "user_state" ) if "user_state" in tables else []
            return apps, users
        
def print_audit_report( config: SessionConfig ) -> None:
    """Render a read-only audit of the configured SQLite database."""
    
    if config.backend is not SessionBackend.SQLITE:
        raise SystemExit(
            "--audit only applies to sqlite backend. "
            "Re-run with: --session-backend sqlite [--db-url ...]"
        )
    
    path = sqlite_path_from_url( normalize_sqlite_url( config.db_url ) )
    if path is None:
        raise SystemExit( f"Could not derive a SQLite file path from --db-url {config.db_url}" )
    if not Path( path ).exists():
        raise SystemExit(
            f"No database at {path!r} yet. Run a chat with "
            f"--session-backend sqlite first to create it."
        )
    
    auditor = SessionAuditor( path )
    size_kb = Path( path ).stat().st_size / 1024
    
    print( "\n" + "=" * 70 )
    print( f"Session audit -> {path} ({size_kb:.1f} KiB)" )
    print( "\n" + "=" * 70 )
    
    # 1. Session overview
    summaries = auditor.summaries()
    print( f"\nSessions ({len( summaries )})")
    if not summaries:
        print( "    (none recorded yet)" )
    for summary in summaries:
        span = f"{summary.first_ts} -> {summary.last_ts}" if summary.first_ts else "no events"
        print(
            f"  \u2022 {summary.session_id} "
            f"[app_name={summary.app_name} user={summary.user_id}] "
            f"events={summary.events} {span}"
        )
        
    # 2. Cross-session shared state( the 'app:' / 'user:' scratchpads)
    apps, users = auditor.shared_state()
    if apps or users:
        print( "\nShared state (across sessions):" )
        for entry in apps:
            print( f"   app={entry[ "app_name" ]}   app=* -> {entry["state"] or '{}'}" )
        for entry in users:
            print( f"   app={entry[ "app_name" ]}   user={entry["user_id"]} -> {entry["state"] or '{}'}" )
    
    # 3. Detailed timeline for the requested session
    target = config.session_id
    known = { summary.session_id for summary in summaries }
    if target in known:
        print( f"\nTimeline for session {target!r}:" )
        for entry in auditor.timeline( target ):
            detail = f" - {entry["snippet"] if entry["snippet"] else ""}"
            print( f"   [{entry["timestamp"]}   {entry["author"]:<10}   [{entry["kind"]}]{detail}" )
    elif summaries:
        print(
            f"\nNo session {target!r} found. "
            f"Tip: pass --session-id <id> to see a full timeline, e.g. "
            f"--audit --session-id {summaries[0].session_id}"
        )
    print()
    
async def delete_session_command( config: SessionConfig ) -> None:
    """Delete the session named by --session-id from the persistent store, then exit."""
    
    if config.backend is not SessionBackend.SQLITE:
        raise SystemExit(
            "--delete-session only applies to the sqlite backend "
            "(in-memory sessions vanish on exit). "
            "re-run with: --session-backend sqlite [--db-url ...] --session-id <id>"
        )
        
    path = sqlite_path_from_url( normalize_sqlite_url( config.db_url ) )
    if path is None:
        raise SystemExit( f"Could not derive SQLite file path from --db-url {config.db_url!r}." )
    if not Path( path ).exists():
        raise SystemExit( f"No database at {path!r}; nothing to delete." )
    
    service, _ = build_session_service( config )
    existing = await service.get_session(
        app_name = config.app_name,
        user_id = config.user_id,
        session_id = config.session_id
    )
    if existing is None:
        print(
            f"No session {config.session_id!r} for user {config.user_id!r} in {path}. "
            f"Nothing to delete."
        )
        return

    events = len( getattr( existing, "events", [] ) or [] )
    print( f"About to delete session {config.session_id!r} (user={config.user_id}, events={events}) from {path}" )
    
    try:
        reply = input( "Delete permanently? [y/N]" ).strip().lower()
    except EOFError:
        reply = ""
    
    if reply not in ( "yes", "y" ):
        print( "Aborted; nothing deleted." )
        return
    
    await service.delete_session(
        app_name = config.app_name,
        user_id = config.user_id,
        session_id = config.session_id
    )
    print( f"Deleted session {config.session_id!r}." )
    
# ==============================================================================
# Agent + Resumable app
# ==============================================================================
def build_agent( toolsets: list[ McpToolset ], before_tool_callback: BeforeToolCallback | None = None ) -> Agent:
    return Agent(
        name = AGENT_NAME,
        model = MODEL,
        instruction = INSTRUCTIONS,
        tools = list( toolsets ),
        before_tool_callback = before_tool_callback
    )

def build_app( agent: Agent ) -> App:
    """Wrap the agent in resumable app."""
    
    return App(
        name = APP_NAME,
        root_agent = agent,
        resumability_config = ResumabilityConfig( is_resumable = True )
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
async def chat( config: SessionConfig ) -> None:
    """Run the interactive chat. Needs first to check credentials and MCP 
    runtimes.
    
    The session backend is chosen by `config`. The rest is session-agnostic.
    """
    
    print( f"[auth] using {check_auth()}" )
    specs = select_available_servers( SERVERS )
    
    async with mcp_toolsets( specs ) as toolsets:
        discovered = await discover_tools( toolsets )
        print_servers( specs, discovered )
        
        # Collect gated tools and wire them to the agent's callback
        gated = gated_tool_names( specs, discovered )
        before_tool_callback: BeforeToolCallback = make_confirmation_callback( gated )
        
        # Pick the session backend
        session_service, backend_label = build_session_service( config )
        print( f"[session] {backend_label}" )
        log.info( f"session.backend={config.backend.value} {backend_label}" )
        
        agent = build_agent( toolsets, before_tool_callback )
        app = build_app( agent )
        runner = Runner( app = app, session_service = session_service )
        log.info( f"Startup app={APP_NAME} agent={agent.name} model={MODEL}" )
        
        user_id, session_id = config.user_id, config.session_id
        session, created = await open_session(
            service = session_service,
            app_name = config.app_name,
            user_id = user_id,
            session_id = session_id 
        )
        state = "created" if created else "resumed"
        replayed = 0 if created else len( getattr( session, "events", [] ) or [] )
        print( f"[session] {state} {session_id!r} (user={user_id}, prior_events={replayed})" )
        log.info( f"session.ready state={state} user={user_id} session={session_id} prior_events={replayed})" )
        
        await chat_loop( runner, user_id = user_id, session_id = session_id, approver = terminal_approver )

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
def configure_logging( debug: bool, quiet: bool ) -> None:
    """Set output verbosity:
    
        --quiet -> Only the printed server/session preview + WARNING+ logs. The 
                   cleanest view for reading what the agent actually does.
        --debug -> everything: DEBUG-level logs from this app and its dependencies.
        default -> this app's own INFO logs.
    """
    if debug:
        root_level, app_level = logging.DEBUG, logging.DEBUG
    elif quiet:
        root_level, app_level = logging.ERROR, logging.WARNING
    else:
        root_level, app_level = logging.WARNING, logging.INFO
        
    logging.basicConfig(
        level = root_level,
        format = LOG_FORMAT,
        datefmt = DATE_FORMAT,
        stream = sys.stderr,
        force = True
    )
    
    # Warning of this app are kept
    log.setLevel( app_level )
    
    logging.captureWarnings( True )
    logging.getLogger( "py.warnings" ).setLevel( logging.ERROR if quiet else logging.WARNING )
    
# ==============================================================================
# Entry point
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser( description = "Minimal ADK search chat backed by MCP servers, with selectable session storage." )
    parser.add_argument(
        "--debug", action = "store_true", help = "Verbose logging: one line per raw event in the loop."
    )
    parser.add_argument(
        "--quiet", action = "store_true", help = "clean preview: suppress the app's INFO and library logs."
    )
    parser.add_argument(
        "--list-tools", action = "store_true", help = "Connect, print each server's advertised tools, then exit."
    )
    parser.add_argument(
        "--delete-session", action = "store_true", help = "Delete the session by --session-id for sqlite store, then exit."
    )
    parser.add_argument(
        "--session-backend", 
        choices = [ backend.value for backend in SessionBackend ],
        default = SessionBackend.MEMORY.value,
        help = (
            f"Where the conversion history lives: '{SessionBackend.MEMORY.value}' (ephemeral) or '{SessionBackend.SQLITE.value}' (persistent). "
            f"Default: {SessionBackend.MEMORY.value}"
        )
    )
    parser.add_argument(
        "--db-url",
        default = DEFAULT_DB_URL,
        help = f"SQLALchemy URL for the sqlite backend. Default:{DEFAULT_DB_URL}"
    )
    parser.add_argument(
        "--user-id",
        default = USER_ID,
        help = f"User of the session belongs to. Default:{USER_ID}"
    )
    parser.add_argument(
        "--session-id",
        default = SESSION_ID,
        help = f"Session to open. Re-use the same id with sqlite to resume. Default:{SESSION_ID}"
    )
    parser.add_argument(
        "--audit",
        action = "store_true",
        help = "Print a read-only audit of the sqlite session database, then exit."
    )
    args = parser.parse_args()
    
    configure_logging( args.debug, args.quiet )
    
    config = SessionConfig(
        backend = SessionBackend( args.session_backend ),
        db_url = args.db_url,
        app_name = APP_NAME,
        user_id = args.user_id,
        session_id = args.session_id
    )
    
    # --audit: synchronous, read-only report
    if args.audit:
        print_audit_report( config )
        return
    
    if args.delete_session:
        asyncio.run( delete_session_command( config ) )
        return
    
    if args.list_tools:
        asyncio.run( list_tools() )
        return
    
    asyncio.run( chat( config ) )
    
if __name__ == "__main__":
    main()