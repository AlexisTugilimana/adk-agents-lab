"""
==============================================================================
chat-1-simple-chat.py
==============================================================================

Small useful ADK agent: a terminal chat with Google Search tool.
Goal is to explore the following:

    1. asyncio      - Python's async/await concurrency model.
    2. Runner       - The object that drives the agent.
    3. Event loop   - How the agent streams back a series of events.

Authentication done by the individual login with the gcloud account:

    - gcloud auth application-default login
    - export GOOGLE_CLOUD_PROJECT="your-project-id"
    - export GOOGLE_CLOUD_LOCATION="your-location"
    - export GOOGLE_GENAI_USE_VERTEXAI=TRUE

Run it with: python chat-1-simple-chat.py
"""
import os
import time
import asyncio
import logging
import sys
import argparse
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.adk.events import Event
from google.genai import types

# ==============================================================================
# Constants
# ==============================================================================
AGENT_NAME = "search_agent"
MODEL = "gemini-2.5-flash"
APP_NAME = "simple-chat"
USER_ID = "user-0"
SESSION_ID = "session-0"
UNKNOWN_RESPONSE_NAME = "<unknown>"

INSTRUCTIONS = (
    "You are a helpful assistant. When a question needs current or "
    "factual information, use Google Search, then answer concisely and "
    "cite what you found."
)

# ==============================================================================
# Logger
# ==============================================================================
log = logging.getLogger( "simple-chat" )

# ==============================================================================
# Classes
# ==============================================================================
@dataclass
class ToolCall:
    name: str
    args: dict = field( default_factory = dict )

@dataclass
class ToolResult:
    name: str

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
# Agent
# ==============================================================================
def buildAgent() -> Agent:
    return Agent(
        name = AGENT_NAME,
        model = MODEL,
        instruction = INSTRUCTIONS,
        tools = [ google_search ]
    )
    
# ==============================================================================
# Event loop
# ==============================================================================
def observe( event: Event ) -> list[ Update ]:
    """Translate one raw ADK event into zero or more semantic updates."""
    
    updates: list[ Update ] = []
    
    # Tool calls and responses
    for call in event.get_function_calls():
        updates.append( ToolCall( call.name or UNKNOWN_RESPONSE_NAME, dict( call.args or {} ) ) )
    for resp in event.get_function_responses():
        updates.append( ToolResult( resp.name or UNKNOWN_RESPONSE_NAME ) )
    
    # Built-il tools for grounding
    gm = getattr( event, "grounding_metadata", None )
    queries = getattr( gm, "web_search_queries", None ) if gm else None
    if queries:
        updates.append( ToolCall( "google_search", { "queries": list( queries ) } ) )
    
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
            
            usage = getattr( event, "usage_metada", None )
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
    elif isinstance( update, Answer ):
        print( f"agent \u25b8 {update.text}\n" )
        
# ==============================================================================
# Runner (chat loop)
# ==============================================================================
async def chat() -> None:
    
    session_service = InMemorySessionService()
    agent = buildAgent()
    runner = Runner( app_name = APP_NAME, agent = agent, session_service = session_service )
    log.info( f"Startup app={APP_NAME} agent={agent.name} model={MODEL}" )
    
    user_id, session_id = USER_ID, SESSION_ID
    await session_service.create_session(
        app_name = APP_NAME, user_id = user_id, session_id = session_id
    )
    log.info( f"session.ready user={user_id} session={session_id}" )
    
    print( "Search chat - type a question, or 'exit' to quit.\n" )
    
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
# Authentication
# ==============================================================================
def checkAuth() -> str:
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
    parser = argparse.ArgumentParser( description = "Minimal ADK search chat." )
    parser.add_argument(
        "--debug", action = "store_true", help = "Verbose logging: one line per raw event in the loop."
    )
    args = parser.parse_args()
    
    configure_logging( args.debug )
    print( f"[auth] using {checkAuth()}\n" )
    asyncio.run( chat() )
    
if __name__ == "__main__":
    main()