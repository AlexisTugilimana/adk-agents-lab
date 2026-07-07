"""
==============================================================================
chat-1-team-chat.py
==============================================================================

A multi-agent ADK "travel team" you can chat with in the terminal:

    - AgentTool:        Wrapping the whole agent so another agent can call it
                        like a function. 
    - Coordinator:      An LLM decides, per query, which specialists to call and 
                        in what order.
    - Generator/Critic: An independant reviewer that provides a grade to the 
                        initary before it reaches the user.
    - Artifacts:        Named, versioned binary blobs held by the ArtifactService. 
                        The finished initary is saved as one. 
    - Authored events:  Every event carries an author, so we can see the team 
                        collaborate rather than just the final blob.
        
The team:
    - Trip coordinator (root):  Orchestrates, saved the deliverable, and write the 
                                answer.
    - Trip researcher:          Finds real places/restaurants/ etc (google_search)
    - Route planner:            Order stops and plan travel (google_search)
    - Itinerary critic:         Independently review the draft (no tools)
    
Authentication done by the individual login with the gcloud account:

    - gcloud auth application-default login
    - export GOOGLE_CLOUD_PROJECT="your-project-id"
    - export GOOGLE_CLOUD_LOCATION="your-location"
    - export GOOGLE_GENAI_USE_VERTEXAI=TRUE

Run it with: python chat-1-team-chat.py
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
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search, ToolContext
from google.adk.events import Event
from google.genai import types

# ==============================================================================
# Constants
# ==============================================================================
APP_NAME = "Trip-team-chat"
USER_ID = "user-0"
SESSION_ID = "session-0"

# One model for all agents
MODEL = "gemini-3.5-flash"

RESEARCHER_NAME = "trip_researcher"
ROUTE_NAME = "route_planner"
CRITIC_NAME = "itinerary_critic"
COORDINATOR_NAME = "trip_coordinator"
ITINERARY_TOOL = "save_itinerary"
SPECIALISTS = { RESEARCHER_NAME, ROUTE_NAME, CRITIC_NAME }

# Unknown variable name to display
UNKNOWN = "<unknown>"

# ==============================================================================
# Model
# ==============================================================================
RETRY = types.HttpRetryOptions(
    attempts = 5,                               # Total attempts per request (1 initial + 4 retries)
    initial_delay = 1.0,                        # Seconds before the first retry
    max_delay = 60.0,                           # Cap any single wait
    exp_base = 2.0,                             # Exponential backoff between 2 retries
    jitter = 1.0,                               # De-sync concurent retries
    http_status_codes = [ 429, 500, 503, 504 ]  # throttling (429) + transient backend error
)

LLM = Gemini( model = MODEL, retry_options = RETRY )

# ==============================================================================
# Logger
# ==============================================================================
log = logging.getLogger( "trip-team-chat" )

# ==============================================================================
# Classes
# ==============================================================================
@dataclass
class ToolCall:
    author: str
    name: str
    args: dict = field( default_factory = dict )

@dataclass
class ToolResult:
    author: str
    name: str
    preview: str | None = None

@dataclass
class ArtifactSaved:
    author: str
    filename: str
    version: int
    
@dataclass
class Answer:
    author: str
    text: str

Update = Union[ ToolCall, ToolResult, Answer, ArtifactSaved ]

@dataclass
class TurnMetrics:
    """Per-turn operational telemetry"""
    
    invocation_id: str | None = None
    events: int = 0
    tool_calls: int = 0
    delegations: dict[ str, int ] = field( default_factory = dict )
    artifacts_saved: int = 0
    # Summed across the turn
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    
# ==============================================================================
# INSTRUCTIONS
# ==============================================================================
RESEARCHER_INSTRUCTION = (
    "You are a travel research specialist. Given a request about a city, "
    "neighborhood, activity, restaurant, or attraction, use Google Search to "
    "find concrete, real, currently-open options - never invent places.\n"
    "For each suggestion give: name, area/neighborhood, what it is known for, "
    "rough price level, and any timing notes (hours, best time to go). Prefer a "
    "handful of specific named places over generic advice. If the request is "
    "ambiguous, state the one assumption you made in a single line, then answer."
)

ROUTE_INSTRUCTION = (
    "You are a route and logistics planner. You are given a set of locations "
    "(optionally a start point, a travel mode, or time constraints) and must "
    "produce a sensible visiting order that minimizes backtracking.\n"
    "Use Google Search to look up approximate travel times/distances and "
    "walking or transit options when it helps. Group stops by area and respect "
    "opening hours. Output an ordered list of legs in the form "
    "'Stop -> Stop  (~time, mode)', then a one-line rationale. Do not invent "
    "precise distances; search for them or mark the estimate as approximate."
)

CRITIC_INSTRUCTION = (
    "You are an independent quality critic for travel plans. You have NO tools; "
    "you judge only the draft you are handed. Check: factual plausibility, "
    "feasibility (timing, geography, opening hours), whether it actually answers "
    "the user's request, and clarity.\n"
    "Reply with a first line that is exactly 'VERDICT: APPROVE' or "
    "'VERDICT: REVISE', then a short bulleted list of concrete issues and fixes. "
    "Approve only when the plan is genuinely solid. Be terse and specific."
)

COORDINATOR_INSTRUCTION = (
    "You are the coordinator of a travel-planning team. You answer the user by "
    "orchestrating specialist tools - you never search the web yourself.\n"
    "\n"
    "Your tools:\n"
    f"  - {RESEARCHER_NAME}: finds real places, restaurants, attractions.\n"
    f"  - {ROUTE_NAME}: orders stops and plans travel between locations.\n"
    f"  - {CRITIC_NAME}: independently reviews a DRAFT, returns APPROVE/REVISE.\n"
    f"  - {ITINERARY_TOOL}: saves a finished itinerary as a versioned file.\n"
    "\n"
    "How to operate:\n"
    "1. Work out what the user actually needs. A simple lookup ('a good ramen "
    "place near X') may need only the researcher. A 'best route between A and B' "
    "needs the route planner. A full itinerary needs research, then routing.\n"
    "2. Call specialists in whatever order, and as many times, as the task "
    "needs - passing each the specific context it requires. The sequence is "
    "yours to decide; that is your job.\n"
    "3. For itinerary-grade answers, assemble a draft and send it to "
    f"{CRITIC_NAME} once. If the verdict is REVISE, fix the issues (re-research "
    "or re-route as needed) and you may review at most once more - never loop "
    "more than twice.\n"
    "4. When (and only when) you have produced an itinerary-grade plan, call "
    f"{ITINERARY_TOOL} once with a short title and the final plan as markdown, "
    "so the user keeps a saved copy. If the user later asks you to change a "
    "saved plan, save again under the same title - that becomes a new version. "
    "Do NOT save for quick one-off lookups.\n"
    "5. Reply to the user with one clear, concrete answer. Do not expose the "
    "internal tool chatter or the critic's verdict. Never fabricate places or "
    "distances - that is what your specialists are for."
)

# ==============================================================================
# Tools
# ==============================================================================
def slugify( text: str ) -> str:
    keep = "".join( c.lower() if c.isalnum() else "-" for c in text.strip() )
    slug = "-".join( filter( None, keep.split( "-" ) ) )
    return slug or "itinerary"

async def save_itinerary( title: str, markdown: str, tool_context: ToolContext ) -> dict:
    """Persist a finished plan as a versioned ADK artifact.
    
    ArtifactService registered on the runner dorst the storage. `save_artifact` 
    returns the new integer version, so re-saving the same filename laters yields to 
    a new version. This enables the critic loop's revisions to become a visible version 
    history.
    """
    
    filename = slugify( title )
    part = types.Part.from_bytes( data = markdown.encode( "utf-8" ), mime_type = "text/markdown" )
    version = await tool_context.save_artifact( filename, part )
    log.info( f"artifact.save filename={filename} version={version} bytes={len( markdown )}")
    return { "status": "saved", "filename": filename, "version": version }

# ==============================================================================
# Agents
# ==============================================================================
def build_researcher() -> Agent:
    return Agent(
        name = RESEARCHER_NAME,
        model = LLM,
        description = "Finds real places, restaurants and attractions using Google Search.",
        instruction = RESEARCHER_INSTRUCTION,
        tools = [ google_search ]
    )

def build_route_panner() -> Agent:
    return Agent(
        name = ROUTE_NAME,
        model = LLM,
        description = "Orders a set of stops and plans the travel/route between them.",
        instruction = ROUTE_INSTRUCTION,
        tools = [ google_search ]
    )
    
def build_critic() -> Agent:
    return Agent(
        name = CRITIC_NAME,
        model = LLM,
        description = "Independently reviews a draft plan and returns APPROVE / REVISE with issues.",
        instruction = CRITIC_INSTRUCTION
    )

def build_coordinator() -> Agent:
    return Agent(
        name = COORDINATOR_NAME,
        model = LLM,
        description = "Coordinates the travel team, saves the deliverable, writes the final answer.",
        instruction = COORDINATOR_INSTRUCTION,
        tools = [
            AgentTool( agent = build_researcher(), propagate_grounding_metadata = True ),
            AgentTool( agent = build_route_panner(), propagate_grounding_metadata = True ),
            AgentTool( agent = build_critic() ),
            save_itinerary
        ]
    )
    
# ==============================================================================
# Event loop
# ==============================================================================
def summarize_response( resp ) -> str | None:
    """Response string of a function/tool response payload."""
    
    val = getattr( resp, "response", None )
    if val is None: return None
    
    if isinstance( val, dict ):
        for key in ( "result", "text", "output", "response" ):
            inner = val.get( key )
            if isinstance( inner, str ) and inner:
                return inner
        return str( val )
    return str ( val )

def observe( event: Event ) -> list[ Update ]:
    """Translate one raw ADK event into zero or more semantic updates."""
    
    author = getattr( event, "author", None ) or UNKNOWN
    updates: list[ Update ] = []
    
    # Delegations
    for call in event.get_function_calls():
        updates.append( ToolCall( author, call.name or UNKNOWN, dict( call.args or {} ) ) )
        
    # Results coming back
    for resp in event.get_function_responses():
        updates.append( ToolResult( author, resp.name or UNKNOWN, summarize_response( resp ) ) )
    
    # Built-il tools for grounding
    gm = getattr( event, "grounding_metadata", None )
    queries = getattr( gm, "web_search_queries", None ) if gm else None
    if queries:
        updates.append( ToolCall( author, "google_search", { "queries": list( queries ) } ) )
    
    # Artifact written by the agent during the event: ADK records them as a { name: version } 
    # delta on event.actions.
    actions = getattr( event, "actions" )
    delta = getattr( actions, "artifact_delta", None ) if actions else None
    if delta:
        for filename, version in delta.items():
            updates.append( ArtifactSaved( author, filename, version ) )
            
    # Final response
    if event.is_final_response():
        parts = ( event.content.parts or [] ) if event.content else []
        text = "".join( ( p.text or "" ) for p in parts if getattr( p, "text", None ) )
        if text: updates.append( Answer( author, text ) )
    
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
            for call in calls:
                if call.name in SPECIALISTS:
                    metrics.delegations[ call.name ] = metrics.delegations.get( call.name, 0 ) + 1
                    
            actions = getattr( event, "actions", None )
            delta = getattr( actions, "artifact_delta", None ) if actions else None
            if delta:
                metrics.artifacts_saved += len( delta )
            
            usage = getattr( event, "usage_metadata", None )
            if usage:
                metrics.prompt_tokens = getattr( usage, "prompt_token_count", None ) or metrics.prompt_tokens
                metrics.output_tokens = getattr( usage, "candidates_token_count", None ) or metrics.output_tokens
                metrics.total_tokens = getattr( usage, "total_token_count", None ) or metrics.total_tokens
            
            log.debug(
                f"event inv={metrics.invocation_id} author={event.author} "
                f"final={event.is_final_response()} tools={[c.name for c in calls] or None} "
                f"artifact={list( delta ) if delta else None}"
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
            f"delegations={metrics.delegations or None} artifacts_saved={metrics.artifacts_saved} "
            f"latency={elapsed_ms} tokens(prompt={metrics.prompt_tokens} output={metrics.output_tokens} "
            f"total={metrics.total_tokens})"
        )

def truncate( text: str, n: int = 160 ) -> str:
    return text if len( text ) <= n else text[ : n - 1 ] + "\u2026"

def present( update: Update ) -> None:
    """Single presentation layer that writes to the terminal."""
    
    if isinstance( update, ToolCall ):
        if update.name in SPECIALISTS:
            detail = ", ".join( f"{k}={truncate( str( v ), 80 )!r}" for k, v in update.args.items() )
            print( f"   \u21B3 {update.author} \u2192 {update.name}({detail})" )
        elif update.name == "google_search":
            detail = ", ".join( f"{k}={v!r}" for k, v in update.args.items() )
            print( f"   \U0001F50D  {update.author}: google_search({detail})" )
        else:
            detail = ", ".join( f"{k}={truncate( str( v ), 60 )!r}" for k, v in update.args.items() )
            print( f"   \U0001F527  {update.author}: {update.name}({detail})" )
    elif isinstance( update, ToolResult ):
        tail = f"\u2014 \"{truncate( update.preview )}\"" if update.preview else ""
        print( f"   \u2713 {update.name} returned{tail}" )
    elif isinstance( update, ArtifactSaved ):
        print( f"   \U0001F4BE artifact saved: {update.filename} (v{update.version})" )
    elif isinstance( update, Answer ):
        if update.author == COORDINATOR_NAME:
            print( f"\nagent \u25b8 {update.text}\n" )
        
# ==============================================================================
# Artifact analysis
# ==============================================================================
def artifact_text( part ) -> str:
    if part is None: return "<missing>"
    if getattr( part, "text", None ): return part.text
    
    blob = getattr( part, "inline_data", None )
    if blob and getattr( blob, "data", None ):
        try:
            return blob.data.decode( "utf-8" )
        except UnicodeDecodeError:
            return f"<{len(blob.data)} bytes, {blob.mime_type}>"
    return "<empty>"

async def handle_cmd( text: str, artifact_service: InMemoryArtifactService, *, user_id: str, session_id: str ) -> None:
    
    parts = text.split( maxsplit = 1 )
    cmd = parts[ 0 ].lower()
    arg = parts[ 1 ].strip() if len( parts ) > 1 else ""
    
    if cmd in { "/help", "/?" }:
        print( "commands: /saved (list artifacts)\t/show <name> (print latest)\t/help\n" )
        return
    
    if cmd == "/saved":
        keys = await artifact_service.list_artifact_keys(
            app_name = APP_NAME, user_id = user_id, session_id = session_id
        )
        if not keys:
            print( "(no artifacts saved yet)\n")
        for name in keys:
            versions = await artifact_service.list_artifact_versions(
                app_name = APP_NAME, user_id = user_id, session_id = session_id, filename = name
            )
            print( f"   \U0001F4BE {name}   versions={versions}" )
        return
    
    if cmd == "/show":
        if not arg:
            print( "usage: /show <filename>\n" )
            return
        part = await artifact_service.load_artifact(
            app_name = APP_NAME, user_id = user_id, session_id = session_id, filename = arg
        )
        print( f"\n----- {arg} (latest) -----\n{artifact_text( part )}" )
        return
    
    print( f"unknown command {cmd!r} - try /help\n" )
    
# ==============================================================================
# Runner (chat loop)
# ==============================================================================
async def chat() -> None:
    
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    coordinator: Agent = build_coordinator()
    runner = Runner(
        app_name = APP_NAME, 
        agent = coordinator, 
        session_service = session_service,
        artifact_service = artifact_service
    )
    log.info( f"Startup app={APP_NAME} agent={coordinator.name} model={MODEL} team={sorted(SPECIALISTS)}" )
    
    user_id, session_id = USER_ID, SESSION_ID
    await session_service.create_session(
        app_name = APP_NAME, user_id = user_id, session_id = session_id
    )
    log.info( f"session.ready user={user_id} session={session_id}" )
    
    print( "Team-travel chat - ask for a place, a route, or a full itinerary." )
    print( "Commands: /saved    /show <name> /help (or 'exit' to quit)\n" )
    
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

        # Slash commands are handled locally, never sent to the agent
        if text.startswith( "/" ):
            await handle_cmd( text, artifact_service, user_id = user_id, session_id = session_id )
            continue
        
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
        level = logging.DEBUG if debug else logging.WARNING,
        format = "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt = "%H:%M:%S",
        stream = sys.stderr
    )
    
    log.setLevel( logging.DEBUG if debug else logging.WARNING )
    
    
# ==============================================================================
# Entry point
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser( description = "Multi-agent ADK travel-team chat." )
    parser.add_argument(
        "--debug", action = "store_true", help = "Verbose logging: one line per raw event in the loop."
    )
    args = parser.parse_args()
    
    configure_logging( args.debug )
    print( f"[auth] using {checkAuth()}\n" )
    asyncio.run( chat() )
    
if __name__ == "__main__":
    main()