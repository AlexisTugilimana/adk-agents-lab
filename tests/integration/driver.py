"""Driving a turn - prompts, one-turn execution, session creation."""

import asyncio
from typing import Any

from tests.integration.events import Event
from tests.integration.hitl import GATED_TOOL

# ==============================================================================
# Prompts
# ==============================================================================
GATED_PROMPT = f"Use the {GATED_TOOL} tool to echo exactly: hello world"

TWO_GATED_PROMPT = (
    f"Call the {GATED_TOOL} tool twice in the same turn: once with the message "
    "'alpha' and once with the message 'beta'. Issue both calls together."
)

NO_TOOL_PROMPT = "What is 2+2? Answer with just the number. Do not use any tools."

MEMORY_PROMPT = "Remember this token and repeat it back if asked later: platypus-42"
MEMORY_RECALL_PROMPT = "What was the token I asked you to remember? Answer with the token only."

USER_ID = "it-user"

TURN_TIMEOUT_S = 120.0

# ==============================================================================
# Driving
# ==============================================================================
async def drain(
    app: Any,
    message: Any,
    *,
    user_id: str = USER_ID,
    session_id: str | None = None,
    timeout: float = TURN_TIMEOUT_S
) -> list[ Event ]:
    """Run one turn and collect streamed event.
    
    `session_id` is created by ADK when ommited. Passing an unknown id raises 
    SessionNotFoundError."""
    
    kwargs: dict[ str, Any ] = { "message": message, "user_id": user_id }
    if session_id is not None:
        kwargs[ "session_id" ] = session_id
    
    async def _collect() -> list[ Event ]:
        return [ e async for e in app.async_stream_query( **kwargs ) ]
    
    return await asyncio.wait_for( _collect(), timeout = timeout )

async def new_session( app: Any, user_id: str = USER_ID ) -> str:
    """Create a new session and return its id, tolerating dict or object return shape."""
    
    session = await app.async_create_session( user_id = user_id )
    return session[ "id" ] if isinstance( session, dict ) else session.id