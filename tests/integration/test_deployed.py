"""Deployed-only behaviour.

Only test a simple interaction with a deployed agent, not the HITL loop.
"""

from tests.integration.apps import USER_ID, deployed_agent
from tests.integration.driver import (
    MEMORY_PROMPT,
    MEMORY_RECALL_PROMPT,
    NO_TOOL_PROMPT,
    drain,
    session_scope
)
from tests.integration.events import has_text
from tests.integration.hitl import confirmation_requests

async def test_managed_sessions_persist_history_across_turns():
    """Two turns in one session must share history."""
    
    app = deployed_agent()
    
    async with session_scope( app ) as scope:
        session_id = await scope.create()
        
        await drain( app, MEMORY_PROMPT, session_id = session_id )
        second = await drain( app, MEMORY_RECALL_PROMPT, session_id = session_id )
        
        # Structural check only, no assertion on the response quality
        assert has_text( second ), "second turn in a managed session produced no reply"
        
        fetched = await app.async_get_session( user_id = USER_ID, session_id = session_id )
        assert fetched, "the platform did not retain the session"

async def test_deployed_plain_turn_completes():
    """Ordinary turns are not gated."""
    
    app = deployed_agent()
    
    async with session_scope( app ) as scope:
        session_id = await scope.create()
        
        events = await drain( deployed_agent(), NO_TOOL_PROMPT, session_id = session_id )
        
        assert events, "expected at least one event"
        assert not confirmation_requests( events ), "a no-tool turn must not request approval"
        assert has_text( events ), "expected a textual response"