"""Local runtime integration: session persistence, plain turns, observability contract."""

import uuid
from pathlib import Path
import warnings

from tests.integration.apps import USER_ID, capture_logs, local_agent, local_config
from tests.integration.driver import GATED_PROMPT, NO_TOOL_PROMPT, drain, new_session
from tests.integration.events import calls_named, executed_responses, has_text
from tests.integration.guards import require_confirmation, require_tool_call
from tests.integration.hitl import GATED_TOOL, build_approval, confirmation_requests

async def test_plain_turn_completes_without_gating():
    """A prompt that needs no tool must complete normally.
    
    Regression guard against over-gating.
    """
    
    async with local_agent() as app:
        events = await drain( app, NO_TOOL_PROMPT )
    
    assert events, "expected at least one event"
    assert not confirmation_requests( events ), "a no-tool turn must not request approval"
    assert not calls_named( events, GATED_TOOL )
    assert has_text( events ), "expected a textual response"

async def test_metrics_line_carries_both_correlation_ids():
    """One aggregate metrics line per invocation, with session_id and invocation_id.
    
    Similar test already implemented in the unit test `test_metrics.py`, with a 
    FakeClock. Here the Metrics plugin is tested in real in-process conditions.
    
    Instead of `caplog` fixture, a custom handler is implemented for testing.
    """
    
    async with local_agent() as app:
        with capture_logs( "agent.metrics" ) as records:
            await drain( app, NO_TOOL_PROMPT )
    
    # Get metrics logs only (which starts with "metrics ...")
    lines = [ r.getMessage() for r in records if r.getMessage().startswith( "metrics ") ]
    assert len( lines ) == 1, f"expected exactly one aggregate line, got {lines!r}"
    
    line = lines[ 0 ]
    assert "session_id=" in line and "inv=" in line 
    assert "latency_ms=" in line
    # IDs must be non empty
    assert "session_id= " not in line and "inv= " not in line
    
async def test_parked_turn_survives_a_new_app_instance():
    """SQlite-backed sessions: park in one AdkApp, resume in a different one."""
    
    from agent.config import SessionBackend
    
    db_path = Path( f"it_sessions_{uuid.uuid4().hex[:8]}.db" )
    cfg = local_config(
        session_backend = SessionBackend.SQLITE,
        db_url = f"sqlite+aiosqlite:///{db_path}"
    )
    
    try:
        # --------------------------------------------------------------------------
        # app instance #1: park a turn
        # --------------------------------------------------------------------------
        async with local_agent( cfg ) as app:
            session_id = await new_session( app )
            parked = await drain( app, GATED_PROMPT, session_id = session_id )
            require_tool_call( parked )
            confirmation_id = require_confirmation( parked )[ 0 ][ "id" ]
        
        # --------------------------------------------------------------------------
        # app instance #2: the previous app is gone
        # --------------------------------------------------------------------------
        async with local_agent( cfg ) as app:
            fetched = await app.async_get_session( user_id = USER_ID, session_id = session_id )
            assert fetched, "session dit not survive the app instance"
            
            resumed = await drain(
                app,
                build_approval( confirmation_id = confirmation_id, confirmed = True ),
                session_id = session_id
            )
            assert executed_responses( resumed, GATED_TOOL ), (
                "a persisted parked turn should be resumable by a new app"
            )
    finally:
        _remove_sqlite_files( db_path )
        
def _remove_sqlite_files( db_path: Path ) -> None:
    """Delete the temporary database created for the testing.
    
    Failure modes in deleting fails loudly 
    """
    
    try:
        db_path.unlink( missing_ok = True )
    except PermissionError:
        warnings.warn(
            f"could not delete {db_path}: still open.",
            stacklevel = 2
        )