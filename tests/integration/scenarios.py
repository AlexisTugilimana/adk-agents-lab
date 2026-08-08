"""Reusable HITL flows, shared by local and deployed test files.

Local and deployed suites are kepts in separate files: two have different failure 
modes and different debugging paths (deployed failure sends to container log, local 
to the in-process one), so they should be runnable independently. But the flow to 
verify - park, approve, reject, batch, session binding - is identical.
"""

from typing import Any

from tests.integration.driver import GATED_PROMPT, TWO_GATED_PROMPT, drain, new_session
from tests.integration.events import (
    Event,
    calls_named,
    executed_responses,
    long_running_ids,
    pending_responses,
    responses_named
)
from tests.integration.guards import require_batch, require_confirmation, require_tool_call
from tests.integration.hitl import (
    CONFIRMATION_NAME,
    GATED_TOOL,
    build_approvals,
    build_approval
)

# How many times to re-prompt when the model declines to call the gated tool.
TOOL_CALL_ATTEMPTS = 3

async def _park(
    app: Any, 
    prompt: str, 
    *, 
    attempts: int = TOOL_CALL_ATTEMPTS
) -> tuple[ str, list, list ]:
    """Drive `prompt` until the model gates a tool, in a fresh session per attempt.
    
    Returns `(session_id, events, confirmation_requests)`. A fresh session per attempt 
    keeps retries clean.
    """
    
    events: list[ Event ] = []
    session_id = ""
    for _ in range( 1, attempts + 1 ):
        session_id = await new_session( app )
        events = await drain( app, prompt, session_id = session_id )
        if calls_named( events, GATED_PROMPT ):
            break
        
    # Fail here if `GATED_PROMPT` never called in the attempts
    require_tool_call( events )
    return ( session_id, events, require_confirmation( events ) )
    
# ==============================================================================
# Pause
# ==============================================================================
async def verify_turn_parks( app: Any ) -> None:
    """call -> pending -> confirmation request, with the tool never running."""
    
    _, events, requests = await _park( app, GATED_PROMPT )
    
    # The plugin answer `pending` instead of the tool output
    assert pending_responses( events, GATED_TOOL ), "expected a pending response for the gated tool"
    assert not executed_responses( events, GATED_TOOL ), (
        "gated tool produced a real response - it may have executed: "
        f"{responses_named( events, GATED_TOOL )!r}"
    )
    
    # ADK converted it into a long-running confirmation call whose id is advertised.
    assert any( r[ "advertised" ] for r in requests ), (
        f"a {CONFIRMATION_NAME} is must appear in long_running_tool_ids; "
        f"saw ids={long_running_ids( events )!r}"
    )
    
    # The payload must carry enough information for a human to decide - assert the 
    # keys
    first = requests[ 0 ]
    assert first[ "original_call" ], "confirmation must reference the original function call"
    assert first[ "tool_confirmation" ].get( "confirmed" ) is False

# ==============================================================================
# Resume
# ==============================================================================
async def verify_approval_executes( app: Any ) -> None:
    """`confirmed: true` in the same session -> the tool finally runs."""
    
    session_id, _, requests = await _park( app, GATED_PROMPT )
    
    resumed = await drain(
        app,
        build_approval( requests[ 0 ][ "id" ], confirmed = True ),
        session_id = session_id
    )
    
    assert executed_responses( resumed, GATED_TOOL ), (
        "after approval the gated tool should have executed; "
        f"saw responses={responses_named( resumed, GATED_TOOL )!r}"
    )

async def verify_rejection_does_not_executes( app: Any ) -> None:
    """`confirmed: false` must not run the tool."""
    
    session_id, _, requests = await _park( app, GATED_PROMPT )
        
    resumed = await drain(
        app,
        build_approval( requests[ 0 ][ "id" ], confirmed = False ),
        session_id = session_id
    )
    
    for fr in responses_named( resumed, GATED_TOOL ):
        payload = fr.get( "response" ) or {}
        assert payload.get( "status" ) != "success", f"tool executed despite rejection:{payload!r}"
        assert "hello world" not in str( payload.get( "result", "" ) ), (
            "tool output present after rejection"
        )

# ==============================================================================
# Security properties of the resume
# ==============================================================================
async def _assert_resume_is_refused( app: Any, message: Any, session_id: str, what: str ) -> None:
    """A resume attempts that cannot be resolved must not execute the gated tool.
    
    ADK matches every incoming function_response against the function_calls in that 
    session's history to infer invocation. When no match exists, it raises a ValueError.
    
    Defensive guard: since a future version of ADK could switch to ignoring instead 
    of raising, another assertion is done outsine the ValueError exception.
    """
    
    try:
        resumed = await drain(
            app,
            message,
            session_id = session_id
        )
    except ValueError as exc:
        assert "not found" in str( exc ).lower(), (
            f"{what}: expected an unresolved-function-response ValueError, got {exc!r}"
        )
        return
    
    assert not executed_responses( resumed, GATED_TOOL ), (
        f"{what}: ADK accepted the resume without raising, but the gated tool executed; "
        f"saw {responses_named( resumed, GATED_TOOL )!r}"
    )
    
async def verify_approval_is_bound_to_its_session( app: Any ) -> None:
    """An approval is bound to the session holding the parked turn.
    
    A valid confirmation ID played with another session ID must not release the 
    tool.
    """
    
    parked_session_id = await new_session( app )
    parked = await drain( app, GATED_PROMPT, session_id = parked_session_id )
    require_tool_call( parked )
    requests = require_confirmation( parked )
    
    other_session_id = await new_session( app )
    await _assert_resume_is_refused(
        app,
        build_approval( requests[ 0 ][ "id" ], confirmed = True ),
        other_session_id,
        "approval replayed into another session"
    )

async def verify_unknown_confirmation_id_does_nothing( app: Any ) -> None:
    """A fabricated confirmation id must resolve nothing.
    
    ADK matches the id against `adk_request_confirmation` calls in session history, 
    so an id that was never issued has no target and must never execute.
    """
    
    session_id = await new_session( app )
    parked = await drain( app, GATED_PROMPT, session_id = session_id )
    require_tool_call( parked )
    requests = require_confirmation( parked )
    
    other_session_id = await new_session( app )
    await _assert_resume_is_refused(
        app,
        build_approval( "adk-nonexistence-000", confirmed = True ),
        other_session_id,
        "fabricated confirmation id"
    )