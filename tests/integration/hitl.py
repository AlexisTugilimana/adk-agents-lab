"""HITL wire protocol - How a parked turn is observed and resumed."""
from typing import Any, Iterable

from tests.integration.events import Event, function_calls, long_running_ids

CONFIRMATION_NAME = "adk_request_confirmation"

# Natively-provided demo tool the policy gates under both local and cloud.
GATED_TOOL = "sensitive_echo"

def confirmation_request( events: Iterable[ Event ] ) -> list[ dict[ str, Any ] ]:
    """Every pending confirmation in the stream, normalized for the approver.
    
    Returns one entry per gated call:
        `{ id, advertised, original_call, tool_confirmation }`
    
    `advertised` records whether the id also appeared in `long_running_tool_ids`. 
    This is a defensive approach where, in addition to the tool name, the long running 
    ID is also used to mark the tool call as awaiting for a response.
    """
    
    events = list( events )
    advertised = long_running_ids( events )
    
    out: list[ dict[ str, Any ] ] = []
    for e in events:
        for fc in function_calls( e ):
            if fc.get( "name" ) != CONFIRMATION_NAME:
                continue
            args = fc.get( "args" ) or {}
            out.append(
                {
                    "id": fc.get( "id" ),
                    "advertised": fc.get( "id" ) in advertised,
                    "original_call": args.get( "originalFunctionCall" ) 
                    or args.get( "original_function_call" ) 
                    or {},
                    "tool_confirmation": args.get( "toolConfirmation" )
                    or args.get( "tool_confirmation" )
                    or {}
                }
            )
    return out

def build_approval( confirmation_id: str, *, confirmed: bool ) -> dict[ str, Any ]:
    """The message resolving one confirmation request.
    
    `confirmation_id` must the ID of an `adk_confirmation_request` function call, 
    not the original function call ID (requiring confirmation).
    """
    
    return build_approvals( { confirmation_id: confirmed } )
    
def build_approvals( decisions: dict[ str, bool ] ) -> dict[ str, Any ]:
    """Resolve several confirmations in one user turn, each owning its decision.
    
    ADK resolves per confirmation ID, so a batch may mix approvals and rejections - 
    `{ "id_a": True, "id_b": False }`.
    """
    
    return {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "id": confirmation_id,
                    "name": CONFIRMATION_NAME,
                    "response": { "confirmed": confirmed }
                }
            }
            for confirmation_id, confirmed in decisions.items()
        ]
        
    }