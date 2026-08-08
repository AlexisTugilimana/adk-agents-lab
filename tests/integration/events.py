"""Event-shape library - pure functions over the streamed event dicts.

No pytest, no I/O: plain library to be imported from everywhere. 
"""
from typing import Any, Iterable

Event = dict[ str, Any ]

# ==============================================================================
# Part-level accessors
# ==============================================================================
def _first( d: Any, *names: str, default: Any = None ) -> Any:
    """First present key among `names`."""
    
    if not isinstance( d, dict ):
        return default
    for name in names:
        if name in d:
            return d[ name ]
    return default

def parts( event: Event ) -> list[ dict[ str, Any ] ]:
    return _first( _first( event, "content", default = {} ) or {}, "parts", default = [] ) or []

def function_calls( event: Event ) -> list[ dict[ str, Any ] ]:
    """Every function_call part in one event."""
    
    out = []
    for part in parts( event ):
        func_call = _first( part, "function_call", "functionCall" )
        if func_call:
            out.append( func_call )
    return out

def function_responses( event: Event ) -> list[ dict[ str, Any ] ]:
    """Every function_response part in one event."""
    
    out = []
    for part in parts( event ):
        func_resp = _first( part, "function_response", "functionResponse" )
        if func_resp:
            out.append( func_resp )
    return out

def long_running_ids( events: Iterable[ Event ] ) -> list[ str ]:
    
    out = []
    for event in events:
        out.extend(
            _first( event, "long_running_tool_ids", "longRunningToolIds", default = [] ) or []
        )
    return out

# ==============================================================================
# Stream-level queries
# ==============================================================================
def calls_named( events: Iterable[ Event ], name: str ) -> list[ dict[ str, Any ] ]:
    return [ fc for e in events for fc in function_calls( e ) if fc.get( "name" ) == name ]

def responses_named( events: Iterable[ Event ], name: str ) -> list[ dict[ str, Any ] ]:
    return [ fr for e in events for fr in function_responses( e ) if fr.get( "name" ) == name ]

def response_status( function_response: dict[ str, Any ] ) -> str | None:
    payload = function_response.get( "response" ) or {}
    return payload.get( "status" ) if isinstance( payload, dict ) else None

def pending_responses( events: Iterable[ Event ], tool: str ) -> list[ dict[ str, Any ] ]:
    """Response for `tool` carrying a `pending` status i.e., the tool was intercepted 
    rather than executed."""
    
    return [ fr for fr in responses_named( events, tool ) if response_status( fr ) == "pending" ]

def executed_responses( events: list[ Event ], tool: str ) -> list[ dict[ str, Any ] ]:
    """Responses for `tool` that are not in pending i.e., tool that actually ran."""
    
    return [ fr for fr in responses_named( events, tool ) if response_status( fr ) != "pending" ]

def has_text( events: Iterable[ Event ] ) -> bool:
    """Whether any model text came back. Presence only."""
    
    return any( part.get( "text " ) for e in events for part in parts( e ) )

def event_shape( events: Iterable[ Event ] ) -> list[ str ]:
    """Reduce a stream into a comparable sequences of structural labels."""

    shape: list[ str ] = []
    for e in events:
        calls = function_calls( e )
        responses = function_responses( e )
        for fc in calls:
            shape.append( f"call:{fc.get( "name" )}" )
        for fr in responses:
            status = response_status( fr )
            shape.append( f"response:{fr.get( "name" )}" + ( f":{status}" if status else "" ) )
        if not calls and not responses and has_text( [ e ] ):
            shape.append( "text" )
    return shape