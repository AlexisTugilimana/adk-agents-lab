"""Metrics plugin test."""

import logging
from dataclasses import dataclass
from typing import cast

from google.adk.agents.invocation_context import InvocationContext

from agent.plugins import MetricsPlugin
from tests.fakes import FakeClock

@dataclass
class _FakeSession:
    
    id: str
    
@dataclass
class _FaceInvocationContext:
    
    invocation_id: str
    session: _FakeSession
    
async def test_metrics_emits_one_line_with_latency_and_ids( caplog ):
    
    clock = FakeClock( [ 10.0, 10.5 ] )
    logger = logging.getLogger( "agent.metrics.test" )
    plugin = MetricsPlugin( clock = clock, logger = logger )
    
    ctx = _FaceInvocationContext(
        invocation_id = "inv-123", session = _FakeSession( id = "session-abc" )
    )
    
    with caplog.at_level( logging.INFO, logger = "agent.metrics.test" ):
        await plugin.before_run_callback( invocation_context = cast( InvocationContext, ctx ) )
        await plugin.after_run_callback( invocation_context = cast( InvocationContext, ctx ) )
    
    lines = [ r.getMessage() for r in caplog.records ]
    assert len( lines ) == 1, f"expected exactly one aggregate line, got: {lines}"
    line = lines[ 0 ]
    assert "latency_ms=500" in line
    assert "session_id=session-abc" in line
    assert "inv=inv-123" in line

async def test_metrics_state_is_popped_after_run():
    
    clock = FakeClock( [ 0.0, 1.0 ] )
    plugin = MetricsPlugin( clock = clock )
    ctx = _FaceInvocationContext( invocation_id = "inv-x", session = _FakeSession( id = "s" ) )
    await plugin.before_run_callback( invocation_context = cast( InvocationContext, ctx ) )
    await plugin.after_run_callback( invocation_context = cast( InvocationContext, ctx ) )
    
    assert "inv-x" not in plugin._runs
    assert "inv-x" not in plugin._sessions