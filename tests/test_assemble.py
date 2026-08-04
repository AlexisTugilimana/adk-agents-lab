"""Assembly wiring test."""

from typing import cast
import pytest

from agent.ports import SessionServiceBuilder
from agent.assemble import assemble_adk_app
from agent.bindings import AgentBindings
from agent.config import AgentConfig, Profile
from agent.policy import GatingSpec
from tests.fakes import (
    FakeClock,
    FakeCredentials,
    FakeMemory,
    FakeSessions,
    FakeTelemetry,
    FakeTools,
    RecordAdkAppStub
)

def _local_bindings( creds: FakeCredentials, telemetry: FakeTelemetry ) -> AgentBindings:
    builder = cast( SessionServiceBuilder, lambda : object() )
    return AgentBindings(
        tools = FakeTools( GatingSpec.empty() ),
        sessions = FakeSessions( builder = builder ),
        memory = FakeMemory(),
        telemetry = telemetry,
        credentials = creds,
        clock = FakeClock( [ 0.0 ] )
    )

def _cloud_bindings( creds: FakeCredentials ) -> AgentBindings:
    return AgentBindings(
        tools = FakeTools( GatingSpec.empty() ),
        sessions = FakeSessions( builder = None ),
        memory = FakeMemory(),
        telemetry = FakeTelemetry( enabled_tracing = True ),
        credentials = creds,
        clock = FakeClock( [ 0.0 ] )
    )

def test_local_wiring_tracing_off_and_concrete_session_builder():
    cfg = AgentConfig( profile = Profile.LOCAL )
    creds = FakeCredentials()
    telemetry = FakeTelemetry( enabled_tracing = False)
    bindings = _local_bindings( creds, telemetry )
    
    stub = assemble_adk_app( cfg, bindings, adk_app_factory = RecordAdkAppStub )
    
    assert creds.verify_calls == 1  # credentials verified before build
    assert stub.kwargs[ "enable_tracing" ] is False
    assert stub.kwargs[ "session_service_builder" ] is not None
    assert stub.kwargs[ "app" ] is not None
    assert telemetry.instrumentor_calls == 1
    
def test_cloud_wiring_flips_both_through_the_same_function_no_branch():
    cfg = AgentConfig( profile = Profile.CLOUD )
    creds = FakeCredentials()
    bindings = _cloud_bindings( creds )
    
    stub = assemble_adk_app( cfg, bindings, adk_app_factory = RecordAdkAppStub )
    
    assert stub.kwargs[ "enable_tracing" ] is True
    assert stub.kwargs[ "session_service_builder" ] is None

def test_local_credential_failure_propagates_before_build():
    cfg = AgentConfig( profile = Profile.LOCAL )
    creds = FakeCredentials( raises = True )
    telemetry = FakeTelemetry( enabled_tracing = False)
    bindings = _local_bindings( creds, telemetry )
    
    with pytest.raises( SystemExit ):
        assemble_adk_app( cfg, bindings, adk_app_factory = RecordAdkAppStub )