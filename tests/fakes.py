"""Fakes for implementing ports for the tests."""

from typing import Any, Callable

from agent.ports import (
    Clock,
    CredentialsCheck,
    MemoryProvisioning,
    ServerTelemetry,
    SessionProvisioning,
    SessionServiceBuilder,
    ToolProvisioning
)
from agent.policy import GatingSpec

class FakeTools( ToolProvisioning ):
    """Return no tools and a caller-supplied GatingSpec. Records lifecycle calls 
    so tests can assert setup/teardown ordering."""
    
    def __init__( self, gating: GatingSpec | None = None ) -> None:
        self._gating = gating or GatingSpec.empty()
        self.setup_calls = 0
        self.teardown_calls = 0
        
    async def setup( self ) -> None:
        self.setup_calls += 1
    
    async def teardown( self ) -> None:
        self.teardown_calls += 1
    
    def tools( self ) -> list[ Any ]:
        return []

    def gating( self ) -> GatingSpec:
        return self._gating
        
class FakeSessions( SessionProvisioning ):
    """Return a builder or None (to emulate local vs cloud-managed)"""
    
    def __init__( self, builder: SessionServiceBuilder | None ) -> None:
        self._builder = builder
    
    def builder( self ) -> SessionServiceBuilder | None:
        return self._builder

class FakeMemory( MemoryProvisioning ):
    """Return a memory builder (set to None for now)"""
    
    def builder( self ) -> None:
        return None
    
class FakeTelemetry( ServerTelemetry ):
    """Records wether the instrumentor was requested and exposes settable tracing flag."""
    
    def __init__( self, enabled_tracing: bool = False ) -> None:
        self._enabled_tracing = enabled_tracing
        self.instrumentor_calls = 0
        self.instrumentor_ran = 0
        
    @property
    def enabled_tracing( self ) -> bool:
        return self._enabled_tracing
    
    def instrumentor_builder( self ) -> Callable[..., Any]:
        self.instrumentor_calls += 1
        
        def _instrumentor( *args: Any, **kwargs: Any ) -> None:
            self.instrumentor_ran += 1
        
        return _instrumentor

class FakeCredentials( CredentialsCheck ):
    """No-op by default; can be told to raise to emulate local ADC failure."""
    
    def __init__( self, *, raises: bool = False, label: str = "fake-creds" ) -> None:
        self._raises = raises
        self._label = label
        self.verify_calls = 0
        
    def verify( self ) -> str:
        self.verify_calls += 1
        if self._raises:
            raise SystemExit( "fake credentials failure" )
        return self._label
    
class FakeClock( Clock ):
    """Deterministic `now()` function returning value."""
    
    def __init__( self, values: list[ float ] ) -> None:
        self._values = list( values )
        self._i = 0
    
    def now( self ) -> float:
        value = self._values[ min( self._i, len( self._values ) -1 ) ]
        self._i += 1
        return value

class RecordAdkAppStub:
    """AdkApp constructor (could be created using the AdkAppFactory)."""
    
    def __init__(
        self, *, app: Any, session_service_builder: Any, enable_tracing: bool,
        instrumentor_builder: Any
    ) -> None:
        self.kwargs = {
            "app": app,
            "session_service_builder": session_service_builder,
            "enable_tracing": enable_tracing,
            "instrumentor_builder": instrumentor_builder,
        }