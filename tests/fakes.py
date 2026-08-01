"""Fakes for implementing ports for the tests."""

from typing import Any

from agent.ports import (
    Clock,
    ToolProvisioning,
    GatingSpec
)

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
        
class FakeClock( Clock ):
    """Deterministic `now()` function returning value."""
    
    def __init__( self, values: list[ float ] ) -> None:
        self._values = list( values )
        self._i = 0
    
    def now( self ) -> float:
        value = self._values[ min( self._i, len( self._values ) -1 ) ]
        self._i += 1
        return value