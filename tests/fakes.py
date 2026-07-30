"""Fakes for implementing ports for the tests."""

from agent.ports import Clock

class FakeClock( Clock ):
    """Deterministic `now()` function returning value."""
    
    def __init__( self, values: list[ float ] ) -> None:
        self._values = list( values )
        self._i = 0
    
    def now( self ) -> float:
        value = self._values[ min( self._i, len( self._values ) -1 ) ]
        self._i += 1
        return value