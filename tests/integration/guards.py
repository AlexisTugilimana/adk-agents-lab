"""Guards - deciding what to do when something under the test is not reachable.

There are two reasons the assertion cannot be checked:

    * Configuration:    no ADC, missing env variables, etc.
    * Non-determinism:  ex: the model declined to call a gated tool, which makes 
                        unobservable the information to be checked.

Here, this is CI, note evaluation. But rather than skipping tests because of missing 
ressources, loud failures is prefered (to be discussed further).
"""

from typing import Any
import pytest

from tests.integration.events import Event, calls_named
from tests.integration.hitl import GATED_TOOL, confirmation_request

# ==============================================================================
# Guards
# ==============================================================================
def require_tool_call( events: list[ Event ], tool: str = GATED_TOOL ) -> None:
    """Requie that the model actually called `tool`."""
    
    if not calls_named( events, tool ):
        pytest.fail(
            f"model did not call {tool!r}. Retries exhausted."
        )

def require_confirmation( events: list[ Event ] ) -> list[ dict[ str, Any ] ]:
    """Return the confirmation requests or apply failure policy."""
    
    requests = confirmation_request( events )
    if not requests:
        pytest.fail( 
            "no confirmation request emitted - the gate did not park the turn"
        )
    return requests

# Weak part
def require_batch( requests: list[ dict[ str, Any ] ], minimum: int = 2 ) -> None:
    """Require at least `minimum` gated tools.
    
    Up to the model to batch parallel calls.
    """
    
    if len ( requests ) < minimum:
        pytest.fail(
            f"model issued {len( requests )} gated call(s), needed {minimum}."
        )