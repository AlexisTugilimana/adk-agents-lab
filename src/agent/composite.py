"""CompositeTools - Treat more than one source of tools as a single one.

Layer: adapter. This class let cloud be `CompositeTools([Native])` and local 
be `CompositeTools([Native, Mcp])` with identical downstream code.

`gating()` is a merge, so a native tool and an MCP tool having the same runtime 
name are flagged ambiguous.
"""
import logging

from google.adk.agents.llm_agent import ToolUnion

from agent.policy import GatingSpec, merged_all
from agent.ports import ToolProvisioning

log = logging.getLogger( "agent.tools" )

class CompositeTools( ToolProvisioning ):
    """Compose several tool providers into one."""
    
    def __init__( self, providers: list[ ToolProvisioning ] ) -> None:
        self._providers = providers
        self._started: list[ ToolProvisioning ] = []
    
    async def setup( self ) -> None:
        for provider in self._providers:
            try:
                await provider.setup()
                self._started.append( provider )
            except Exception:
                log.exception(
                    f"composite.setup.error provider={type(provider).__name__}"
                )
                # Roll back what we already startedn then re-raise the error.
                await self._teardown_started()
                raise
            
    async def teardown( self ) -> None:
        await self._teardown_started()
    
    async def _teardown_started( self ) -> None:
        # Reverse order so teardown mirrors setup.
        while self._started:
            provider = self._started.pop()
            try:
                await provider.teardown()
            except Exception:
                log.exception(
                    f"composite.teardown.error provider={type(provider).__name__}"
                )
    
    def tools( self ) -> list[ ToolUnion ]:
        collected: list[ ToolUnion ] = []
        for provider in self._providers:
            collected.extend( provider.tools() )
        return collected
    
    def gating( self ) -> GatingSpec:
        return merged_all( [ provider.gating() for provider in self._providers ] )