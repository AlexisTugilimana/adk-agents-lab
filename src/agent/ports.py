"""Abstract contracts defining what the agent needs.

This modules defines the application boundary (tools, sessions, memory, 
telemetry, credentials, and a clock). Concrete adapters implement the ports and 
enable swapping easily from local to cloud.

Two kinds of seam live here:

    * Behaviour ports: `ToolProvisioning`, `SessionProvisioning`, `MemoryProvisioning`, 
    `ServerTelemetry`, `CredentialsCheck`, `Clock`. Implemented using `abc.ABC` 
    with `@abstractmethod`.
    
    * Builder seams: `SessionServiceBuilder` and `MemoryServiceBuilder` as `AdkApp` 
    calls for such contract.

Ports.py still import `google.adk` `ToolUnion`, `BaseMemoryService`, and `BaseSessionService`. 
They are pure type alias and abstract base classes. Implementing ourselves them 
would be over-engineering.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable

from google.adk.agents.llm_agent import ToolUnion
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.base_session_service import BaseSessionService

from agent.policy import GatingSpec

# ==============================================================================
# Callable seams
# ==============================================================================
SessionServiceBuilder = Callable[ [], BaseSessionService ]
MemoryServiceBuilder = Callable[ [], BaseMemoryService | None ]

# ==============================================================================
# Behaviour ports
# ==============================================================================
class ToolProvisioning( ABC ):
    """A source of tool plus the gating it declares.
    
    Bundles the whole lifecycle of a tool source behind four methods so the rest 
    of the application can treat everysource (hardcoded py, spawned MCP, etc) 
    identically.
    
    Lifecycle: for MCP or HTTP connectionn ask for what tools are available, 
    later shut down cleanly. For native tool, no-op here.
    
    Metadata: provides GatingSpec for safety and compile policy.
    """
    
    @abstractmethod
    async def setup( self ) -> None:
        """MCP: connect + discover. Native: no-op."""
    
    @abstractmethod
    async def teardown( self ) -> None:
        """MCP: close subprocesses. Native: no-op. Safe to call upon failure."""
    
    @abstractmethod
    def tools( self ) -> list[ ToolUnion ]:
        """The tools handed to the agent."""
    
    @abstractmethod
    def gating( self ) -> GatingSpec:
        """Gating this source declares, merged into the confirmation policy."""

class SessionProvisioning( ABC ):
    """Where conversation lives."""
    
    @abstractmethod
    def builder( self ) -> SessionServiceBuilder | None:
        """Locally: a zero-arg builder. Cloud: None because owns the session."""

class MemoryProvisioning( ABC ):
    """Long-term memory. Deferred for now."""
    
    @abstractmethod
    def builder( self ) -> MemoryServiceBuilder | None:
        ...
    
class ServerTelemetry( ABC ):
    """Server-side telemetry: tracing and logging inside the serving process.
    
    Requires an instrumentor to handle both local and cloud telemetry:
    
        * Locally:  install the log formatter onto the root logger. Step ahead 
                    of the process app being constructed.
        * Cloud:    start OpenTelemetry trace exporter. Provide to AdkApp the 
                    callable to run at the right time on the container.
    
    Both must happen inside the process that actually 
    """
    
    @property
    @abstractmethod
    def enabled_tracing( self ) -> bool:
        ...
    
    @abstractmethod
    def instrumentor_builder( self ) -> Callable[ ..., Any ]:
        """Returns a callable that `AdkApp` runs later, inside the runtime 
        to install the log formatter (locally) or the trace exporter (cloud)."""
        
class CredentialsCheck( ABC ):
    """Profile-aware credential verification."""
    
    @abstractmethod
    def verify( self ) -> str:
        """Return a human label. Local may raise `SystemExit`; cloud never raises."""

class Clock( ABC ):
    """A monotonic clock, injected so `MetricsPlugin` latency is deterministic."""
    
    @abstractmethod
    def now( self ) -> float:
        ...