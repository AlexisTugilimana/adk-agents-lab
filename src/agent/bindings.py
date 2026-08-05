"""The switch: the only module that reads `Profile` the agent configuration 
called `AgentBindings`.

Every consumer of `AgentBindings` are profile-blind, making the downstream code 
(cloud and local) identical and modular.

MCP provider only concerns local agent.
"""
import time
from dataclasses import dataclass

from agent.config import AgentConfig, Profile
from agent.ports import (
    Clock,
    CredentialsCheck,
    MemoryProvisioning,
    ServerTelemetry,
    SessionProvisioning,
    ToolProvisioning
)
from agent.sessions import LocalSessions, ManagedSessions
from agent.telemetry import CloudServerTelemetry, LocalServerTelemetry
from agent.tools.composite import CompositeTools
from agent.tools.native import NativeToolProvider

# ==============================================================================
# The bundle every entrypoint receives
# ==============================================================================
@dataclass
class AgentBindings:
    """Fully-resolved set of adapters for one build. Assembled there, consumed 
    by `assemble_adk_app` with zero profile awareness."""
    
    tools: ToolProvisioning
    sessions: SessionProvisioning
    memory: MemoryProvisioning
    telemetry: ServerTelemetry
    credentials: CredentialsCheck
    clock: Clock

# ==============================================================================
# Tiny adapters - Small enough so defined here.
# ==============================================================================
class SystemClock( Clock ):
    """``Perf_counter` used here because we only diff for latency."""
    
    def now( self ) -> float:
        return time.perf_counter()

class NoMemory( MemoryProvisioning ):
    """Memory is deferred here. `None` under local and cloud profile."""
    
    def builder( self ) -> None:
        return None

class AdcCredentials( CredentialsCheck ):
    """Local credentials checker via Application default credentials.
    
    Although credentials are checked via environment variables, never uses `os` 
    directly, so everything is unit-testable."""
    
    def __init__( self, cfg: AgentConfig ) -> None:
        self._cfg = cfg
    
    def verify( self ) -> str:
        cfg = self._cfg
        _SETUP_INT = (
            "   gcloud auth application-default login\n"
            "   export GOOGLE_GENAI_USE_VERTEXAI=TRUE\n"
            "   export GOOGLE_CLOUD_PROJECT=your-project-id\n"
            "   export GOOGLE_CLOUD_MODEL_LOCATION=your-location\n"
        )
        if not cfg.use_vertex:
            raise SystemExit( f"No credentials found. Set up ADC: \n {_SETUP_INT}" )
        if not cfg.project:
            raise SystemExit( f"GOOGLE_CLOUD_PROJECT is not set.\n {_SETUP_INT}" )
        if not cfg.location:
            raise SystemExit( f"GOOGLE_CLOUD_LOCATION is not set.\n {_SETUP_INT}" )
        return f"Vertex AI (project={cfg.project} location={cfg.location})"

class ServiceAccountCredentials( CredentialsCheck ):
    """Cloud credentials check. Auth checking fully handled by the cloud provider."""
    
    def verify( self ) -> str:
        return "Vertex AI (Agent Engine account; implicit credentials)"
    
# ==============================================================================
# Switch local/cloud
# ==============================================================================
def build_agent_bindings( cfg: AgentConfig ) -> AgentBindings:
    """Resolve `cfg.config` into a concrete `AgentBindings`. """
    
    # Local
    if cfg.profile is Profile.LOCAL:
        
        providers: list[ ToolProvisioning ] = [ NativeToolProvider() ]
        if cfg.enable_mcp:
            from agent.tools.mcp import SERVERS, McpToolProvider
            providers.append( McpToolProvider( SERVERS ) )
        
        return AgentBindings(
            tools = CompositeTools( providers ),
            sessions = LocalSessions( cfg.session_backend, cfg.db_url ),
            memory = NoMemory(),
            telemetry = LocalServerTelemetry(),
            credentials = AdcCredentials( cfg ),
            clock = SystemClock()
        )
    
    # Cloud
    return AgentBindings(
        tools = CompositeTools( [ NativeToolProvider() ] ),
        sessions = ManagedSessions(),
        memory = NoMemory(),
        telemetry = CloudServerTelemetry( cfg.project, cfg.location ),
        credentials = ServiceAccountCredentials(),
        clock = SystemClock()
    )