"""Single construction site for the app object.

Builds the app object from `(cfg, bindings)` with zero-profile conditionals. 
"""
from typing import Protocol, Any

from agent.app_factory import build_app
from agent.bindings import AgentBindings
from agent.config import AgentConfig
from agent.policy import compile_policy

class AdkAppFactory( Protocol ):
    """Construction contract `assemble_adk_app`."""
    
    def __call__(
        self,
        *,
        app: Any,
        session_service_builder: Any,
        enable_tracing: bool,
        instrumentor_builder: Any
    ) -> Any: ...
    
def assemble_adk_app(
    cfg: AgentConfig,
    bindings: AgentBindings,
    *,
    adk_app_factory: AdkAppFactory
) -> Any:
    """Assemble the deployable/local app object."""
    
    # Check credentials first
    bindings.credentials.verify()
    
    # Build app
    app = build_app(
        cfg,
        tools = bindings.tools.tools(),
        policy =  compile_policy( bindings.tools.gating() ),
        clock = bindings.clock
    )
    return adk_app_factory(
        app = app,
        session_service_builder = bindings.sessions.builder(),
        enable_tracing = bindings.telemetry.enabled_tracing,
        instrumentor_builder = bindings.telemetry.instrumentor_builder()
    )
