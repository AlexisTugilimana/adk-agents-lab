"""Server-side ADK plugins.

Plugins are registered to the App and therefore run inside the runtime 
server-side part of the application (not the client). It owns the HITL authority 
and the metric management.

Two plugins lives here:

    * `ConfirmationPolicyPlugin`:   global HITL gate which decides whether a tool 
                                    requires a human approval.
    * `MetricsPlugin`:              pure observation aggregating log line per 
                                    invocation.
"""
import logging
from dataclasses import dataclass
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai.types import Content

from agent.config import CONFIRMATION_HINT
from agent.policy import ConfirmationPolicy
from agent.ports import Clock

# ==============================================================================
# Confirmation policy plugin
# ==============================================================================
class ConfirmationPolicyPlugin( BasePlugin ):
    """Global HITL gate.
    
    Registered once on the App such that it applies to every agent/tool and cannot 
    be bypassed by an agent. `ConfirmationPolicy` is the only decision gate.
    """
    
    def __init__( self, policy: ConfirmationPolicy ) -> None:
        super().__init__( name = "confirmation_policy" )
        self._policy = policy
        self._log = logging.getLogger( "agent.confirmation" )
        
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        
        # Not gated -> let run the tool
        if not self._policy.requires_confirmation( tool.name ):
            self._log.info( f"confirmation.bypass tool={tool.name}" )
            return None
        
        confirmation = tool_context.tool_confirmation
        if confirmation is None:
            # First encountered: ask the human. The runtime turns this into an 
            # `adk_request_confirmation` long-running call.
            tool_context.request_confirmation(
                hint = CONFIRMATION_HINT,
                payload = { "tool": tool.name, "args": tool_args or {} }
            )
            self._log.info( f"confirmation.request tool={tool.name}" )
            return { "status": "pending", "message": f"Awaiting for approval for {tool.name!r}" }
        
        if confirmation.confirmed:
            self._log.info( f"confirmation.approved tool={tool.name}" )
            return None

        self._log.info( f"confirmation.rejected tool={tool.name}" )
        return { "status": "rejected", "message": f"{tool.name} was not approved; skipped." }

# ==============================================================================
# Metrics plugin
# ==============================================================================
@dataclass
class RunMetrics:
    """Mutable per-invocation accumulator."""
    
    start: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    model_errors: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
@dataclass
class MetricsPlugin( BasePlugin ):
    """Pure observation plugin: collects metrics, emits one aggegate summary 
    per invocation.
    
    Logger and Clock are injected because this plugin's product - the one metrics 
    line - is defined by latency (need a controllable clock) and its content (need 
    observable logger).
    """
    
    def __init__( self, *, clock: Clock, logger: logging.Logger | None = None ) -> None:
        super().__init__( name = "metrics" )
        self._clock = clock # Injected such that we never run time.perf() and we can mock.
        self._log = logger or logging.getLogger( "agent.metrics" )
        self._runs: dict[ str, RunMetrics ] = {}
        # Cache session ID per invocation ID because error hooks do not expose 
        # session ID directly (only invocation ID). Keyed by invocation ID in case 
        # multiple invocations run at the same time (useful if Agent Engine serves 
        # multiple users concurrently).
        self._sessions: dict[ str, str ] = {}
    
    def _run( self, invocation_id: str ) -> RunMetrics:
        return self._runs.setdefault( invocation_id, RunMetrics() )
    
    @staticmethod
    def _session_id( ctx: InvocationContext ) -> str:
        session = getattr( ctx, "session", None )
        return getattr( session, "id", "" ) if session is not None else ""
    
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Content | None:
        run = self._run( invocation_context.invocation_id )
        run.start = self._clock.now()
        # Cache session ID
        self._sessions[ invocation_context.invocation_id ] = self._session_id( invocation_context )
        return None
    
    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        usage = getattr( llm_response, "usage_metadata", None )
        if usage:
            run = self._run( callback_context.invocation_id )
            run.prompt_tokens += getattr( usage, "prompt_token_count", 0 ) or 0
            run.output_tokens += getattr( usage, "candidates_token_count", 0 ) or 0
            run.total_tokens += getattr( usage, "total_token_count", 0 ) or 0
        return None
    
    async def after_tool_callback(
        self, 
        *, 
        tool: BaseTool, 
        tool_args: dict[str, Any], 
        tool_context: ToolContext, 
        result: dict[str, Any]
    ) -> dict[str, Any] | None:
        self._run( tool_context.invocation_id ).tool_calls += 1
        return None
    
    async def on_tool_error_callback(
        self, 
        *, 
        tool: BaseTool, 
        tool_args: dict[str, Any], 
        tool_context: ToolContext, 
        error: Exception
    ) -> dict[str, Any] | None:
        invocation_id: str = tool_context.invocation_id
        self._run( invocation_id ).tool_errors += 1
        self._log.warning(
            f"tool.error session_id={self._sessions.get( invocation_id, "" )} "
            f"inv={invocation_id} tool={tool.name} err={error}"
        )
        return None
    
    async def on_model_error_callback(
        self, 
        *, 
        callback_context: CallbackContext, 
        llm_request: LlmRequest, 
        error: Exception
    ) -> LlmResponse | None:
        invocation_id: str = callback_context.invocation_id
        self._run( invocation_id ).model_errors += 1
        self._log.warning(
            f"model.error session_id={self._sessions.get( invocation_id, "" )} "
            f"inv={invocation_id} err={error}"
        )
        return None
    
    async def after_run_callback( self, *, invocation_context: InvocationContext ) -> None:
        invocation_id = invocation_context.invocation_id
        run = self._runs.pop( invocation_id, None )
        session_id = self._sessions.pop( invocation_id, self._session_id( invocation_context ) )
        if run is None:
            return None
        elapsed_ms = ( self._clock.now() - run.start ) * 1000
        self._log.info(
            f"metrics session_id={session_id} inv={invocation_id} latency_ms={elapsed_ms:.0f} "
            f"tool_calls={run.tool_calls} tool_errors={run.tool_errors} model_errors={run.model_errors} "
            f"tokens(prompt={run.prompt_tokens} output={run.output_tokens} total={run.total_tokens})"
        )
        return None