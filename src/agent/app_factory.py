"""Single ADK `App` builder.

Only place where the ADK `App` (agent + plugins + resumability + compaction) is 
constructed, for local and cloud. Fully profile-independent.
"""
import logging
from typing import Callable

from google.adk.agents import Agent
from google.adk.agents.llm_agent import ToolUnion
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

from agent.config import AGENT_NAME, MODEL, AgentConfig, CompactionSettings
from agent.plugins import ConfirmationPolicyPlugin, MetricsPlugin
from agent.policy import ConfirmationPolicy
from agent.ports import Clock

log = logging.getLogger( "agent.app" )

def compose_instruction( cgf: AgentConfig ) -> str:
    """Assemble the agent instruction. Trivial for now but to be modified when 
    memory will be included."""
    
    return cgf.instructions

def build_summarizer( settings: CompactionSettings ) -> BaseEventsSummarizer | None:
    """Build a custom event summarizer only when the user asked for it."""
    
    if settings.summarizer_model is None and settings.summarize_prompt is None:
        return None
    
    model_name = settings.summarizer_model or MODEL
    summarizer = LlmEventSummarizer(
        llm = Gemini( model = model_name ), prompt_template = settings.summarize_prompt
    )
    log.info(
        f"compaction.summarizer model={model_name} custom_prompt={settings.summarize_prompt is not None}"
    )
    return summarizer

def build_events_compaction_config( settings: CompactionSettings ) -> EventsCompactionConfig | None:
    """Translate `CompactionSettings` into ADK's `EventsCompactionConfig`.
    
    Return None when compaction is disabled. When only token-triggering is selected, 
    the interval is set to a very large number so it never fiers on turn count alone.
    """
    
    if not settings.enabled:
        return None
    
    kwargs: dict = {
        "compaction_interval": settings.compaction_interval if settings.uses_interval else 10**9,
        "overlap_size": settings.overlap_size,
        "summarizer": build_summarizer( settings )
    }
    if settings.uses_token:
        kwargs[ "token_threshold" ] = settings.token_threshold
        kwargs[ "event_retention_size" ] = settings.event_retention_size
    
    config = EventsCompactionConfig( **kwargs )
    log.info( f"compaction.config {settings.describe()}" )
    return config

def build_app(
    cfg: AgentConfig,
    *,
    tools: list[ ToolUnion ],
    policy: ConfirmationPolicy,
    clock: Clock,
    memory_tools: list[ ToolUnion ] | None = None,  # empty for now. For memory only
    after_agent_callback: Callable | None = None    # empty for now. For memory only
) -> App:
    """Construct the ADK `App` used for local and deployed processes.
    
    Plugins order matter: `ConfirmationPolicyPlugin` before `MetricsPlugin`.
    """
    
    agent = Agent(
        name = AGENT_NAME,
        model = cfg.model,
        instruction = compose_instruction( cfg ),
        tools = [ *tools, *( memory_tools or [] ) ],
        after_agent_callback = after_agent_callback
    )
    plugins = [ ConfirmationPolicyPlugin( policy ), MetricsPlugin( clock = clock ) ]
    return App(
        name = cfg.app_name,
        root_agent = agent,
        plugins = plugins,
        resumability_config = ResumabilityConfig( is_resumable = True ),
        events_compaction_config = build_events_compaction_config( cfg.compaction )
    )