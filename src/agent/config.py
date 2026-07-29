from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

# ==============================================================================
# Profile and backend
# ==============================================================================
class Profile( str, Enum ):
    
    LOCAL = "local"
    CLOUD = "cloud"
    
class SessionBackend( str, Enum ):
    """Where local conversation history lives.
    
    Ignore under the CLOUD profile.
    """
    
    MEMORY = "memory"   # InMemorySessionService - fast, ephemeral
    SQLITE = "sqlite"   # DatabaseSessionService - persisent (local)
    
class CompactionTrigger( str, Enum ):
    """ When the runner should compact history.
    
    off         -> never (behavourial identical to no compaction)
    interval    -> after every `compaction_interval` user invocations
    token       -> when the last prompt's token count crosses `token_threshold`
    hybrid      -> whichever of the two fire first 
    """
    
    OFF = "off"
    INTERVAL = "interval"
    TOKEN = "token"
    HYBRID = "hybrid"
    
# ==============================================================================
# Constants
# ==============================================================================
APP_NAME = "mcp-chat"
AGENT_NAME = "mcp_agent"
MODEL = "gemini-3.5-flash"  # default, overridable via AgentConfig.model

INSTRUCTIONS = (
    "You are a helpful assistant whose abilities come from connected MCP "
    "tools. Inspect the tools available to you and use them when they fit the "
    "user's request; otherwise answer directly. Be concise, and when you use "
    "a tool, briefly say what it returned. Some tools require human approval "
    "before they run; if an approval is rejected, report that plainly and do "
    "not retry the same action."
)

# ADK fixed name for long-running confirmation call. 
CONFIRMATION_FUNCTION_NAME = "adk_request_confirmation"
CONFIRMATION_HINT = "This tool requires your approval before it runs."

# Default SQLite URL
DEFAULT_DB_URL = "sqlite+aiosqlite:///mcp_chat_session_db"

# --- Compaction defaults ---
DEFAULT_COMPACTION_INTERVAL = 3 # compact every N user invocations
DEFAULT_COMPACTION_OVERLAP = 1  # invocations kept for context overlap
DEFAULT_COMPACTION_TOKEN_THRESHOLD = 8000   # prompt-token trigger (token/hybrid)
DEFAULT_COMPACTION_EVENT_RETENTION = 6  # raw events kept after a token compaction

# ==============================================================================
# Compaction settings
# ==============================================================================
@dataclass
class CompactionSettings:
    """Framework-agnostic description of the compaction policy."""
    
    trigger: CompactionTrigger = CompactionTrigger.OFF
    compaction_interval: int = DEFAULT_COMPACTION_INTERVAL
    overlap_size: int = DEFAULT_COMPACTION_OVERLAP
    token_threshold: int = DEFAULT_COMPACTION_TOKEN_THRESHOLD
    event_retention_size: int = DEFAULT_COMPACTION_EVENT_RETENTION
    summarizer_model: str | None = None
    summarize_prompt: str | None = None
    
    @property
    def enabled( self ) -> bool:
        return self.trigger is not CompactionTrigger.OFF
    
    @property
    def uses_interval( self ) -> bool:
        return self.trigger in ( CompactionTrigger.INTERVAL, CompactionTrigger.HYBRID )
    
    @property
    def uses_token( self ) -> bool:
        return self.trigger in ( CompactionTrigger.TOKEN, CompactionTrigger.HYBRID )
    
    def __post_init__( self ) -> None:
        """Bad compaction policy fails loudly here."""
        
        if not self.enabled:
            return
        
        if self.uses_interval:
            if self.compaction_interval < 1:
                raise ValueError( "compaction_interval must be >= 1" )
            if self.overlap_size < 0:
                raise ValueError( "overlap_size must be >=0" )
            if self.overlap_size >= self.compaction_interval:
                raise ValueError(
                    "overlap_size must be smaller than compaction_interval "
                    f"(got overlap={self.overlap_size}) interval={self.compaction_interval})"
                )
        if self.uses_token:
            if self.token_threshold < 1:
                raise ValueError( "token_threshold must be >= 1" )
            if self.event_retention_size < 0:
                raise ValueError( "event_retention_size must be >= 0" )
        
    def describe( self ) -> str:
        """One-line summary for startup logging."""
        
        if not self.enabled:
            return "compaction disabled"
        who = self.summarizer_model or f"{MODEL} (default summarizer)"
        parts: list[ str ] = []
        if self.uses_interval:
            parts.append(
                f"every {self.compaction_interval} turns (overlap {self.overlap_size})"
            )
        if self.uses_token:
            parts.append(
                f"or >= {self.token_threshold} prompt tokens "
                f"(keep last {self.event_retention_size})"
            )
        return f"compaction {self.trigger.value} {' '.join( parts )} via {who}"

# ==============================================================================
# AgentConfig
# ==============================================================================
@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for one agent build.
    
    Everything an adapter needs - model, session wiring, etc - is read here 
    off the instance. It's the single source of thruth.
    """
    
    profile: Profile
    model: str = MODEL
    app_name: str = APP_NAME
    instructions: str = INSTRUCTIONS
    
    # --- local session wiring (ignored in cloud) ---
    session_backend: SessionBackend = SessionBackend.MEMORY
    db_url: str = DEFAULT_DB_URL
    
    # --- Compaction ---
    compaction: CompactionSettings = field( default_factory = CompactionSettings )

    # --- Cloud identity / auth ---
    use_vertex: bool = False    # from GOOGLE_GENAI_USE_VERTEXAI Consumed by AddCredentials
    project: str | None = None  # from GOOGLE_CLOUD_PROJECT
    location: str | None = None # from GOOGLE_CLOUD_LOCATION
    
    # --- MCP ---
    enable_mcp: bool = False    # Local sets TRUE, Cloud leaves FALSE
    
    @classmethod
    def from_env( cls, profile: Profile ) -> AgentConfig:
        """Build `AgentConfig` for `profile` from the process environment."""

        thruthy = { "1", "true", "yes" }
        return cls(
            profile = profile,
            use_vertex = os.environ.get( "GOOGLE_GENAI_USE_VERTEXAI", "" ).lower() in thruthy,
            project = os.environ.get( "GOOGLE_CLOUD_PROJECT" ),
            location = os.environ.get( "GOOGLE_CLOUD_LOCATION" ),
            session_backend = SessionBackend(
                os.environ.get( "AGENT_SESSION_BACKEND", SessionBackend.MEMORY.value )
            ),
            db_url = os.environ.get( "AGENT_DB_URL", DEFAULT_DB_URL ),
            enable_mcp = os.environ.get( "AGENT_ENABLE_MCP", "" ).lower() in thruthy
        )
    