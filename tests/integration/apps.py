"""Building/fetching the applications that run locally on in the cloud."""

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from typing import Any, Iterator

import pytest

from agent.config import AgentConfig, Profile, SessionBackend

USER_ID = "it-user"

# ==============================================================================
# Environment preconditions
# ==============================================================================
def _missing_local_env() -> list[ str ]:
    missing = []
    if os.environ.get( "GOOGLE_GENAI_USE_VERTEXAI", "" ).lower() not in { "1", "true", "yes" }:
        missing.append( "GOOGLE_GENAI_USE_VERTEXAI" )
    if not os.environ.get( "GOOGLE_CLOUD_PROJECT", "" ):
        missing.append( "GOOGLE_CLOUD_PROJECT" )
    if not os.environ.get( "GOOGLE_CLOUD_LOCATION", "" ):
        missing.append( "GOOGLE_CLOUD_LOCATION" )
    return missing

def require_local_env() -> None:
    """Fail loudly unless ADC and model env are configured."""
    
    missing = _missing_local_env()
    if missing:
        pytest.fail( f"need ADC + env; missing {', '.join( missing )}." )

# ==============================================================================
# Config
# ==============================================================================
def local_config( **overrides: Any ):
    """Local config the smoke uses - native only, in memory - with optional overrides."""
    
    require_local_env()
    
    base = replace(
        AgentConfig.from_env( Profile( "local" ) ),
        session_backend = SessionBackend.MEMORY,
        enable_mcp = False
    )
    return replace( base, **overrides ) if overrides else base

# ==============================================================================
# Live apps
# ==============================================================================
@asynccontextmanager
async def local_agent( cfg: AgentConfig | None = None ):
    """Yield a live in-process `AdkApp` with the tool lifecycle managed.
    
    Lazy import of `local_runtime`.
    """
    
    from agent.local_runtime import local_adk_app
    
    if cfg is None:
        cfg = local_config()
    
    try:
        async with local_adk_app( cfg ) as app:
            yield app
    except SystemExit as exc:
        pytest.fail( f"credentials not configured: {exc}" )
    except Exception as exc:
        pytest.fail( f"local AdkApp could not be built: {exc}" )
        raise

def _resolved_deployed( resource_name: str ) -> Any:
    """Resolve the engine once per run."""
    
    import vertexai
    from vertexai import agent_engines
    
    cfg = AgentConfig.from_env( Profile( "cloud" ) )
    vertexai.init( project = cfg.project, location = cfg.engine_location )
    return agent_engines.get( resource_name )

def deployed_agent() -> Any:
    """Handle to the deployed agent engine."""
    
    require_local_env()
    
    resource_name: str = os.environ.get( "AGENT_RESOURCE_NAME", "" ).strip()
    if not resource_name:
        pytest.fail(
            f"AGENT_RESOURCE_NAME not set; deploy first and export it."
        )
    return _resolved_deployed( resource_name )

# ==============================================================================
# Log capture
# ==============================================================================
def capture_logs(
    logger_name: str, level: int = logging.INFO
) -> Iterator[ list[ logging.LogRecord ] ]:
    """Collect records emitted by `logger_name` for the duration of the block."""
    
    logger = logging.getLogger( logger_name )
    records: list[ logging.LogRecord ] = []
    
    class _Collector( logging.Handler ):
        def emit( self, record: logging.LogRecord ) -> None:
            records.append( record )
    
    handler = _Collector()
    previous_level = logger.level
    logger.addHandler( handler )
    logger.setLevel( level )
    try:
        yield records
    finally:
        logger.removeHandler( handler )
        logger.setLevel( previous_level )