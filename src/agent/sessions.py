"""Session provisioning + builders.

Builders are zero-arg callables. `AdkApp` calls them itself since we provide a 
callable. Locally, we provide a builder directly, in the cloud it's none as cloud 
manages itself the sessions.
"""

import logging

from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from agent.config import SessionBackend
from agent.ports import SessionProvisioning, SessionServiceBuilder

log = logging.getLogger( "agent.sessions" )

def normalize_sqlite_url( url: str ) -> str:
    """Ensure SQLite URL uses async `aiosqlite` driver ADK requires."""
    if url.startswith( "sqlite://" ) and "+aiosqlite" not in url:
        fixed = url.replace( "sqlite://", "sqlite+aiosqlite://", 1 )
        log.warning(
            f"session.db_url.normalized from={url} to={fixed} (async driver required)"
        )
        return fixed
    return url

def in_memory_session_builder() -> SessionServiceBuilder:
    """Zero-arg builder for a in-memory session service."""
    def _build() -> InMemorySessionService:
        return InMemorySessionService()
    return _build

def sqlite_session_builder( db_url: str ) -> SessionServiceBuilder:
    """Zero-arg builder for a persitent sqlite session service."""
    
    url = normalize_sqlite_url( db_url )
    
    def _build() -> BaseSessionService:
        from google.adk.sessions.database_session_service import DatabaseSessionService
        
        return DatabaseSessionService( db_url = url )
    return _build

class LocalSessions( SessionProvisioning ):
    """Local session provisioning: in-memory (default) or sqlite (persistent)."""
    
    def __init__( self, backend: SessionBackend, db_url: str ) -> None:
        self._backend = backend
        self._db_url = db_url
    
    def builder( self ) -> SessionServiceBuilder:
        if self._backend is SessionBackend.SQLITE:
            return sqlite_session_builder( self._db_url )
        return in_memory_session_builder()

class ManagedSessions( SessionProvisioning ):
    """Cloud session provisioning: since cloud owns the sessions when 
    `GOOGLE_CLOUD_AGENT_ENGINE_ID` is present, we hand `AdkApp` to `None`."""
    
    def builder( self ):
        return None
        