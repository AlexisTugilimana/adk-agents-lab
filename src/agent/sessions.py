"""Session provisioning + builders.

Builders are zero-arg callables. `AdkApp` calls them itself since we provide a 
callable. Locally, we provide a builder directly, in the cloud it's none as cloud 
manages itself the sessions.
"""

import logging
from typing import Callable

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

def sqlite_session_builder(
    db_url: str,
    on_created: Callable[ [ BaseSessionService ], None ] | None = None
) -> SessionServiceBuilder:
    """Zero-arg builder for a persitent sqlite session service."""
    
    url = normalize_sqlite_url( db_url )
    
    def _build() -> BaseSessionService:
        from google.adk.sessions.database_session_service import DatabaseSessionService
        
        service = DatabaseSessionService( db_url = url )
        if on_created is not None:
            on_created( service )
        return service
    
    return _build

class LocalSessions( SessionProvisioning ):
    """Local session provisioning: in-memory (default) or sqlite (persistent)."""
    
    def __init__( self, backend: SessionBackend, db_url: str ) -> None:
        self._backend = backend
        self._db_url = db_url
        self._created: list[ BaseSessionService ] = []
    
    def builder( self ) -> SessionServiceBuilder:
        if self._backend is SessionBackend.SQLITE:
            return sqlite_session_builder( self._db_url, on_created = self._created.append )
        return in_memory_session_builder()
    
    async def teardown( self ) -> None:
        """Dispose every session service this adapter's builder created.
        
        For sqlite this reaches `DatabaseSessionService.close()`, which calls 
        `db_engine.dispose()` and clooses the connection.
        
        Use `get_attr()` to get the close function in case the session service 
        does not expose the `close()` in its contract, or does not implement it.
        """
        
        pending, self._created = self._created, []
        for service in reversed( pending ):
            close = getattr( service, "close", None )
            if close is None:
                continue
            try:
                await close()
            except Exception:
                log.exception( f"session.teardown.error service={type(service).__name__}" )

class ManagedSessions( SessionProvisioning ):
    """Cloud session provisioning: since cloud owns the sessions when 
    `GOOGLE_CLOUD_AGENT_ENGINE_ID` is present, we hand `AdkApp` to `None`."""
    
    def builder( self ):
        return None
    
    async def teardown( self ) -> None:
        """Nothing to release on the cloud."""
        return None