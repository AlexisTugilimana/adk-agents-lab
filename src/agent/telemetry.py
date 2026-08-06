"""Server telemetry adapters.

Telemetry decides where and how to log/traces (compared to metrics pluging 
that decides what to emit.).

Server telemetry is profile-dependent: 

    * Local: LocalServerTelemetry ->    readable logs, no exporter, no trace.
    * Cloud: CloudServerTelemetry ->    JSON logs, tracing on, Cloud Trace
                                        exporter.
                                        
Telemetry vs metric pluging interaction:
MetricsPluging emit a line. Telemetry.py determines how it looks like

    * Local:    LocalServerTelemetry prints `12:03:07 INFO agent.metrics | ...` 
                in the terminal
    * Cloud:    installed JSON formatting. Same log lands in Cloud Logging as a 
                structured JSON payload that can be queried by session_id.
"""
import logging
import json
import sys
from typing import Any, Callable

from agent.ports import ServerTelemetry

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

def _install_human_logging() -> None:
    """Install a human-readable stderr log formatter.
    
    Uses `force=True` so re-running inside the serving process replaces any 
    handler ADK/Vertex may have started.
    """
    
    logging.basicConfig(
        level = logging.INFO,
        format = LOG_FORMAT,
        datefmt = DATE_FORMAT,
        stream = sys.stderr,
        force = True
    )
    logging.captureWarnings( True )
    
class _JsonLogFormatter( logging.Formatter ):
    """Render each record in a JSON line so cloud Logging capture a queryiable 
    `json_payload`. Without this, Cloud logging stores standard textPayload 
    without being able to query for specific field."""
    
    def format( self, record: logging.LogRecord ) -> str:
        payload = {
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime( record, "%Y-%m-%dT%H:%M:%S%z" )
        }
        if record.exc_info:
            payload[ "exception" ] = self.formatException( record.exc_info )
        
        return json.dumps( payload )
    
def _install_json_logging() -> None:
    """Install a JSON stdout log formatter for the cloud runtime."""
    # Capture sdtout logs from container's stdout stream.
    handler = logging.StreamHandler( stream = sys.stdout )
    handler.setFormatter( _JsonLogFormatter() )
    root = logging.getLogger()
    root.handlers[ : ] = [ handler ]
    root.setLevel( logging.INFO )
    logging.captureWarnings( True )
    
class LocalServerTelemetry( ServerTelemetry ):
    """Local telemetry: human logs, no tracing, no exporter."""
    
    @property
    def enabled_tracing( self ) -> bool:
        return False
    
    def instrumentor_builder( self ) -> Callable[ ..., Any ]:
        """Returned to AdkApp(instrumentor_builder=...); Vertex AI runs it 
        inside the in-process runtime at startup."""
        
        def _instrumentor( *args: Any, **kwargs: Any ) -> None:
            _install_human_logging()
        
        return _instrumentor
    
    def configure( self ) -> None:
        """Fallback in case the `local_adk_app` instrumentor does not fire early 
        enough in the local in-process app."""
        
        _install_human_logging()

class CloudServerTelemetry( ServerTelemetry ):
    """Cloud telemetry: JSON logs + Cloud Trace exporter."""
    
    def __init__( self, project: str | None, location: str | None ) -> None:
        self._project = project
        self._location = location
        
    @property
    def enabled_tracing( self ) -> bool:
        return True

    def instrumentor_builder( self ) -> Callable[..., Any]:
        
        def _instrumentor( *args: Any, **kwargs: Any ) -> None:
            # 1. Structured logs so MetricPlugins lines land as queryable payload.
            _install_json_logging()
            
            try:
                # 2. Cloud Trace instrumentation
                from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
                GoogleGenAiSdkInstrumentor().instrument()
                logging.getLogger( "agent.telemetry" ).info(
                    f"telemetry.cloud instrumented project={self._project} location={self._location}"
                )
            except Exception:
                logging.getLogger( "agent.telemetry" ).exception(
                    f"telemetry.cloud.instrument.error (continuing without GenAI tracing)"
                )
        return _instrumentor