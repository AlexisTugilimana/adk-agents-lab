"""Ship the App to Agent Engine.

Run locally: builds the cloud-profile `AdkApp` and assembles, then read the 
`deployment/requirements.txt`, packages the `src/agent`, and calls 
`agent_engine.create(...)`.

Must be run from the repo root as `extra_packages=["src/agent"]` resolves relative 
to the cwd. 

Requires ADC plus GOOGLE_CLOUD_PROJECT/GOOGLE_PROJECT_LOCATION/GOOGLE_CLOUD_STAGING_BUCKET. 
Returns the Agent Engine resource name.
"""
import logging
from pathlib import Path

import vertexai
from vertexai import agent_engines
from vertexai.agent_engines import AdkApp

from agent.assemble import assemble_adk_app
from agent.bindings import build_agent_bindings
from agent.config import AgentConfig

log = logging.getLogger( "agent.deploy" )

_DEPLOYMENT_DIR = Path( __file__ ).resolve().parents[ 2 ] / "deployment"
_REQUIREMENTS = _DEPLOYMENT_DIR / "requirements.txt"

def _read_requirements() -> list[ str ]:
    """Read the base-only requirements lines that define what deploys."""
    lines = _REQUIREMENTS.read_text( encoding = "utf-8" ).splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith( "#" )
    ]

def deploy( cfg: AgentConfig ) -> str:
    """Deploy the cloud-profile `AdkApp` and returns its Agent Engine ressource name."""
    
    # Fail fast with actionable message
    missing = [
        name 
        for name, value in (
            ( "GOOGLE_CLOUD_PROJECT", cfg.project ),
            ( "GOOGLE_CLOUD_ENGINE_LOCATION", cfg.engine_location ),
            ( "GOOGLE_CLOUD_STAGING_BUCKET", cfg.staging_bucket ),
        )
        if not value
    ]
    if missing:
        raise SystemExit( f"deploy: missing required environment: {', '.join( missing )}" )
    
    vertexai.init(
        project = cfg.project,
        location = cfg.engine_location,
        staging_bucket = cfg.staging_bucket
    )
    log.info( f"deploy.init project={cfg.project} location={cfg.engine_location} bucket={cfg.staging_bucket}" )
    
    bindings = build_agent_bindings( cfg )
    adk_app = assemble_adk_app( cfg, bindings, adk_app_factory = AdkApp )
    
    requirements = _read_requirements()
    log.info( f"deploy.requirements {requirements}" )
    
    env_vars: dict[ str, str ] = { "GOOGLE_GENAI_USE_VERTEXAI": "1" }
    if cfg.location:
        env_vars[ "GOOGLE_CLOUD_LOCATION" ] = cfg.location
    env_vars[ "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY" ] = str(
        bindings.telemetry.enabled_tracing
    ).lower()
    
    engine = agent_engines.create(
        agent_engine = adk_app,
        requirements = requirements,
        extra_packages = [ "src/agent" ],
        env_vars = dict( env_vars ),
        display_name = cfg.app_name
    )
    
    resource_name = getattr( engine, "resource_name", str( engine ) )
    log.info( f"deploy.created resource_name={resource_name}" )
    return resource_name