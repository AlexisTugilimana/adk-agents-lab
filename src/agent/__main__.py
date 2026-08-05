"""Package CLI - the only discoverable entrypoint.

Reads `argv` and translates into `AgentConfig`, then calls the same switch for 
both local and cloud via `build_agent_bindings`. It owns policy and branches and is 
not profile conditional.

Commands:

    agent smoke [--prompt "..."]
        Run one turn in-process - a liveness check, not a pytest test. Requires ADC 
        (it makes a real model call). The config is forced native-only (no mcp), 
        and in-memory. A gated tool surfaces an `adk_request_confirmation` events and 
        parks the turn; there is no approver here (client's job), so that pauses 
        is HITL working.
    
    agent deploy
        Build the CLOUD app using Agent Engine. Requires ADC plus GOOGLE_GENAI_USE_VERTEXAI / 
        GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_MODEL_LOCATION. Must be run from the repo root 
        because `extra_packages=["src/agents"]` in deploy.py resolves relative to 
        cwd. Prints the Agent Engine ressource name to stdout.
"""
import argparse
import asyncio
import logging
import sys

from agent.config import AgentConfig, Profile

_CLOUD = Profile( "cloud" )

def _cmd_smoke( args: argparse.Namespace ) -> int:
    """Run one in-process turn (native-only, in-memory). Need ADC for the model call."""
    
    from agent.local_runtime import _default_smoke_config, run_smoke
    
    cfg = _default_smoke_config()
    asyncio.run( run_smoke( cfg, prompt = args.prompt ) )
    return 0

def _cmd_deploy( args: argparse.Namespace ) -> int:
    """Build cloud `AdkApp` and create the agent on Agent Engine, then print the 
    ressource name."""
    
    from agent.deploy import deploy
    
    cfg = AgentConfig.from_env( _CLOUD )
    resource_name = deploy( cfg )
    print( resource_name )
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "agent", description = "ADK agent backend - local smoke and cloud deploy."
    )
    sub = parser.add_subparsers( dest = "command", required = True )
    
    smoke = sub.add_parser( "smoke", help = "Run one turn in-process (offline HITL check)." )
    smoke.add_argument(
        "--prompt",
        default = "Please echo 'hello world' using your echo tool.",
        help = "Message to send for the single smoke turn."
    )
    smoke.set_defaults( func = _cmd_smoke ) # Handler
    
    deploy = sub.add_parser( "deploy", help = "Deploy the cloud app to Vertex AI Engine." )
    deploy.set_defaults( func = _cmd_deploy )
    
    return parser

def main( argv: list[ str ] | None = None ) -> int:
    logging.basicConfig( level = logging.INFO )
    args = build_parser().parse_args( argv )
    return args.func( args )

if __name__ == "__main__":
    sys.exit( main() )