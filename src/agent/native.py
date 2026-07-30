"""Native tool provider + gated demo tool.

Layer: adapter. It ships to AgentEngine.

The `NativeToolProvider` is specific to one tool implementation, provided as 
a python function wrapped into a `FunctionTool`, to make it ADK ready-to-use.

The confirmation gate is compiled by the `NativeToolProvider` to be 
tool-agnostic in its output.

The native gated demo is here only for demonstration purpose, to test the HITL 
on the cloud when deployed on Agent Engine.
"""
from google.adk.agents.llm_agent import ToolUnion
from google.adk.tools import FunctionTool

from agent.policy import GatingSpec
from agent.ports import ToolProvisioning

# Server label for native tools (used in GatingSpec)
NATIVE_SERVER = "native"

def sensitive_echo( message: str ) -> dict:
    """Echo the message back to the caller. Requires human approval before it 
    runs.
    
    Deliberately trivial: it's job is to exercice the HITL pause without depending 
    on external system. The name does not contain any destructive words and is 
    therefore gated by the deny-by-default approach.
    """
    
    return { "echoed": message }

def make_gated_demo_tool() -> ToolUnion:
    """Construct the native demo gated tool."""
    
    return FunctionTool( func = sensitive_echo )

class NativeToolProvider( ToolProvisioning ):
    """In-process tool provider. No lifecycle here."""
    
    def __init__( self ) -> None:
        self._tools = [ make_gated_demo_tool() ]
    
    async def setup(self) -> None:
        return None
    
    async def teardown(self) -> None:
        return None
    
    def tools( self ) -> list[ ToolUnion ]:
        return list( self._tools )
    
    def gating( self ) -> GatingSpec:
        """Static declaration: the demo tool is known but not bypassed and does 
        not contain destructive words."""
        
        return GatingSpec(
            bypass = frozenset(),
            tool_to_server = { sensitive_echo.__name__: NATIVE_SERVER },
            runtime_to_raw = { sensitive_echo.__name__: sensitive_echo.__name__ },
            ambiguous = frozenset()
        )