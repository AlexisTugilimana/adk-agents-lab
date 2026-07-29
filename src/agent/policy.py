"""Human-in-the-loop core.

It answers to a single question: "must human approve this tool before it runs? 
The module is composed of three pieces:

    * GatingSpec:           Information about the available tools. Which server 
                            each  tool belongs, which are pre-approved, which 
                            are flagged unsafe. Whatever supplies the tools 
                            produces one GatingSpec.
    * ConfirmationPolicy :  The decision built from GatingSpec. Its one method 
                            `requires_confirmation( tool_name )` applies 4 rules 
                            in order:
                            1. Destructive verb in the name -> always gate.
                            2. Ambiguous (two sources claimed the same name) -> 
                            gate.
                            3. Never seen in startup discovery -> gate.
                            4. Otherwise, gate unless explicitly pre-approved.
    * compile_policy:       Turns a GatingSpec into a ConfirmationPolicy.
    
When the agent has several tool sources at once, each emits a GatingSpec and 
they are combined with `GatingSpec.merge/merge_all`. Merging is safe by design: 
if two sources are defined.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import reduce

log = logging.getLogger( "agent.policy" )

# ==============================================================================
# Destructive floor
# ==============================================================================
# Verbs that must always gate, event if a (server, tool) pair is bypass by the 
# user. This is a non-passable floor.
DESTRUCTIVE_TOKENS: frozenset[ str ] = frozenset(
    { "delete", "remove", "destroy", "drop", "rm", "force", "push", "merge", "reset", "revert" }
)

def name_tokens( tool_name: str ) -> set[ str ]:
    return { token for token in re.split( r"[^a-z0-9]+", tool_name.lower() ) if token }

def is_destructive( tool_name: str ) -> bool:
    """True if the tool name contains destructive verbs."""
    return bool( name_tokens( tool_name ) & DESTRUCTIVE_TOKENS )

# ==============================================================================
# GatingSpec - Data every tool source emits
# ==============================================================================
@dataclass( frozen = True )
class GatingSpec:
    """Source-agnostic description of what must be confirmed."""
    
    # (server, runtime_tool) pairs explicitely allowed to skip the gate.
    bypass: frozenset[ tuple[ str, str ] ]
    # runtime_tool -> owning server name.
    tool_to_server: dict[ str, str ]
    # runtime_tool -> server's own raw tool name
    runtime_to_raw: dict[ str, str ]
    # runtime tools with cross-provider/cross-server name collision
    ambiguous: frozenset[ str ]
    
    @staticmethod
    def empty() -> GatingSpec:
        return GatingSpec(
            bypass = frozenset(),
            tool_to_server = {},
            runtime_to_raw = {},
            ambiguous = frozenset()
        )
    
    def merge( self, other: GatingSpec ) -> GatingSpec:
        """Combine two specs. Any identical runtime tool claimed by both 
        providers is forced ambiguous."""
        
        # Collision between runtime_tools
        collisions = set( self.tool_to_server ) & set( other.tool_to_server )
        for tool in sorted( collisions ):
            log.warning(
                f"gating.collision tool={tool!r} "
                f"providers=({self.tool_to_server[tool]!r}, {other.tool_to_server[tool]!r}) "
                "-> forcing confirmation"
            )
        
        merged_tool_to_server = {**self.tool_to_server, **other.tool_to_server }
        merged_runtime_to_raw = {**self.runtime_to_raw, **other.runtime_to_raw }
        return GatingSpec(
            bypass = self.bypass | other.bypass,
            tool_to_server = merged_tool_to_server,
            runtime_to_raw = merged_runtime_to_raw,
            ambiguous = self.ambiguous | other.ambiguous | frozenset( collisions )
        )

# ==============================================================================
# Confirmation policy
# ==============================================================================
@dataclass( frozen = True )
class ConfirmationPolicy:
    """Whether a human must approve a tool before it runs."""
    
    spec: GatingSpec
    
    def requires_confirmation( self, tool_name: str ) -> bool:
        spec = self.spec
        
        # 1. Destructive floor
        raw = spec.runtime_to_raw.get( tool_name, tool_name )
        if is_destructive( raw ):
            return True
        
        # 2. Ambiguous
        if tool_name in spec.ambiguous:
            return True
        
        # 3. Unknown tool - not seen at discovery
        server = spec.tool_to_server.get( tool_name )
        if server is None:
            return True
        
        # 4. Deny-by-default unless bypassed
        return ( server, tool_name ) not in spec.bypass
        
def compile_policy( spec: GatingSpec ) -> ConfirmationPolicy:
    """Turned a `GatingSpec` into a `ConfirmationPolicy`."""
    
    policy = ConfirmationPolicy( spec )
    bypassed = sorted( f"{server}:{tool}" for server, tool in spec.bypass )
    gated = sorted(
        f"{server}:{tool}"
        for server, tool in spec.tool_to_server.items()
        if policy.requires_confirmation( tool )
    )
    log.info( f"confirmation.policy gated={gated}" )
    log.info( f"confirmation.policy bypassed={bypassed}" )
    return policy

def merged_all( specs: list[ GatingSpec ] ) -> GatingSpec:
    """Fold a list of `GatingSpec` into one."""
    
    return reduce( GatingSpec.merge, specs, GatingSpec.empty() )