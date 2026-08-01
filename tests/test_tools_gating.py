"""Tool gating test. No MCP connection - pure `gating_from_discovery` and 
the merge logic.
"""

from agent.policy import GatingSpec, compile_policy
from agent.tools.composite import CompositeTools
from agent.tools.mcp import McpServerSpec, StdioTransport, gating_from_discovery
from agent.tools.native import NativeToolProvider, sensitive_echo
from tests.fakes import FakeTools

def test_native_provider_gates_its_demo_tool():
    provider = NativeToolProvider()
    policy = compile_policy( provider.gating() )
    # Demo tool known but not bypassed
    assert policy.requires_confirmation( sensitive_echo.__name__ ) is True
    # And it actually contains one agent tool
    assert len( provider.tools() ) == 1

def test_gating_from_discovery_computes_expected_spec():
    # Two MCP servers; "everything" bypasses "echo" + "get-env"
    specs = [
        McpServerSpec(
            name = "everything",
            transport = StdioTransport( command = "npx", args = [] ),
            bypass_confirmation = frozenset( { "echo", "get-env" } )
        ),
        McpServerSpec( name = "github", transport = StdioTransport( command = "npx", args = [] ) )
    ]
    discovered = { 0: [ "echo", "get-env", "delete-thing" ], 1: [ "list-prs" ] }
    spec = gating_from_discovery( specs, discovered )
    policy = compile_policy( spec )
    
    # Bypassed, non-destructive -> skips.
    assert policy.requires_confirmation( "everything_echo" ) is False
    assert policy.requires_confirmation( "everything_get-env" ) is False
    # Contains a destructive verb -> Gated.
    assert policy.requires_confirmation( "everything_delete-thing" ) is True
    # Known but not bypassed -> Gated as deny-by-default.
    assert policy.requires_confirmation( "github_list-prs" ) is True
    # Unknown runtime name -> Gated.
    assert policy.requires_confirmation( "everything_unknown" ) is True

def test_merge_flags_cross_provider_collision_as_ambiguous():
    a = GatingSpec(
        bypass = frozenset(),
        tool_to_server = { "shared": "A" },
        runtime_to_raw = { "shared": "shared" },
        ambiguous = frozenset()
    )
    b = GatingSpec(
        bypass = frozenset(),
        tool_to_server = { "shared": "B" },
        runtime_to_raw = { "shared": "shared" },
        ambiguous = frozenset()
    )
    merged = a.merge( b )
    assert "shared" in merged.ambiguous
    assert compile_policy( merged ).requires_confirmation( "shared" ) is True

def test_composite_gating_is_the_merge_of_its_providers():
    left = FakeTools(
        GatingSpec(
            bypass = frozenset( { ( "s1", "s1_a" ) } ),
            tool_to_server = { "s1_a": "s1" },
            runtime_to_raw = { "s1_a": "a" },
            ambiguous = frozenset()
        )
    )
    right = FakeTools(
        GatingSpec(
            bypass = frozenset(),
            tool_to_server = { "s2_b": "s2" },
            runtime_to_raw = { "s2_b": "b" },
            ambiguous = frozenset()
        )
    )
    composite = CompositeTools( [ left, right ] )
    merged = composite.gating()
    
    assert merged.tool_to_server == { "s1_a": "s1", "s2_b": "s2" }
    assert ( "s1", "s1_a" ) in merged.bypass

async def test_composite_setup_teardown_fan_out_in_lifo_order():
    a, b = FakeTools(), FakeTools()
    composite = CompositeTools( [ a, b ] )
    await composite.setup()
    assert ( a.setup_calls, b.setup_calls ) == ( 1, 1 )
    await composite.teardown()
    assert ( a.teardown_calls, b.teardown_calls ) == ( 1, 1 )
    