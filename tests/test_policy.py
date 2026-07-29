"""Policy test.

Covers load-bearing decision order:

    - Destructive floor overrides bypass
    - Unkown is gated
    - Ambiguous is gated
    - Properly bypassed tool is skipped
"""
from typing import Any
from agent.policy import (
    ConfirmationPolicy,
    GatingSpec,
    compile_policy,
    is_destructive,
    name_tokens
)

def _spec( **overrides ) -> GatingSpec:
    base: dict[ str, Any ] = dict(
        bypass = frozenset(),
        tool_to_server = {},
        runtime_to_raw = {},
        ambiguous = frozenset()
    )
    base.update( overrides )
    return GatingSpec( **base )

def test_name_tokens_and_is_destructive():
    assert name_tokens( "git_delete_branch" ) == { "git", "delete", "branch" }
    assert is_destructive( "git_delete_branch" ) is True
    assert is_destructive( "echo" ) is False

def test_destructive_floor_beats_bypass():
    spec = _spec(
        bypass = frozenset( { ( "git", "git_force_push" ) } ),
        tool_to_server = { "git_force_push": "git" },
        runtime_to_raw = { "git_force_push": "force_push" }
    )
    policy = compile_policy( spec )
    assert policy.requires_confirmation( "git_force_push" ) is True
    
def test_unknown_tool_is_gated():
    policy = compile_policy( _spec() )
    assert policy.requires_confirmation( "never_seen" ) is True

def test_ambiguous_is_gated():
    spec = _spec(
        tool_to_server = { "dup_tool": "serverA" },
        runtime_to_raw = { "dup_tool": "tool" },
        ambiguous = frozenset( { "dup_tool"} )
    )
    policy = compile_policy( spec )
    assert policy.requires_confirmation( "dup_tool" ) is True

def test_bypassed_non_destructive_tool_skips():
    spec = _spec(
        bypass = frozenset( { ( "everything", "everything_echo" ) } ),
        tool_to_server = { "everything_echo": "everything" },
        runtime_to_raw = { "everything_echo": "echo" }
    )
    policy = compile_policy( spec )
    assert policy.requires_confirmation( "everything_echo" ) is False

def test_known_but_not_bypassed_is_gated_deny_by_default():
    spec = _spec(
        tool_to_server = { "everything_echo": "everything" },
        runtime_to_raw = { "everything_echo": "echo" }
    )
    policy = compile_policy( spec )
    assert policy.requires_confirmation( "everything_echo" ) is True

def test_compile_policy_returns_policy_over_same_spec():
    spec = _spec( tool_to_server = { "t": "s" }, runtime_to_raw = { "t": "s" } )
    policy = compile_policy( spec )
    assert isinstance( policy, ConfirmationPolicy )
    assert policy.spec is spec