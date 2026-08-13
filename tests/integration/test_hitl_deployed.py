"""HITL in the deployed container - the deployed integration test for the HITL.

Local and deployed are split because their failure modes and debugging path differ. 
The flows themselves are shared by `scenario.py` so the two suites cannot drift.

Each test builds its app explictly with `async with deployed_agent()`: setup on entry, 
teardown on exit, even when assertions fail.

Running this test requires `AGENT_RESOURCE_NAME` enviornment variable to be defined, 
and pointing towards a deployed agent to be tested. Further improvement should consists 
in deploying a test agent temporarily.
"""

from tests.integration import scenarios
from tests.integration.apps import deployed_agent

# ==============================================================================
# The pause
# ==============================================================================
async def test_gated_tool_parks_the_turn_and_does_not_execute():
    await scenarios.verify_turn_parks( deployed_agent() )

# ==============================================================================
# Resume
# ==============================================================================
async def test_approval_resumes_the_turn_and_the_tool_executes():
    await scenarios.verify_approval_executes( deployed_agent() )
        
async def test_rejection_leaves_the_tool_unexecuted():
    await scenarios.verify_rejection_does_not_executes( deployed_agent() )

# ==============================================================================
# Security properties of the resume
# ==============================================================================
async def test_approval_in_a_different_session_does_not_execute():
    await scenarios.verify_approval_is_bound_to_its_session( deployed_agent() )
        
async def test_unknown_confirmation_id_does_not_execute():
    await scenarios.verify_unknown_confirmation_id_does_nothing( deployed_agent() )