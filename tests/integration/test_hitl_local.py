"""HITL in-process - the local integration test for the HITL.

Local and deployed are split because their failure modes and debugging path differ. 
The flows themselves are shared by `scenario.py` so the two suites cannot drift.

Each test builds its app explictly with `async with local_agent()`: setup on entry, 
teardown on exit, even when assertions fail.
"""

from tests.integration import scenarios
from tests.integration.apps import local_agent

# ==============================================================================
# The pause
# ==============================================================================
async def test_gated_tool_parks_the_turn_and_does_not_execute():
    async with local_agent() as app:
        await scenarios.verify_turn_parks( app )

# ==============================================================================
# Resume
# ==============================================================================
async def test_approval_resumes_the_turn_and_the_tool_executes():
    async with local_agent() as app:
        await scenarios.verify_approval_executes( app )
        
async def test_rejection_leaves_the_tool_unexecuted():
    async with local_agent() as app:
        await scenarios.verify_rejection_does_not_executes( app )

# ==============================================================================
# Security properties of the resume
# ==============================================================================
async def test_approval_in_a_different_session_does_not_execute():
    async with local_agent() as app:
        await scenarios.verify_approval_is_bound_to_its_session( app )
        
async def test_unknown_confirmation_id_does_not_execute():
    async with local_agent() as app:
        await scenarios.verify_unknown_confirmation_id_does_nothing( app )