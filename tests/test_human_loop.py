import asyncio
import pytest

from peldrun.tools.builtins.human_input import HumanInputTool, HumanInputRegistry


@pytest.mark.asyncio
async def test_human_input_resolution_cycle():
    tool = HumanInputTool()

    async def simulate_human_operator(req_id: str):
        await asyncio.sleep(0.05)
        HumanInputRegistry.resolve_request(req_id, "Permission granted by operator.")

    # Start tool execution task
    task = asyncio.create_task(tool.execute(query="Do you approve deleting temporary files?"))

    # Wait brief moment to capture request_id
    await asyncio.sleep(0.01)
    assert len(HumanInputRegistry._pending_requests) == 1
    active_req_id = list(HumanInputRegistry._pending_requests.keys())[0]

    # Operator response
    await simulate_human_operator(active_req_id)

    result = await task
    assert result.exit_code == 0
    assert "Permission granted" in result.output
    assert result.metadata.get("approved") is True