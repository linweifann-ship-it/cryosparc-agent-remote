import unittest
from unittest.mock import MagicMock, patch

from action_registry import execute_model_decision_payload
from job_executor import execute_job_action


def interactive_candidate(job_type="inspect_picks_v2"):
    return {
        "action_id": f"registry_J8_{job_type}",
        "action_type": "forward",
        "workflow_node_id": f"J8:{job_type}",
        "job_type": job_type,
        "execution_mode": "interactive_mcp",
        "mcp_tool_name": "execute_interactive_cryosparc_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": {"particles": [{"source_job_uid": "J8", "source_output": "particles"}]},
        "parameter_template": {},
        "default_parameters": {},
        "job_spec_metadata": {
            "category": "inspection",
            "requires_gpu": False,
            "requires_approval": False,
            "interactive": True,
            "destructive": False,
            "max_auto_gpus": 4,
        },
    }


def decision_for(candidate):
    return {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [{
            "action_id": candidate["action_id"],
            "action_type": "forward",
            "workflow_node_id": candidate["workflow_node_id"],
            "job_type": candidate["job_type"],
            "parameters": {},
        }],
        "rollback_target": None,
        "branch_plan": None,
        "reason": "Model selected a validated QC action.",
        "confidence": 0.9,
        "risk_flags": [],
        "evidence": [],
    }


class InteractiveExecutionPolicyTests(unittest.TestCase):
    def test_inspect_picks_is_autonomous_and_uses_dedicated_dispatch(self):
        candidate = interactive_candidate()
        result = execute_model_decision_payload(decision_for(candidate), [candidate])
        action = result["execution_plan"]["actions"][0]
        self.assertFalse(action["approval_required"])
        self.assertEqual(action["execution_mode"], "interactive_mcp")
        self.assertEqual(action["mcp_tool_name"], "execute_interactive_cryosparc_job")

    def test_select_2d_is_autonomous_and_uses_dedicated_dispatch(self):
        candidate = interactive_candidate("select_2D")
        result = execute_model_decision_payload(decision_for(candidate), [candidate])
        action = result["execution_plan"]["actions"][0]
        self.assertFalse(action["approval_required"])
        self.assertEqual(action["execution_mode"], "interactive_mcp")

    def test_invalid_interactive_tool_is_rejected_before_execution(self):
        candidate = interactive_candidate()
        candidate["mcp_tool_name"] = "missing_tool"
        result = execute_model_decision_payload(decision_for(candidate), [candidate])
        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "execution_tool_unavailable")

    @patch("job_executor.refresh_scheduling_plan")
    @patch("job_executor.cryosparc_client")
    def test_interactive_action_dispatches_without_queue(self, mock_client, mock_schedule):
        mock_schedule.return_value = {
            "queue": {"will_queue": False, "lane": None, "hostname": None, "gpus": [], "cluster_vars": {}},
            "parameter_overrides": {}, "resource_config": {}, "snapshot": {}, "reason": "test",
        }
        job = MagicMock(uid="J9", status="building")
        workspace = MagicMock()
        workspace.create_job.return_value = job
        mock_client.return_value.find_workspace.return_value = workspace
        planned = {
            "approval_required": False, "execution_mode": "interactive_mcp",
            "mcp_tool_name": "execute_interactive_cryosparc_job", "job_type": "inspect_picks_v2",
            "connections": {}, "resolved_parameters": {}, "action_id": "registry_J8_inspect_picks_v2",
            "queue": {}, "resource_scheduling": {},
        }
        with patch("job_executor.register_submission"):
            result = execute_job_action("P9", "W2", planned, dry_run=False)
        self.assertTrue(result["success"])
        self.assertEqual(result["job_uid"], "J9")
        job.queue.assert_not_called()

    def test_non_interactive_action_remains_auto_executable(self):
        candidate = interactive_candidate("blob_picker_gpu")
        candidate.update({"execution_mode": "create_job", "mcp_tool_name": None})
        candidate["job_spec_metadata"].update({"interactive": False, "requires_gpu": True})
        result = execute_model_decision_payload(decision_for(candidate), [candidate])
        self.assertTrue(result["success"])
        self.assertFalse(result["execution_plan"]["actions"][0]["approval_required"])

    def test_destructive_action_still_requires_approval(self):
        candidate = interactive_candidate("destructive_test")
        candidate.update({"execution_mode": "create_job", "mcp_tool_name": None})
        candidate["job_spec_metadata"].update({"interactive": False, "destructive": True})
        result = execute_model_decision_payload(decision_for(candidate), [candidate])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertIn("destructive_action", result["execution_plan"]["approval_reasons"])


if __name__ == "__main__":
    unittest.main()
