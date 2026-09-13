import unittest
from unittest.mock import patch

from action_registry import execute_model_decision_payload


class MissingParameterExecutionFlowTests(unittest.TestCase):
    def test_heuristic_parameter_validates_then_dispatches(self):
        candidate = {
            "action_id": "registry_blob_picker_gpu",
            "action_type": "forward",
            "workflow_node_id": "J10:blob_picker_gpu",
            "job_type": "blob_picker_gpu",
            "execution_mode": "create_job",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "default_parameters": {},
            "parameter_template": {
                "diameter": {"required": True, "type": "number", "minimum": 0}
            },
        }
        payload = {
            "schema_version": "1.0",
            "decision_type": "forward",
            "reason": "estimated particle diameter from domain knowledge",
            "confidence": 0.9,
            "risk_flags": [],
            "evidence": ["assumed 150 A heuristic; this is not an observed fact"],
            "selected_actions": [{
                "action_id": candidate["action_id"],
                "action_type": "forward",
                "workflow_node_id": candidate["workflow_node_id"],
                "job_type": candidate["job_type"],
                "parameters": {"diameter": 150},
                "connections": {},
            }],
        }
        dispatched = {"success": True, "execution_mode": "mock_dispatch", "issues": []}
        with patch("action_registry.execute_job_action", return_value=dispatched) as dispatch:
            result = execute_model_decision_payload(
                payload,
                candidate_actions=[candidate],
                dry_run=False,
                project_uid="P9",
                workspace_uid="W3",
            )

        self.assertTrue(result["validation"]["success"])
        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "live_execution")
        dispatch.assert_called_once()
        self.assertEqual(dispatch.call_args.kwargs["planned_action"]["resolved_parameters"]["diameter"], 150)


if __name__ == "__main__":
    unittest.main()
