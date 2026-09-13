import unittest
from unittest.mock import MagicMock, patch

from job_executor import execute_job_action


class _Job:
    uid = "J99"
    status = "building"
    def queue(self, **kwargs):
        raise RuntimeError("enqueue rejected")


class _Workspace:
    def create_job(self, *args, **kwargs): return _Job()


class _Client:
    def find_workspace(self, *args): return _Workspace()


class JobExecutorTests(unittest.TestCase):
    def test_enqueue_failure_records_created_job_as_partial_side_effect(self):
        action = {"approval_required": False, "job_type": "blob_picker_gpu", "connections": {}, "resolved_parameters": {"diameter": 100}, "queue": {"will_queue": True, "lane": None, "hostname": None, "gpus": [], "cluster_vars": {}}, "action_id": "test"}
        with patch("job_executor.cryosparc_client", return_value=_Client()):
            result = execute_job_action("P1", "W1", action, dry_run=False)
        self.assertFalse(result["success"])
        self.assertTrue(result["partial_side_effect"])
        self.assertEqual(result["job_uid"], "J99")
        self.assertEqual(result["failure_stage"], "queue")
        self.assertTrue(result["enqueue_failed"])

    def test_interaction_failure_is_not_reported_as_enqueue_failure(self):
        job = MagicMock(uid="J100", status="waiting")
        job.wait_for_status.return_value = "waiting"
        job.interact.side_effect = RuntimeError("interactive endpoint failed")
        workspace = MagicMock()
        workspace.create_job.return_value = job
        client = MagicMock()
        client.find_workspace.return_value = workspace
        action = {
            "approval_required": False, "job_type": "inspect_picks_v2",
            "execution_mode": "interactive_mcp",
            "mcp_tool_name": "execute_interactive_cryosparc_job",
            "connections": {}, "resolved_parameters": {}, "queue": {}, "action_id": "test",
        }
        with patch("job_executor.cryosparc_client", return_value=client):
            result = execute_job_action("P1", "W1", action, dry_run=False)
        self.assertEqual(result["failure_stage"], "interaction")
        self.assertTrue(result["interaction_failed"])
        self.assertFalse(result["enqueue_failed"])
