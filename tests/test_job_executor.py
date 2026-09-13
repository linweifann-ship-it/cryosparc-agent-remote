import unittest
from unittest.mock import patch

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
