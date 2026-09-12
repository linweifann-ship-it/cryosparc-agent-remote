import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cryosparc_agent_remote"))

from cryosparc_agent_remote.job_executor import (
    build_cryosparc_payload,
    cancel_physical_race_job,
    execute_job_action,
    extract_http_response,
    kill_non_running_race_losers,
)


class JobExecutorDiagnosticsTests(unittest.TestCase):
    def test_build_cryosparc_payload_records_create_and_queue_inputs(self):
        payload = build_cryosparc_payload(
            project_uid="P2",
            workspace_uid="W13",
            planned_action={
                "job_type": "blob_picker_gpu",
                "action_id": "forward_J161_blob_picker_gpu",
                "connections": {"micrographs": ("J161", "imported_micrographs")},
                "resolved_parameters": {"diameter": 180},
                "queue": {
                    "will_queue": True,
                    "lane": "g8m192_4090_slurm",
                    "hostname": None,
                    "gpus": [],
                    "cluster_vars": {},
                },
            },
        )
        self.assertEqual(payload["project_uid"], "P2")
        self.assertEqual(payload["workspace_uid"], "W13")
        self.assertEqual(payload["create_job"]["job_type"], "blob_picker_gpu")
        self.assertEqual(payload["create_job"]["params"], {"diameter": 180})
        self.assertEqual(payload["queue"]["lane"], "g8m192_4090_slurm")

    def test_extract_http_response_reads_status_url_and_body(self):
        request = SimpleNamespace(method="POST", url="http://localhost/jobs/J171:enqueue")
        response = SimpleNamespace(
            status_code=422,
            reason_phrase="Unprocessable Entity",
            text='{"detail":"bad lane"}',
            request=request,
        )
        exc = SimpleNamespace(response=response)
        result = extract_http_response(exc)
        self.assertEqual(result["status_code"], 422)
        self.assertEqual(result["method"], "POST")
        self.assertEqual(result["body"], '{"detail":"bad lane"}')
        self.assertEqual(result["json"], {"detail": "bad lane"})

    def test_gpu_race_submission_creates_two_physical_jobs_for_one_logical_step(self):
        planned_action = {
            "plan_step": 1,
            "action_id": "registry_J216_patch_ctf_estimation_multi",
            "action_type": "forward",
            "workflow_node_id": "J216:patch_ctf_estimation_multi",
            "job_type": "patch_ctf_estimation_multi",
            "job_category": "ctf",
            "execution_mode": "create_job",
            "connections": {"exposures": ("J216", "imported_micrographs")},
            "resolved_parameters": {"compute_num_gpus": 1},
            "queue": {
                "will_queue": True,
                "lane": "g8m192_4090_slurm",
                "hostname": None,
                "gpus": [],
                "cluster_vars": {},
            },
            "resource_scheduling": {
                "policy": {"race_mode": True},
                "race_lanes": ["g8m192_4090_slurm", "h20_slurm"],
                "parameter_overrides": {},
                "queue": {
                    "will_queue": True,
                    "lane": "g8m192_4090_slurm",
                    "hostname": None,
                    "gpus": [],
                    "cluster_vars": {},
                },
            },
            "requires_gpu": True,
            "interactive": False,
            "approval_required": False,
            "approval_reasons": [],
        }
        queued_lanes = []

        class FakeJob:
            def __init__(self, uid):
                self.uid = uid
                self.status = "building"
                self.killed = False

            def queue(self, lane=None, hostname=None, gpus=None, cluster_vars=None):
                queued_lanes.append(lane)

            def kill(self):
                self.killed = True

        class FakeWorkspace:
            def __init__(self):
                self.next_id = 217
                self.jobs = []

            def create_job(self, *args, **kwargs):
                job = FakeJob(f"J{self.next_id}")
                self.next_id += 1
                self.jobs.append(job)
                return job

            def find_jobs(self):
                return self.jobs

        fake_workspace = FakeWorkspace()
        fake_client = SimpleNamespace(
            find_workspace=lambda project_uid, workspace_uid: fake_workspace
        )

        def fake_refresh(action):
            lane = action["queue"]["lane"]
            return {
                "policy": {
                    "race_mode": True,
                    "queue_start_timeout_seconds": 0,
                    "race_poll_interval_seconds": 0,
                },
                "snapshot": {"probe_source": "test", "lane": lane},
                "selected_lane": lane,
                "race_lanes": ["g8m192_4090_slurm", "h20_slurm"],
                "selected_resource": None,
                "resource_config": {
                    "mode": "gpu",
                    "lane": lane,
                    "hostname": None,
                    "compute_num_gpus": 1,
                },
                "queue": {**action["queue"], "lane": lane},
                "parameter_overrides": {},
                "reason": "race_physical_lane",
            }

        with patch(
            "cryosparc_agent_remote.job_executor.refresh_scheduling_plan",
            side_effect=fake_refresh,
        ), patch(
            "cryosparc_agent_remote.job_executor.cryosparc_client",
            return_value=fake_client,
        ):
            result = execute_job_action("P2", "W16", planned_action, dry_run=False)

        self.assertTrue(result["success"])
        self.assertTrue(result["logical_workflow_step"])
        self.assertEqual(result["physical_execution_redundancy"], "race")
        self.assertEqual(result["logical_job"]["physical_job_ids"], ["J217", "J218"])
        self.assertEqual(queued_lanes, ["g8m192_4090_slurm", "h20_slurm"])

    def test_race_loser_is_killed_when_peer_is_running(self):
        class FakeJob:
            def __init__(self, uid, status):
                self.uid = uid
                self.status = status
                self.killed = False

            def kill(self):
                self.killed = True

        loser = FakeJob("J225", "launched")
        winner = FakeJob("J226", "running")

        result = kill_non_running_race_losers(
            {"J225": loser, "J226": winner},
            "J226",
            {"J225": "launched", "J226": "running"},
        )

        self.assertTrue(loser.killed)
        self.assertFalse(winner.killed)
        self.assertEqual(result, [{
            "job_uid": "J225",
            "previous_status": "launched",
            "action": "kill",
            "method": "kill",
            "success": True,
        }])

    def test_race_loser_is_killed_when_both_physical_jobs_are_running(self):
        class FakeJob:
            def __init__(self, uid):
                self.uid = uid
                self.status = "running"
                self.killed = False

            def kill(self):
                self.killed = True

        loser, winner = FakeJob("J225"), FakeJob("J226")
        result = kill_non_running_race_losers(
            {"J225": loser, "J226": winner}, "J226",
            {"J225": "running", "J226": "running"},
        )
        self.assertTrue(loser.killed)
        self.assertFalse(winner.killed)
        self.assertTrue(result[0]["success"])

    def test_race_loser_cancel_fallback_when_kill_fails(self):
        class FakeJob:
            def __init__(self):
                self.cancelled = False

            def kill(self):
                raise RuntimeError("cannot kill from this state")

            def cancel(self):
                self.cancelled = True

        job = FakeJob()
        result = cancel_physical_race_job(job, "J225", "queued")

        self.assertTrue(job.cancelled)
        self.assertEqual(result["job_uid"], "J225")
        self.assertEqual(result["method"], "cancel")
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
