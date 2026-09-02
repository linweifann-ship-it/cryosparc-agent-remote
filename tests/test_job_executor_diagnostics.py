import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cryosparc_agent_remote"))

from cryosparc_agent_remote.job_executor import (
    build_cryosparc_payload,
    extract_http_response,
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


if __name__ == "__main__":
    unittest.main()
