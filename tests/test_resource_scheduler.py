import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from job_specs import get_job_spec
from resource_scheduler import (
    LogicalJobRecord,
    apply_resource_overrides,
    build_scheduling_plan,
    choose_replacement,
    load_logical_job_record,
    make_logical_job_record,
    parse_sinfo_output,
    parse_squeue_output,
    policy_for_job,
    probe_cluster_resources,
    reconcile_race_jobs,
    register_submission,
    save_logical_job_record,
    scientific_fingerprint,
)


def snapshot(g4090_free: int, h20_free: int, cpu_free: int = 0):
    return {
        "gpu_lanes": [
            {
                "partition": "g8m192_4090_slurm",
                "node": "gpu4090-a",
                "gpu_type": "4090",
                "total_gpus": 8,
                "free_gpus": g4090_free,
                "allocated_gpus": 8 - g4090_free,
                "free_cpus": 64,
                "node_state": "idle" if g4090_free else "alloc",
                "queue_state": "ready",
            },
            {
                "partition": "h20_slurm",
                "node": "h20-a",
                "gpu_type": "H20",
                "total_gpus": 8,
                "free_gpus": h20_free,
                "allocated_gpus": 8 - h20_free,
                "free_cpus": 64,
                "node_state": "idle" if h20_free else "alloc",
                "queue_state": "ready",
            },
        ],
        "cpu": {"free_cpus": cpu_free, "queue_state": "ready"},
        "queue": {"pending": 0},
    }


def planned_action(params=None):
    return {
        "action_id": "create_extract",
        "job_type": "extract_micrographs_multi",
        "connections": {"micrographs": ("J2", "micrographs"), "particles": ("J4", "particles")},
        "resolved_parameters": params or {"compute_num_gpus": 4, "box_size_pix": 440},
    }


class ResourceSchedulerTests(unittest.TestCase):
    def test_all_noninteractive_gpu_jobs_enable_race_policy(self):
        policy = policy_for_job(
            "patch_ctf_estimation_multi",
            get_job_spec("patch_ctf_estimation_multi"),
            {"compute_num_gpus": 1},
        )

        self.assertTrue(policy.race_mode)

    def test_non_gpu_import_does_not_enable_race_policy(self):
        policy = policy_for_job(
            "import_micrographs",
            get_job_spec("import_micrographs"),
            {},
        )

        self.assertFalse(policy.race_mode)

    def test_idle_gpu_selects_preferred_lane(self):
        plan = build_scheduling_plan(
            "patch_motion_correction_multi",
            get_job_spec("patch_motion_correction_multi"),
            {"compute_num_gpus": 4},
            resource_snapshot=snapshot(g4090_free=4, h20_free=8),
        )

        self.assertTrue(plan["policy"]["race_mode"])
        self.assertEqual(plan["race_lanes"], ["g8m192_4090_slurm", "h20_slurm"])
        self.assertEqual(plan["selected_lane"], "g8m192_4090_slurm")
        self.assertEqual(plan["selected_resource"]["free_gpus"], 4)

    def test_cpu_import_uses_unqualified_queue(self):
        plan = build_scheduling_plan(
            "import_micrographs",
            get_job_spec("import_micrographs"),
            {"blob_paths": "/data/*.mrc"},
            resource_snapshot=snapshot(g4090_free=4, h20_free=8),
        )

        self.assertIsNone(plan["selected_lane"])
        self.assertIsNone(plan["queue"]["lane"])
        self.assertEqual(plan["queue"]["gpus"], [])
        self.assertEqual(plan["resource_config"]["mode"], "cpu")
        self.assertEqual(plan["reason"], "job_does_not_require_gpu")

    def test_busy_4090_falls_back_to_idle_h20(self):
        plan = build_scheduling_plan(
            "patch_motion_correction_multi",
            get_job_spec("patch_motion_correction_multi"),
            {"compute_num_gpus": 4},
            resource_snapshot=snapshot(g4090_free=0, h20_free=4),
        )

        self.assertEqual(plan["selected_lane"], "h20_slurm")
        self.assertEqual(plan["selected_resource"]["gpu_type"], "H20")
        self.assertIsNone(plan["queue"]["hostname"])
        self.assertEqual(plan["resource_config"]["hostname"], "h20-a")

    def test_waiting_four_gpu_job_downscales_to_two_gpu_replacement(self):
        record = LogicalJobRecord("logical-a", "source", "science")
        replacement = choose_replacement(
            record,
            "patch_motion_correction_multi",
            get_job_spec("patch_motion_correction_multi"),
            {"compute_num_gpus": 4},
            {"mode": "gpu", "lane": "g8m192_4090_slurm", "compute_num_gpus": 4},
            status="queued",
            waited_seconds=900,
            resource_snapshot=snapshot(g4090_free=0, h20_free=0),
        )

        self.assertEqual(replacement["strategy"], "dynamic_downscale")
        self.assertEqual(replacement["parameter_overrides"], {"compute_num_gpus": 2})
        self.assertTrue(replacement["cancel_original"])

    def test_extraction_waiting_can_fallback_to_cpu(self):
        record = LogicalJobRecord("logical-a", "source", "science")
        replacement = choose_replacement(
            record,
            "extract_micrographs_multi",
            get_job_spec("extract_micrographs_multi"),
            {"compute_num_gpus": 1, "box_size_pix": 440},
            {"mode": "gpu", "lane": "g8m192_4090_slurm", "compute_num_gpus": 1},
            status="launched",
            waited_seconds=900,
            resource_snapshot=snapshot(g4090_free=0, h20_free=0, cpu_free=64),
        )

        self.assertEqual(replacement["strategy"], "cpu_fallback")
        self.assertEqual(replacement["replacement_job_type"], "extract_micrographs_cpu_parallel")
        self.assertEqual(replacement["parameter_overrides"], {"compute_num_cores": 32})

    def test_race_h20_running_cancels_queued_4090(self):
        action = reconcile_race_jobs(
            "J4090",
            "JH20",
            {"J4090": "queued", "JH20": "running"},
        )

        self.assertEqual(action["action"], "cancel")
        self.assertEqual(action["cancel_job_id"], "J4090")
        self.assertEqual(action["winner"], "JH20")

    def test_race_two_running_jobs_are_not_killed(self):
        action = reconcile_race_jobs(
            "J4090",
            "JH20",
            {"J4090": "running", "JH20": "running"},
        )

        self.assertEqual(action["action"], "none")
        self.assertEqual(action["reason"], "both_jobs_running_do_not_cancel")

    def test_race_failed_job_keeps_other_lane(self):
        action = reconcile_race_jobs(
            "J4090",
            "JH20",
            {"J4090": "failed", "JH20": "queued"},
        )

        self.assertEqual(action["action"], "keep")
        self.assertEqual(action["keep_job_id"], "JH20")

    def test_monitor_repeat_does_not_duplicate_same_resource_config(self):
        record = LogicalJobRecord("logical-a", "source", "science")
        config = {"mode": "gpu", "lane": "h20_slurm", "compute_num_gpus": 2}

        register_submission(record, "J101", config, snapshot(0, 4), "first")
        register_submission(record, "J102", config, snapshot(0, 4), "repeat")

        self.assertEqual(record.physical_job_ids, ["J101"])
        self.assertEqual(len(record.submissions), 1)

    def test_logical_job_record_can_be_recovered_after_restart(self):
        record = LogicalJobRecord("logical-a", "source", "science")
        register_submission(record, "J101", {"lane": "h20_slurm"}, snapshot(0, 4), "first")

        with tempfile.NamedTemporaryFile() as handle:
            save_logical_job_record(record, handle.name)
            recovered = load_logical_job_record(handle.name)

        self.assertEqual(recovered.logical_job_id, "logical-a")
        self.assertEqual(recovered.physical_job_ids, ["J101"])
        self.assertEqual(recovered.submissions[0]["job_uid"], "J101")

    def test_resource_scheduling_does_not_change_scientific_decision(self):
        original = planned_action()
        before = deepcopy(original)
        record = make_logical_job_record(original)
        effective_params = apply_resource_overrides(
            original["resolved_parameters"],
            {"compute_num_gpus": 2},
        )

        self.assertEqual(original, before)
        self.assertEqual(effective_params["box_size_pix"], 440)
        self.assertNotEqual(effective_params["compute_num_gpus"], original["resolved_parameters"]["compute_num_gpus"])
        self.assertEqual(record.scientific_fingerprint, scientific_fingerprint(original))

    def test_slurm_probe_parses_gpu_cpu_and_queue_state(self):
        sinfo = "\n".join([
            "g8m192|4090a|mix|gpu:nvidia_geforce_rtx_4090_d:8(S:0-1)|96|1031672",
            "g8m768|H20a|idle|gpu:nvidia_h20:8(S:0-1)|224|2063814",
            "c512m1536*|cpu1|idle|(null)|512|1547616",
        ])
        squeue = "\n".join([
            "1305333|g8m192|cryosparc_P2_J211|PENDING|0:00|1|(Resources)",
            "1308703|g8m768|cryosparc_P2_J214|RUNNING|0:01|1|H20a",
        ])

        lanes, cpu = parse_sinfo_output(sinfo)
        queue = parse_squeue_output(squeue)

        self.assertEqual(lanes[0]["partition"], "g8m192_4090_slurm")
        self.assertEqual(lanes[1]["partition"], "h20_slurm")
        self.assertEqual(lanes[1]["free_gpus"], 8)
        self.assertEqual(cpu["free_cpus"], 512)
        self.assertEqual(queue["state_counts"]["PENDING"], 1)
        self.assertEqual(queue["partition_counts"]["g8m768"], 1)

    def test_cluster_probe_uses_slurm_when_no_json_override(self):
        outputs = {
            ("sinfo", "-N", "-h", "-o", "%P|%N|%t|%G|%c|%m"):
                "g8m768|H20a|idle|gpu:nvidia_h20:8(S:0-1)|224|2063814\n",
            ("squeue", "-h", "-o", "%i|%P|%j|%T|%M|%D|%R"): "",
        }

        def fake_run(command):
            return outputs[tuple(command)]

        with patch.dict("os.environ", {}, clear=True), patch(
            "resource_scheduler.run_probe_command",
            side_effect=fake_run,
        ):
            result = probe_cluster_resources()

        self.assertEqual(result["probe_source"], "slurm")
        self.assertEqual(result["gpu_lanes"][0]["partition"], "h20_slurm")

    def test_json_snapshot_remains_explicit_override(self):
        with patch.dict(
            "os.environ",
            {"CRYOAGENT_RESOURCE_SNAPSHOT_JSON": '{"gpu_lanes":[]}'},
            clear=True,
        ), patch("resource_scheduler.run_probe_command") as probe:
            result = probe_cluster_resources()

        probe.assert_not_called()
        self.assertEqual(result["probe_source"], "CRYOAGENT_RESOURCE_SNAPSHOT_JSON")


if __name__ == "__main__":
    unittest.main()
