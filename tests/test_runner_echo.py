import unittest
from unittest.mock import Mock, patch

from courselens_worker.protocol import (
    JOB_SCHEMA,
    PROCESS_CANARY_FIXTURE_BYTES,
    PROCESS_CANARY_FIXTURE_RECORDS,
    PROCESS_CANARY_FIXTURE_SHA256,
    PROCESS_CANARY_PIPELINE,
    PROCESS_CANARY_SCHEMA,
    PROTOCOL_VERSION,
)
from courselens_worker.runner import process_job


class RunnerEchoTests(unittest.TestCase):
    def test_echo_does_not_require_compute_dependencies(self):
        result = process_job({
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "echo",
            "input_hash": "0" * 64,
            "pipeline": {"version": "actions-echo-v2"},
        })
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["outputs"]["echo"]["ok"])

    def test_materialized_source_session_closes_after_processing_failure(self):
        close = Mock()
        job = {
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "subtitle",
            "input_hash": "0" * 64,
            "pipeline": {"version": "actions-v2"},
            "payload": {"source_session": {"provider": "runner-session-v1"}},
        }
        materialized = {
            **job,
            "payload": {"_close_source_session": close},
        }
        with (
            patch(
                "courselens_worker.platform_session.materialize_job_sources",
                return_value=materialized,
            ),
            patch(
                "courselens_worker.runner._process_materialized_job",
                side_effect=RuntimeError("failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                process_job(job)
        close.assert_called_once_with()

    def test_process_canary_uses_only_runner_owned_fixture_and_process_profile(self):
        job = {
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "process_canary",
            "input_hash": "0" * 64,
            "pipeline": {"version": PROCESS_CANARY_PIPELINE},
            "payload": {},
        }
        with patch.dict("os.environ", {
            "GITHUB_SHA": "a" * 40,
            "COURSELENS_WORKFLOW_PROFILE": "process-v1",
        }, clear=False):
            result = process_job(job)
        self.assertEqual(result["outputs"], {"process_canary": {
            "schema": PROCESS_CANARY_SCHEMA,
            "fixture_sha256": PROCESS_CANARY_FIXTURE_SHA256,
            "fixture_bytes": PROCESS_CANARY_FIXTURE_BYTES,
            "fixture_records": PROCESS_CANARY_FIXTURE_RECORDS,
            "worker_commit": "a" * 40,
            "workflow_profile": "process-v1",
        }})
        self.assertEqual(result["metrics"], {
            "synthetic_bytes": PROCESS_CANARY_FIXTURE_BYTES,
            "synthetic_records": PROCESS_CANARY_FIXTURE_RECORDS,
        })

    def test_process_canary_rejects_wrong_workflow_profile(self):
        job = {
            "schema": JOB_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "process_canary", "input_hash": "0" * 64,
            "pipeline": {"version": PROCESS_CANARY_PIPELINE}, "payload": {},
        }
        with patch.dict("os.environ", {
            "GITHUB_SHA": "a" * 40,
            "COURSELENS_WORKFLOW_PROFILE": "echo-v1",
        }, clear=False):
            with self.assertRaisesRegex(Exception, "workflow profile"):
                process_job(job)


if __name__ == "__main__":
    unittest.main()
