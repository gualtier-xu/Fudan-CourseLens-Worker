import unittest
from unittest.mock import Mock, patch

from courselens_worker.protocol import JOB_SCHEMA, PROTOCOL_VERSION
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


if __name__ == "__main__":
    unittest.main()
