import unittest

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
            "pipeline": {"version": "actions-echo-v1"},
        })
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["outputs"]["echo"]["ok"])


if __name__ == "__main__":
    unittest.main()
