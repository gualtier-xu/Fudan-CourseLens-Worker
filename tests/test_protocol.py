from __future__ import annotations

import base64
import hashlib
import json
import time
import unittest

import zstandard
from nacl.public import PrivateKey, SealedBox
from nacl.signing import SigningKey

from courselens_worker.protocol import (
    ENVELOPE_SCHEMA,
    JOB_SCHEMA,
    PROTOCOL_VERSION,
    RESULT_SCHEMA,
    open_job,
    seal_result,
)


class ProtocolTests(unittest.TestCase):
    def test_worker_opens_job_and_seals_result(self):
        worker = PrivateKey.generate()
        local = PrivateKey.generate()
        signing = SigningKey.generate()
        job = {
            "schema": JOB_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef", "job_kind": "echo",
            "created_at": time.time(), "expires_at": time.time() + 300,
            "result_public_key": base64.b64encode(bytes(local.public_key)).decode(),
            "pipeline": {"version": "test"}, "payload": {}, "requested_outputs": [],
        }
        raw = json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        job["input_hash"] = hashlib.sha256(raw).hexdigest()
        compressed = zstandard.ZstdCompressor().compress(
            json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        )
        ciphertext = SealedBox(worker.public_key).encrypt(compressed)
        envelope = {
            "schema": ENVELOPE_SCHEMA, "sha256": hashlib.sha256(ciphertext).hexdigest(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
        }
        opened = open_job(envelope, base64.b64encode(bytes(worker)).decode())
        result = {
            "schema": RESULT_SCHEMA, "protocol_version": PROTOCOL_VERSION, "task_id": job["task_id"],
            "job_kind": "echo", "input_hash": job["input_hash"], "status": "completed",
        }
        sealed = seal_result(
            result,
            job["result_public_key"],
            base64.b64encode(bytes(signing)).decode(),
        )
        self.assertIn("signature", sealed)
        self.assertNotIn(job["task_id"], sealed["ciphertext"])

    def test_runner_source_session_requires_encrypted_credentials(self):
        worker = PrivateKey.generate()
        local = PrivateKey.generate()
        job = {
            "schema": JOB_SCHEMA, "protocol_version": PROTOCOL_VERSION,
            "task_id": "fedcba9876543210fedcba9876543210", "job_kind": "subtitle",
            "created_at": time.time(), "expires_at": time.time() + 300,
            "result_public_key": base64.b64encode(bytes(local.public_key)).decode(),
            "pipeline": {"version": "test"}, "requested_outputs": ["subtitle"],
            "payload": {
                "mode": "fast", "media": {"start_seconds": 600, "duration_seconds": 300},
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "36941",
                    "sub_id": "652577", "media": True, "slides": False,
                },
            },
            "secrets": {"source_credentials": {"account": "synthetic", "password": "synthetic"}},
        }
        raw = json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        job["input_hash"] = hashlib.sha256(raw).hexdigest()
        compressed = zstandard.ZstdCompressor().compress(
            json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        )
        ciphertext = SealedBox(worker.public_key).encrypt(compressed)
        envelope = {
            "schema": ENVELOPE_SCHEMA, "sha256": hashlib.sha256(ciphertext).hexdigest(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
        }
        opened = open_job(envelope, base64.b64encode(bytes(worker)).decode())
        self.assertEqual(opened["payload"]["source_session"]["sub_id"], "652577")
        self.assertNotIn("synthetic", str(envelope))


if __name__ == "__main__":
    unittest.main()
