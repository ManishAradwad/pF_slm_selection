"""Invented supplied-vector checks; not dataset-backed quality evaluation."""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from pocketfinancer_sms import prefilter as policy

FIXTURE = Path(__file__).parent / "golden" / "prefilter-v1.json"
FIXTURE_SHA256 = "373337f584cd70d4d165e7ff557fb45ece8967f3795d0d2d05a02f61db5053c9"
DOCUMENT = json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_shared_fixture_identity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert DOCUMENT["schema_version"] == 1
    assert DOCUMENT["policy_id"] == policy.POLICY_ID
    assert DOCUMENT["provenance"].startswith("Invented for the prefilter pilot;")
    ids = [vector["id"] for vector in DOCUMENT["vectors"]]
    assert len(ids) == len(set(ids)) == 38


@pytest.mark.parametrize("vector", DOCUMENT["vectors"], ids=lambda vector: vector["id"])
def test_shared_vector(vector):
    body = vector["body"]
    original_bytes = body.encode("utf-8")
    expected = policy.PrefilterResult(vector["decision"], vector["reason"])
    assert policy.prefilter(body) == expected
    assert policy.prefilter(body) == expected  # no operation state
    assert body.encode("utf-8") == original_bytes
    # Sender metadata is deliberately not supplied to the pure body-only policy.
    assert "sender" in vector


def test_pure_policy_has_no_inference_or_storage_dependency():
    tree = ast.parse(inspect.getsource(policy))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {"dataclasses", "re", "typing"}
