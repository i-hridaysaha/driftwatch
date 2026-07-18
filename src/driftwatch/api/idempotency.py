import hashlib
import json
from typing import Any


def compute_payload_hash(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
