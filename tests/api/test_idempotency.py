from driftwatch.api.idempotency import compute_payload_hash


def test_identical_payloads_hash_the_same() -> None:
    a = compute_payload_hash({"features": {"age": 30, "region": "EU"}, "prediction_score": 0.8})
    b = compute_payload_hash({"prediction_score": 0.8, "features": {"region": "EU", "age": 30}})

    assert a == b


def test_different_payloads_hash_differently() -> None:
    a = compute_payload_hash({"features": {"age": 30}, "prediction_score": 0.8})
    b = compute_payload_hash({"features": {"age": 31}, "prediction_score": 0.8})

    assert a != b
