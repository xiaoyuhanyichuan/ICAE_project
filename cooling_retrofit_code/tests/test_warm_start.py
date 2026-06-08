import numpy as np

from decision import DecisionSchema
from warm_start import WarmStartManager


def test_warm_start_manager_finds_exact_and_neighbor_capacity_bucket_matches():
    schema = DecisionSchema(["z1", "z2"])
    manager = WarmStartManager(
        enabled=True,
        capacity_bucket_kw=100.0,
        neighbor_bucket_radius=1,
        max_entries=4,
        td_fingerprint="td-a",
    )
    base = schema.encode_default()
    base[0] = 4
    base[1] = 5
    base[2:] = [
        210.0,
        90.0,
        0.0,
        130.0,
        1200.0,
        600.0,
        500.0,
        240.0,
        800.0,
        900.0,
        300.0,
        400.0,
    ]
    payload = {"p_grid": {0: 12.0}}

    manager.remember(schema.decode(base), payload)

    exact = manager.lookup(schema.decode(base))
    assert exact is not None
    assert exact.payload == payload
    assert exact.match_type == "exact"
    assert manager.stats["exact_hits"] == 1

    nearby = base.copy()
    nearby[2] = 330.0
    neighbor = manager.lookup(schema.decode(nearby))
    assert neighbor is not None
    assert neighbor.payload == payload
    assert neighbor.match_type == "neighbor"
    assert manager.stats["neighbor_hits"] == 1

    miss = base.copy()
    miss[2] = 1000.0
    assert manager.lookup(schema.decode(miss)) is None
    assert manager.stats["misses"] == 1


def test_warm_start_manager_evicts_oldest_entries_when_capacity_is_exceeded():
    schema = DecisionSchema(["z1"])
    manager = WarmStartManager(
        enabled=True,
        capacity_bucket_kw=10.0,
        neighbor_bucket_radius=0,
        max_entries=1,
        td_fingerprint="td-a",
    )
    first = schema.encode_default()
    first[0] = 1
    first[1] = 10.0
    second = schema.encode_default()
    second[0] = 1
    second[1] = 20.0

    manager.remember(schema.decode(first), {"p_grid": {0: 1.0}})
    manager.remember(schema.decode(second), {"p_grid": {0: 2.0}})

    assert manager.lookup(schema.decode(first)) is None
    hit = manager.lookup(schema.decode(second))
    assert hit is not None
    assert hit.payload["p_grid"][0] == 2.0


def test_warm_start_manager_copies_dense_payload_arrays_when_remembering():
    schema = DecisionSchema(["z1"])
    manager = WarmStartManager(
        enabled=True,
        capacity_bucket_kw=100.0,
        neighbor_bucket_radius=0,
        max_entries=4,
        td_fingerprint="td-a",
        config_fp="config-a",
    )
    vector = schema.encode_default()
    vector[0] = 1
    vector[1] = 100.0
    decision = schema.decode(vector)
    values = np.array([[1.0, 2.0]])
    payload = {
        "__format__": "dense_v1",
        "zone_ids": ["z1"],
        "time_count": 2,
        "vars": {"s_air_new": values},
    }

    manager.remember(decision, payload)
    values[0, 0] = 99.0
    payload["vars"]["s_air_new"][0, 1] = 88.0

    hit = manager.lookup(decision)

    assert hit is not None
    assert hit.key[0] == "config-a"
    assert np.asarray(hit.payload["vars"]["s_air_new"]).tolist() == [[1.0, 2.0]]
