"""The test-only brute force of acceptance test 8.3 on hand-checkable instances."""

from __future__ import annotations

from tests.brute_force import Instance, brute_force_optimum, random_instance
from tests.fake_planner import fake_row

ONE_GPU = fake_row("one", "bf-one", 1, 1.0)  # $24/day
TWO_GPU = fake_row("two", "bf-two", 2, 1.5)  # $36/day


def test_one_class_needs_whole_replicas() -> None:
    # 3 req/s per replica against 7 req/s: 3 instances, $72/day.
    instance = Instance((ONE_GPU,), {("bf-one", 1): ((3.0, 100.0),)}, ((7.0, 0.0),))
    assert brute_force_optimum(instance) == 72.0


def test_two_classes_share_a_replica() -> None:
    # One replica serves class 0 (needs 0.5 of it) and class 1 (0.4): one instance. The
    # two-GPU row (two tp1 replicas or one tp2) would cost more.
    caps = {("bf-one", 1): ((2.0, 100.0), (5.0, 100.0))}
    instance = Instance((ONE_GPU,), caps, ((1.0, 0.0), (2.0, 0.0)))
    assert brute_force_optimum(instance) == 24.0
    tight = Instance((ONE_GPU,), caps, ((1.0, 0.0), (3.0, 0.0)))  # 0.5 + 0.6 > 1
    assert brute_force_optimum(tight) == 48.0


def test_tensor_parallel_replicas_and_infeasibility() -> None:
    # Only tp2 serves class 0, only tp1 class 1: one two-GPU instance cannot hold both
    # (2 GPUs), so two instances are needed.
    caps = {("bf-two", 1): (None, (4.0, 100.0)), ("bf-two", 2): ((4.0, 100.0), None)}
    instance = Instance((TWO_GPU,), caps, ((1.0, 0.0), (1.0, 0.0)))
    assert brute_force_optimum(instance) == 72.0
    impossible = Instance((TWO_GPU,), caps, ((100.0, 0.0), (1.0, 0.0)))
    assert brute_force_optimum(impossible) is None


def test_random_instances_respect_the_limits() -> None:
    for seed in range(50):
        instance = random_instance(seed)
        assert 1 <= len(instance.rows) <= 3
        assert len(instance.capacities) <= 4
        assert 1 <= len(instance.demands) <= 2
        for k in range(len(instance.demands)):
            assert any(caps[k] is not None for caps in instance.capacities.values())
