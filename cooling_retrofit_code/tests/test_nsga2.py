from math import inf, isnan

import pytest

from model import EvaluationResult
from nsga2 import crowding_distance, deb_better, dominates, fast_non_dominated_sort


def _result(
    tlcc: float,
    tce: float,
    *,
    feasible: bool = True,
    violation: float = 0.0,
) -> EvaluationResult:
    return EvaluationResult(
        feasible=feasible,
        tlcc=tlcc,
        tce=tce,
        violation=violation,
    )


def test_deb_rule_prefers_feasible_over_infeasible_with_better_objectives():
    feasible = _result(10.0, 10.0, feasible=True)
    infeasible = _result(1.0, 1.0, feasible=False, violation=0.1)

    assert dominates(feasible, infeasible)
    assert deb_better(feasible, infeasible)
    assert not dominates(infeasible, feasible)


def test_fast_non_dominated_sort_places_dominated_feasible_solutions_later():
    results = [
        _result(10.0, 10.0),
        _result(8.0, 12.0),
        _result(12.0, 8.0),
        _result(12.0, 12.0),
    ]

    fronts = fast_non_dominated_sort(results)

    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]


def test_crowding_distance_marks_objective_endpoints_infinite():
    results = [
        _result(1.0, 4.0),
        _result(2.0, 2.0),
        _result(4.0, 1.0),
    ]

    distances = crowding_distance([0, 1, 2], results)

    assert distances[0] == inf
    assert distances[2] == inf
    assert distances[1] == pytest.approx(2.0)


def test_crowding_distance_does_not_return_nan_for_all_inf_infeasible_front():
    results = [
        _result(inf, inf, feasible=False, violation=1.0),
        _result(inf, inf, feasible=False, violation=1.0),
        _result(inf, inf, feasible=False, violation=1.0),
    ]

    distances = crowding_distance([0, 1, 2], results)

    assert not any(isnan(distance) for distance in distances.values())


def test_fast_sort_and_crowding_can_select_best_front_before_crowded_tail():
    results = [
        _result(1.0, 5.0),
        _result(5.0, 1.0),
        _result(3.0, 3.0),
        _result(6.0, 6.0),
        _result(7.0, 7.0),
    ]

    fronts = fast_non_dominated_sort(results)
    distances = crowding_distance(fronts[0], results)

    assert set(fronts[0]) == {0, 1, 2}
    assert distances[0] == inf
    assert distances[1] == inf
    assert fronts[1] == [3]
