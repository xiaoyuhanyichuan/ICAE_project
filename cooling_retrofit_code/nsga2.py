from __future__ import annotations

from collections.abc import Sequence
from math import inf, isfinite

from model import EvaluationResult

OBJECTIVES = ("tlcc", "tce")


def dominates(a: EvaluationResult, b: EvaluationResult) -> bool:
    """Return True when a dominates b under Deb's feasibility rules."""
    if a.feasible != b.feasible:
        return a.feasible

    if not a.feasible and not b.feasible:
        return float(a.violation) < float(b.violation)

    a_values = _objective_values(a)
    b_values = _objective_values(b)
    return all(a_value <= b_value for a_value, b_value in zip(a_values, b_values)) and any(
        a_value < b_value for a_value, b_value in zip(a_values, b_values)
    )


def deb_better(a: EvaluationResult, b: EvaluationResult) -> bool:
    return dominates(a, b)


def fast_non_dominated_sort(results: Sequence[EvaluationResult]) -> list[list[int]]:
    dominated_by: list[int] = [0] * len(results)
    dominates_set: list[list[int]] = [[] for _ in results]
    first_front: list[int] = []

    for i, candidate in enumerate(results):
        for j, other in enumerate(results):
            if i == j:
                continue
            if dominates(candidate, other):
                dominates_set[i].append(j)
            elif dominates(other, candidate):
                dominated_by[i] += 1
        if dominated_by[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = []
    current_front = first_front
    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominates_set[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(
    front: Sequence[int],
    results: Sequence[EvaluationResult],
) -> dict[int, float]:
    distances = {index: 0.0 for index in front}
    if not front:
        return distances

    if len(front) <= 2:
        return {index: inf for index in front}

    for objective in OBJECTIVES:
        ordered = sorted(front, key=lambda index: float(getattr(results[index], objective)))
        distances[ordered[0]] = inf
        distances[ordered[-1]] = inf

        min_value = float(getattr(results[ordered[0]], objective))
        max_value = float(getattr(results[ordered[-1]], objective))
        span = max_value - min_value
        if span == 0.0 or not isfinite(span):
            continue

        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if distances[index] == inf:
                continue
            previous_value = float(getattr(results[ordered[position - 1]], objective))
            next_value = float(getattr(results[ordered[position + 1]], objective))
            neighbor_span = next_value - previous_value
            if isfinite(neighbor_span):
                distances[index] += neighbor_span / span

    return distances


def _objective_values(result: EvaluationResult) -> tuple[float, float]:
    return (float(result.tlcc), float(result.tce))
