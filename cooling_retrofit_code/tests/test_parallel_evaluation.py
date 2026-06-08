import numpy as np
import pandas as pd
import pytest

from decision import DecisionSchema
from model import EvaluationResult
import run_optimization
from run_optimization import _evaluate_population


def test_evaluate_population_uses_configured_parallel_workers_and_worker_gurobi_threads(monkeypatch):
    schema = DecisionSchema(["z1"])
    population = [schema.encode_default() for _ in range(3)]
    for index, vector in enumerate(population):
        vector[0] = index
    config = {
        "solver": {
            "gurobi": {"threads": 0},
            "parallel": {
                "enabled": True,
                "workers": 4,
                "strict": False,
                "chunksize": 1,
                "gurobi_threads_per_worker": 1,
            },
        }
    }
    frame = pd.DataFrame({"it_load_kw": [10.0]})
    peaks = {"z1": 10.0}
    calls = {"max_workers": None, "thread_setting": None, "chunksize": None}

    class FakeExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            calls["max_workers"] = max_workers
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items, chunksize=1):
            calls["chunksize"] = chunksize
            return [fn(item) for item in items]

        def shutdown(self, wait=True):
            return None

    class FakeEvaluationContext:
        def __init__(self, schema_arg, config_arg, frame_arg, peaks_arg):
            calls["thread_setting"] = config_arg["solver"]["gurobi"]["threads"]

        def evaluate(self, vector):
            return EvaluationResult(
                feasible=True,
                tlcc=float(vector[0]),
                tce=float(vector[0]) + 10.0,
                artifacts={"repaired_vector": np.asarray(vector, dtype=float)},
            )

    monkeypatch.setattr(run_optimization, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run_optimization, "EvaluationContext", FakeEvaluationContext)

    results = _evaluate_population(population, schema, config, frame, peaks)

    assert calls["max_workers"] == 4
    assert calls["thread_setting"] == 1
    assert calls["chunksize"] == 1
    assert [result.tlcc for result in results] == [0.0, 1.0, 2.0]
    assert [result.artifacts["diagnostics"]["parallel_fallback"] for result in results] == [False, False, False]


def test_population_evaluator_reuses_one_executor_across_multiple_batches(monkeypatch):
    schema = DecisionSchema(["z1"])
    population = [schema.encode_default() for _ in range(2)]
    config = {
        "solver": {
            "gurobi": {"threads": 0},
            "parallel": {
                "enabled": True,
                "workers": 4,
                "strict": False,
                "chunksize": 1,
                "gurobi_threads_per_worker": 1,
            },
        }
    }
    frame = pd.DataFrame({"it_load_kw": [10.0]})
    peaks = {"z1": 10.0}
    calls = {"executor_init": 0, "shutdown": 0, "context_init": 0}

    class FakeExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            calls["executor_init"] += 1
            initializer(*initargs)

        def map(self, fn, items, chunksize=1):
            return [fn(item) for item in items]

        def shutdown(self, wait=True):
            calls["shutdown"] += 1

    class FakeEvaluationContext:
        def __init__(self, schema_arg, config_arg, frame_arg, peaks_arg):
            calls["context_init"] += 1

        def evaluate(self, vector):
            return EvaluationResult(
                feasible=True,
                tlcc=float(vector[0]),
                tce=float(vector[0]),
                artifacts={"repaired_vector": np.asarray(vector, dtype=float)},
            )

    monkeypatch.setattr(run_optimization, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run_optimization, "EvaluationContext", FakeEvaluationContext)

    with run_optimization.PopulationEvaluator(schema, config, frame, peaks) as evaluator:
        first = evaluator.evaluate(population)
        second = evaluator.evaluate(population)

    assert calls == {"executor_init": 1, "shutdown": 1, "context_init": 1}
    assert len(first) == len(second) == 2


def test_population_evaluator_falls_back_to_serial_on_parallel_oom_when_not_strict(monkeypatch):
    schema = DecisionSchema(["z1"])
    population = [schema.encode_default() for _ in range(2)]
    config = {
        "solver": {
            "parallel": {
                "enabled": True,
                "workers": 2,
                "strict": False,
                "chunksize": 1,
            },
        }
    }
    frame = pd.DataFrame({"it_load_kw": [10.0]})
    peaks = {"z1": 10.0}
    calls = {"shutdown": 0, "context_init": 0}

    class FailingExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            return None

        def map(self, fn, items, chunksize=1):
            raise MemoryError("out of memory in worker")

        def shutdown(self, wait=True):
            calls["shutdown"] += 1

    class FakeEvaluationContext:
        def __init__(self, schema_arg, config_arg, frame_arg, peaks_arg):
            calls["context_init"] += 1

        def evaluate(self, vector):
            return EvaluationResult(
                feasible=True,
                tlcc=float(vector[0]),
                tce=float(vector[0]),
                artifacts={"repaired_vector": np.asarray(vector, dtype=float)},
            )

    monkeypatch.setattr(run_optimization, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(run_optimization, "EvaluationContext", FakeEvaluationContext)

    with run_optimization.PopulationEvaluator(schema, config, frame, peaks) as evaluator:
        results = evaluator.evaluate(population)

    assert calls == {"shutdown": 1, "context_init": 1}
    assert len(results) == 2
    for result in results:
        diagnostics = result.artifacts["diagnostics"]
        assert diagnostics["parallel_fallback"] is True
        assert "out of memory" in diagnostics["parallel_fallback_reason"].lower()
        assert diagnostics["parallel_workers"] == 1
        assert diagnostics["parallel_chunksize"] == 1


def test_population_evaluator_raises_parallel_oom_when_strict(monkeypatch):
    schema = DecisionSchema(["z1"])
    population = [schema.encode_default()]
    config = {
        "solver": {
            "parallel": {
                "enabled": True,
                "workers": 2,
                "strict": True,
                "chunksize": 1,
            },
        }
    }
    frame = pd.DataFrame({"it_load_kw": [10.0]})
    peaks = {"z1": 10.0}

    class FailingExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            return None

        def map(self, fn, items, chunksize=1):
            raise MemoryError("out of memory in worker")

        def shutdown(self, wait=True):
            return None

    monkeypatch.setattr(run_optimization, "ProcessPoolExecutor", FailingExecutor)

    with run_optimization.PopulationEvaluator(schema, config, frame, peaks) as evaluator:
        with pytest.raises(MemoryError):
            evaluator.evaluate(population)
