from __future__ import annotations

import json
import uuid
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_results import write_report
from benchmark_scenarios import evaluate_benchmark_scenarios
from convergence_analysis import analyze_pymoo_convergence
from data_utils import load_config, load_input_data
from decision import DecisionSchema
from model import EvaluationContext, evaluate_solution as _default_evaluate_solution
from nsga2 import crowding_distance, fast_non_dominated_sort
from plotting import (
    save_benchmark_comparison_plot,
    save_convergence_plots,
    save_pareto_distribution_plot,
    save_pareto_plot,
)
from spatial_analysis import run_spatial_analysis
from typical_days import build_time_index


ZONE_CAPACITY_NAMES = ("cap_ac_new", "cap_rdhx", "cap_cdu")
SYSTEM_CAPACITY_NAMES = (
    "cap_ashp",
    "cap_wshp",
    "cap_bess",
    "cap_tes",
    "delta_cap_chiller",
    "delta_cap_tower",
)

_WORKER_SCHEMA: DecisionSchema | None = None
_WORKER_CONFIG: dict | None = None
_WORKER_TIME_FRAME: pd.DataFrame | None = None
_WORKER_PEAK_LOAD_BY_ZONE_KW: dict[str, float] | None = None
_WORKER_CONTEXT: EvaluationContext | None = None
evaluate_solution = _default_evaluate_solution


def infer_zone_ids(data) -> list[str]:
    for frame, column in (
        (getattr(data, "rack_metadata", pd.DataFrame()), "ac_unit"),
        (getattr(data, "rack_metadata", pd.DataFrame()), "zone"),
        (getattr(data, "ac_hourly", pd.DataFrame()), "ac_unit"),
    ):
        if column not in frame.columns:
            continue
        zone_ids = frame[column].dropna().astype(str).unique().tolist()
        if zone_ids:
            return sorted(zone_ids)
    return ["z1"]


def peak_load_by_zone_from_data(
    time_frame: pd.DataFrame,
    rack_metadata: pd.DataFrame,
    zone_ids: list[str],
) -> dict[str, float]:
    rack_cols = [col for col in time_frame.columns if "rack_it_" in col and col.endswith("_kw")]
    if not rack_cols or "rack_id" not in rack_metadata.columns:
        peak = float(pd.to_numeric(time_frame["it_load_kw"], errors="raise").max())
        return {zone_id: peak / max(1, len(zone_ids)) for zone_id in zone_ids}

    zone_set = set(zone_ids)
    metadata = rack_metadata.set_index("rack_id")
    zone_series = {zone_id: pd.Series(0.0, index=time_frame.index) for zone_id in zone_ids}
    fallback = zone_ids[0] if zone_ids else "z1"
    for col in rack_cols:
        rack_id = col.removesuffix("_kw")
        zone_id = _zone_id_for_rack(rack_id, metadata, zone_set, fallback)
        zone_series[zone_id] = zone_series[zone_id] + pd.to_numeric(time_frame[col], errors="raise")

    return {zone_id: float(series.max()) for zone_id, series in zone_series.items()}


def zone_rack_count_from_metadata(rack_metadata: pd.DataFrame, zone_ids: list[str]) -> dict[str, int]:
    fallback = {zone_id: 1 for zone_id in zone_ids}
    if rack_metadata.empty:
        return fallback
    zone_col = "ac_unit" if "ac_unit" in rack_metadata.columns else "zone" if "zone" in rack_metadata.columns else ""
    if not zone_col:
        return fallback

    counts = rack_metadata[zone_col].dropna().astype(str).value_counts()
    return {zone_id: max(1, int(counts.get(zone_id, 0))) for zone_id in zone_ids}


def _zone_id_for_rack(
    rack_id: str,
    metadata: pd.DataFrame,
    zone_set: set[str],
    fallback: str,
) -> str:
    lookup_id = rack_id
    room_id: str | None = None
    if rack_id not in metadata.index and "_rack_it_" in rack_id:
        room_id, lookup_id = rack_id.split("_", 1)
    if lookup_id not in metadata.index:
        return fallback

    row = metadata.loc[lookup_id]
    for candidate_col in ("ac_unit", "zone"):
        if candidate_col not in metadata.columns:
            continue
        candidate = str(row[candidate_col])
        room_candidate = f"{room_id}_{candidate}" if room_id else candidate
        if room_candidate in zone_set:
            return room_candidate
        if candidate in zone_set:
            return candidate
    return fallback


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _vector_json(vector) -> str:
    if vector is None:
        return ""
    values = np.asarray(vector, dtype=float).tolist()
    return _json_dumps([float(value) for value in values])


def _resolve_output_dir(config: dict, config_file: Path) -> Path:
    output_dir = Path(config["paths"]["output_dir"])
    if output_dir.is_absolute():
        return output_dir
    return config_file.parent / output_dir


def _solution_row(
    solution_id: int,
    vector: np.ndarray,
    result,
    front: int | None,
    generation: int | None = None,
) -> dict:
    artifacts = result.artifacts
    decision = artifacts.get("decision")

    if decision is None:
        config_by_zone_json = ""
        zone_capacity_json = ""
        system_capacity_json = ""
    else:
        config_by_zone_json = _json_dumps(decision.s_z)
        zone_capacity_json = _json_dumps(
            {
                name: getattr(decision, name)
                for name in ZONE_CAPACITY_NAMES
            }
        )
        system_capacity_json = _json_dumps(
            {
                name: float(getattr(decision, name))
                for name in SYSTEM_CAPACITY_NAMES
            }
        )

    screening = artifacts.get("screening")
    weight_soft_penalty = artifacts.get("weight_soft_penalty", {})
    weight_hard = artifacts.get("weight_hard_constraint", {})
    diagnostics = artifacts.get("diagnostics", {})
    return {
        "solution_id": solution_id,
        "generation": generation,
        "feasible": result.feasible,
        "tlcc": result.tlcc,
        "tce": result.tce,
        "violation": result.violation,
        "soft_violation_kg": float(getattr(screening, "soft_violation", 0.0)),
        "soft_reasons": ";".join(getattr(screening, "soft_reasons", [])),
        "weight_soft_penalty_cost_yuan_per_year": float(weight_soft_penalty.get("cost_yuan_per_year", 0.0)),
        "weight_soft_penalty_carbon_kg_per_year": float(weight_soft_penalty.get("carbon_kg_per_year", 0.0)),
        "weight_hard_violation_kg": float(
            weight_hard.get("violation_kg", _weight_hard_violation_kg(screening))
        ),
        "weight_hard_reasons": ";".join(weight_hard.get("reasons", _weight_hard_reasons(screening))),
        "front": front,
        "reasons": ";".join(result.reasons),
        "raw_vector_json": _vector_json(vector),
        "repaired_vector_json": _vector_json(artifacts.get("repaired_vector")),
        "config_by_zone_json": config_by_zone_json,
        "zone_capacity_json": zone_capacity_json,
        "system_capacity_json": system_capacity_json,
        "build_mode": diagnostics.get("build_mode", ""),
        "warm_start_hit": bool(diagnostics.get("warm_start_hit", False)),
        "warm_start_key_match": diagnostics.get("warm_start_key_match", ""),
        "warm_start_attempted": bool(diagnostics.get("warm_start_attempted", False)),
        "warm_start_values_applied": int(diagnostics.get("warm_start_values_applied", 0) or 0),
        "warm_start_cache_entries": int(diagnostics.get("warm_start_cache_entries", 0) or 0),
        "warm_start_exact_hits": int(diagnostics.get("warm_start_exact_hits", 0) or 0),
        "warm_start_neighbor_hits": int(diagnostics.get("warm_start_neighbor_hits", 0) or 0),
        "warm_start_misses": int(diagnostics.get("warm_start_misses", 0) or 0),
        "inner_build_total_s": float(diagnostics.get("inner_build_total_s", 0.0) or 0.0),
        "inner_update_s": float(diagnostics.get("inner_update_s", 0.0) or 0.0),
        "inner_optimize_s": float(diagnostics.get("inner_optimize_s", 0.0) or 0.0),
        "template_build_once_s": float(diagnostics.get("template_build_once_s", 0.0) or 0.0),
        "template_reuse_hit": bool(diagnostics.get("template_reuse_hit", False)),
        "persistent_template_backend": diagnostics.get("persistent_template_backend", ""),
        "parallel_fallback": bool(diagnostics.get("parallel_fallback", False)),
        "parallel_fallback_reason": diagnostics.get("parallel_fallback_reason", ""),
        "parallel_workers": int(diagnostics.get("parallel_workers", 1) or 1),
        "parallel_chunksize": int(diagnostics.get("parallel_chunksize", 1) or 1),
        "gurobi_time_limit_seconds": float(diagnostics.get("gurobi_time_limit_seconds", 0.0) or 0.0),
        "gurobi_mip_gap": float(diagnostics.get("gurobi_mip_gap", 0.0) or 0.0),
        "gurobi_threads": int(diagnostics.get("gurobi_threads", 0) or 0),
        "gurobi_mip_focus": int(diagnostics.get("gurobi_mip_focus", 0) or 0),
        "gurobi_nodefile_start_gb": float(diagnostics.get("gurobi_nodefile_start_gb", 0.0) or 0.0),
        "gurobi_numeric_focus": int(diagnostics.get("gurobi_numeric_focus", 0) or 0),
        "gurobi_output_flag": int(diagnostics.get("gurobi_output_flag", 0) or 0),
        "heat_topology_nonzero_edges": int(diagnostics.get("heat_topology_nonzero_edges", 0) or 0),
        "heat_topology_density": float(diagnostics.get("heat_topology_density", 0.0) or 0.0),
    }


def _weight_hard_reasons(screening) -> list[str]:
    return [
        str(reason)
        for reason in getattr(screening, "reasons", [])
        if "weight_margin_exceeded" in str(reason)
    ]


def _weight_hard_violation_kg(screening) -> float:
    total = 0.0
    for reason in _weight_hard_reasons(screening):
        parts = reason.split(":")
        if len(parts) >= 3 and parts[-2] == "weight_margin_exceeded":
            try:
                total += max(0.0, float(parts[-1]))
            except ValueError:
                continue
    return total


def run(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    data = load_input_data(config)
    time_index = build_time_index(data.hourly, config["typical_days"])

    zone_ids = infer_zone_ids(data)
    config = deepcopy(config)
    config.setdefault("scenario", {})["zone_rack_count"] = zone_rack_count_from_metadata(
        getattr(data, "rack_metadata", pd.DataFrame()),
        zone_ids,
    )
    schema = _build_schema(zone_ids, config)

    nsga2_config = config["solver"]["nsga2"]
    rng = np.random.default_rng(int(nsga2_config["random_seed"]))
    population_size = int(nsga2_config["population_size"])
    generations = int(nsga2_config.get("generations", 0))
    crossover_probability = float(nsga2_config.get("crossover_probability", 0.9))
    mutation_probability = float(nsga2_config.get("mutation_probability", 0.1))
    population = [schema.random_vector(rng) for _ in range(population_size)]

    peak_load_by_zone_kw = peak_load_by_zone_from_data(
        time_index.frame,
        getattr(data, "rack_metadata", pd.DataFrame()),
        zone_ids,
    )

    all_vectors: dict[int, np.ndarray] = {}
    all_results = {}
    all_generations: dict[int, int] = {}
    next_solution_id = 0

    with PopulationEvaluator(schema, config, time_index.frame, peak_load_by_zone_kw) as evaluator:
        evaluations = evaluator.evaluate(population)
        population_solution_ids: list[int] = []
        for vector, result in zip(population, evaluations):
            all_vectors[next_solution_id] = vector
            all_results[next_solution_id] = result
            all_generations[next_solution_id] = 0
            population_solution_ids.append(next_solution_id)
            next_solution_id += 1

        for generation in range(1, generations + 1):
            offspring = _make_offspring(
                population,
                evaluations,
                schema,
                rng,
                crossover_probability,
                mutation_probability,
            )
            offspring_evaluations = evaluator.evaluate(offspring)
            offspring_solution_ids: list[int] = []
            for vector, result in zip(offspring, offspring_evaluations):
                all_vectors[next_solution_id] = vector
                all_results[next_solution_id] = result
                all_generations[next_solution_id] = generation
                offspring_solution_ids.append(next_solution_id)
                next_solution_id += 1

            combined_population = population + offspring
            combined_evaluations = evaluations + offspring_evaluations
            combined_solution_ids = population_solution_ids + offspring_solution_ids
            selected = _select_next_population_indices(combined_evaluations, population_size)
            population = [combined_population[index] for index in selected]
            evaluations = [combined_evaluations[index] for index in selected]
            population_solution_ids = [combined_solution_ids[index] for index in selected]

    feasible_solution_ids = [
        solution_id
        for solution_id, result in all_results.items()
        if bool(result.feasible)
    ]
    feasible_evaluations = [all_results[solution_id] for solution_id in feasible_solution_ids]
    front_by_solution = {}
    if feasible_evaluations:
        fronts = fast_non_dominated_sort(feasible_evaluations)
        front_by_solution = {
            feasible_solution_ids[index]: front_id
            for front_id, front in enumerate(fronts)
            for index in front
        }

    all_evaluations = pd.DataFrame(
        [
            _solution_row(
                solution_id,
                all_vectors[solution_id],
                result,
                front_by_solution.get(solution_id),
                all_generations.get(solution_id),
            )
            for solution_id, result in all_results.items()
        ]
    )
    pareto = all_evaluations[
        (all_evaluations["front"] == 0) & (all_evaluations["feasible"])
    ].copy()

    output_dir = _resolve_output_dir(config, config_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_growth_age_path = output_dir / "scenario_growth_age_values.csv"
    if isinstance(getattr(data, "scenario_growth_age", None), pd.DataFrame):
        data.scenario_growth_age.to_csv(scenario_growth_age_path, index=False, encoding="utf-8-sig")
    artifact_rows = _write_pareto_artifacts(output_dir, pareto, all_vectors, all_results)
    if artifact_rows:
        artifacts = pd.DataFrame(artifact_rows)
        pareto = pareto.merge(artifacts, on="solution_id", how="left")
    all_evaluations.to_csv(output_dir / "all_evaluations.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(output_dir / "pareto_solutions.csv", index=False, encoding="utf-8-sig")
    figure_path = save_pareto_plot(pareto, output_dir) if not pareto.empty else None
    convergence_result = analyze_pymoo_convergence(all_evaluations, output_dir, config)
    convergence_figure_paths = save_convergence_plots(convergence_result["metrics"], output_dir)
    distribution_path = save_pareto_distribution_plot(pareto, output_dir, config) if not pareto.empty else None
    spatial_result = run_spatial_analysis(
        pareto=pareto,
        rack_metadata=getattr(data, "rack_metadata", pd.DataFrame()),
        hourly=time_index.frame,
        config=config,
        output_dir=output_dir,
    ) if not pareto.empty else {"available": False, "reason": "empty_pareto"}
    benchmark_comparison = evaluate_benchmark_scenarios(
        schema=schema,
        config=config,
        time_frame=time_index.frame,
        peak_load_by_zone_kw=peak_load_by_zone_kw,
        pareto=pareto,
    )
    benchmark_path = output_dir / "benchmark_comparison.csv"
    benchmark_comparison.to_csv(benchmark_path, index=False, encoding="utf-8-sig")
    benchmark_plot_path = save_benchmark_comparison_plot(benchmark_comparison, output_dir)
    report_path = write_report(all_evaluations, pareto, config, output_dir)

    return {
        "config": config,
        "all_evaluations": all_evaluations,
        "pareto": pareto,
        "output_dir": output_dir,
        "report_path": report_path,
        "figure_path": figure_path,
        "convergence_result": convergence_result,
        "convergence_figure_paths": convergence_figure_paths,
        "distribution_path": distribution_path,
        "spatial_result": spatial_result,
        "benchmark_comparison": benchmark_comparison,
        "benchmark_path": benchmark_path,
        "benchmark_plot_path": benchmark_plot_path,
        "scenario_growth_age_path": scenario_growth_age_path,
    }


def _evaluate_population(
    population: list[np.ndarray],
    schema: DecisionSchema,
    config: dict,
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
) -> list:
    with PopulationEvaluator(schema, config, time_frame, peak_load_by_zone_kw) as evaluator:
        return evaluator.evaluate(population)


class PopulationEvaluator:
    def __init__(
        self,
        schema: DecisionSchema,
        config: dict,
        time_frame: pd.DataFrame,
        peak_load_by_zone_kw: dict[str, float],
    ) -> None:
        self.schema = schema
        self.config = config
        self.time_frame = time_frame
        self.peak_load_by_zone_kw = peak_load_by_zone_kw
        self.executor: ProcessPoolExecutor | None = None
        self.context: EvaluationContext | None = None
        parallel_config = config.get("solver", {}).get("parallel", {})
        self.workers = int(parallel_config.get("workers", 1)) if isinstance(parallel_config, dict) else 1
        self.parallel_enabled = bool(parallel_config.get("enabled", False)) if isinstance(parallel_config, dict) else False
        self.strict = bool(parallel_config.get("strict", False)) if isinstance(parallel_config, dict) else False
        self.chunksize = max(1, int(parallel_config.get("chunksize", 1))) if isinstance(parallel_config, dict) else 1
        self.use_legacy_evaluate_hook = evaluate_solution is not _default_evaluate_solution
        self.parallel_fallback_reason = ""

    def __enter__(self) -> "PopulationEvaluator":
        if self.use_legacy_evaluate_hook:
            return self
        if self.parallel_enabled and self.workers > 1:
            worker_config = _parallel_worker_config(self.config)
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_parallel_worker,
                initargs=(self.schema, worker_config, self.time_frame, self.peak_load_by_zone_kw),
            )
        else:
            self.context = EvaluationContext(
                self.schema,
                self.config,
                self.time_frame,
                self.peak_load_by_zone_kw,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._shutdown_executor()
        return False

    def evaluate(self, population: list[np.ndarray]) -> list:
        if not population:
            return []
        if self.use_legacy_evaluate_hook:
            return [
                evaluate_solution(
                    vector,
                    self.schema,
                    self.config,
                    self.time_frame,
                    self.peak_load_by_zone_kw,
                )
                for vector in population
            ]
        if self.executor is not None:
            try:
                results = list(
                    self.executor.map(
                        _evaluate_vector_in_parallel_worker,
                        population,
                        chunksize=self.chunksize,
                    )
                )
                return self._attach_parallel_diagnostics(results, fallback=False, reason="")
            except Exception as exc:
                if self.strict or not _is_parallel_memory_failure(exc):
                    raise
                reason = _exception_summary(exc)
                self.parallel_fallback_reason = reason
                self._shutdown_executor()
                if self.context is None:
                    self.context = EvaluationContext(
                        self.schema,
                        self.config,
                        self.time_frame,
                        self.peak_load_by_zone_kw,
                    )
                results = [self.context.evaluate(vector) for vector in population]
                return self._attach_parallel_diagnostics(results, fallback=True, reason=reason)
        if self.context is None:
            self.context = EvaluationContext(
                self.schema,
                self.config,
                self.time_frame,
                self.peak_load_by_zone_kw,
            )
        results = [self.context.evaluate(vector) for vector in population]
        fallback = bool(self.parallel_fallback_reason)
        return self._attach_parallel_diagnostics(
            results,
            fallback=fallback,
            reason=self.parallel_fallback_reason if fallback else "",
        )

    def _shutdown_executor(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def _attach_parallel_diagnostics(self, results: list, fallback: bool, reason: str) -> list:
        active_workers = self.workers if self.executor is not None and self.parallel_enabled and self.workers > 1 else 1
        for result in results:
            artifacts = getattr(result, "artifacts", None)
            if not isinstance(artifacts, dict):
                continue
            diagnostics = artifacts.setdefault("diagnostics", {})
            diagnostics["parallel_fallback"] = bool(fallback)
            diagnostics["parallel_fallback_reason"] = reason if fallback else ""
            diagnostics["parallel_workers"] = int(active_workers)
            diagnostics["parallel_chunksize"] = int(self.chunksize)
        return results


def _is_parallel_memory_failure(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, BrokenProcessPool, OSError)):
        return True
    text = _exception_summary(exc).lower()
    return any(
        token in text
        for token in (
            "out of memory",
            "oom",
            "memoryerror",
            "cannot allocate memory",
            "not enough memory",
            "brokenprocesspool",
        )
    )


def _exception_summary(exc: BaseException) -> str:
    parts = [type(exc).__name__, str(exc)]
    for attr in ("__cause__", "__context__"):
        nested = getattr(exc, attr, None)
        if nested is not None:
            parts.extend([type(nested).__name__, str(nested)])
    return ": ".join(part for part in parts if part)


def _parallel_worker_config(config: dict) -> dict:
    worker_config = deepcopy(config)
    solver = worker_config.setdefault("solver", {})
    gurobi = solver.setdefault("gurobi", {})
    parallel = solver.get("parallel", {})
    if isinstance(parallel, dict) and "gurobi_threads_per_worker" in parallel:
        gurobi["threads"] = int(parallel["gurobi_threads_per_worker"])
    return worker_config


def _init_parallel_worker(
    schema: DecisionSchema,
    config: dict,
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
) -> None:
    global _WORKER_SCHEMA, _WORKER_CONFIG, _WORKER_TIME_FRAME, _WORKER_PEAK_LOAD_BY_ZONE_KW, _WORKER_CONTEXT
    _WORKER_SCHEMA = schema
    _WORKER_CONFIG = config
    _WORKER_TIME_FRAME = time_frame
    _WORKER_PEAK_LOAD_BY_ZONE_KW = peak_load_by_zone_kw
    _WORKER_CONTEXT = EvaluationContext(schema, config, time_frame, peak_load_by_zone_kw)


def _evaluate_vector_in_parallel_worker(vector: np.ndarray):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Parallel worker was not initialized.")
    return _WORKER_CONTEXT.evaluate(vector)


def _build_schema(zone_ids: list[str], config: dict) -> DecisionSchema:
    if hasattr(DecisionSchema, "from_config"):
        return DecisionSchema.from_config(zone_ids, config)
    return DecisionSchema(zone_ids)


def _make_offspring(
    population: list[np.ndarray],
    evaluations: list,
    schema: DecisionSchema,
    rng: np.random.Generator,
    crossover_probability: float,
    mutation_probability: float,
) -> list[np.ndarray]:
    lower, upper = schema.bounds()
    ranks, distances = _rank_and_crowding(evaluations)
    offspring: list[np.ndarray] = []
    while len(offspring) < len(population):
        parent_a = population[_tournament_index(rng, ranks, distances)]
        parent_b = population[_tournament_index(rng, ranks, distances)]
        child_a, child_b = _crossover(parent_a, parent_b, rng, crossover_probability)
        offspring.append(_mutate(child_a, schema, rng, mutation_probability, lower, upper))
        if len(offspring) < len(population):
            offspring.append(_mutate(child_b, schema, rng, mutation_probability, lower, upper))
    return offspring


def _rank_and_crowding(evaluations: list) -> tuple[dict[int, int], dict[int, float]]:
    fronts = fast_non_dominated_sort(evaluations)
    ranks: dict[int, int] = {}
    distances: dict[int, float] = {}
    for rank, front in enumerate(fronts):
        ranks.update({index: rank for index in front})
        distances.update(crowding_distance(front, evaluations))
    return ranks, distances


def _tournament_index(
    rng: np.random.Generator,
    ranks: dict[int, int],
    distances: dict[int, float],
) -> int:
    a, b = rng.integers(0, len(ranks), size=2)
    rank_a = ranks[int(a)]
    rank_b = ranks[int(b)]
    if rank_a != rank_b:
        return int(a) if rank_a < rank_b else int(b)
    distance_a = distances.get(int(a), 0.0)
    distance_b = distances.get(int(b), 0.0)
    if distance_a != distance_b:
        return int(a) if distance_a > distance_b else int(b)
    return int(a) if rng.random() < 0.5 else int(b)


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    child_a = np.asarray(parent_a, dtype=float).copy()
    child_b = np.asarray(parent_b, dtype=float).copy()
    if rng.random() >= probability:
        return child_a, child_b

    mask = rng.random(child_a.shape) < 0.5
    child_a[mask] = parent_b[mask]
    child_b[mask] = parent_a[mask]
    return child_a, child_b


def _mutate(
    vector: np.ndarray,
    schema: DecisionSchema,
    rng: np.random.Generator,
    probability: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    mutated = np.clip(np.asarray(vector, dtype=float).copy(), lower, upper)
    active_configs = getattr(schema, "active_config_values", tuple(range(7)))
    zone_count = len(getattr(schema, "zone_ids", []))
    for index in range(mutated.size):
        if rng.random() >= probability:
            continue
        if index < zone_count:
            mutated[index] = rng.choice(active_configs)
        else:
            mutated[index] = rng.uniform(lower[index], upper[index])
    return np.clip(mutated, lower, upper)


def _select_next_population_indices(evaluations: list, population_size: int) -> list[int]:
    selected: list[int] = []
    for front in fast_non_dominated_sort(evaluations):
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue
        distances = crowding_distance(front, evaluations)
        ordered = sorted(front, key=lambda index: (-distances.get(index, 0.0), index))
        selected.extend(ordered[: population_size - len(selected)])
        break
    return selected


def _write_pareto_artifacts(
    output_dir: str | Path,
    pareto: pd.DataFrame,
    vectors_by_solution: dict[int, np.ndarray],
    results_by_solution: dict[int, Any],
) -> list[dict[str, object]]:
    details_dir = Path(output_dir) / "pareto_details" / f"run_{uuid.uuid4().hex}"
    details_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []

    for _, row in pareto.iterrows():
        solution_id = int(row["solution_id"])
        result = results_by_solution[solution_id]
        artifacts = result.artifacts
        dispatch = artifacts.get("dispatch")
        dispatch_path = details_dir / f"dispatch_solution_{solution_id}.csv"
        if isinstance(dispatch, pd.DataFrame):
            dispatch.to_csv(dispatch_path, index=False, encoding="utf-8-sig")

        solution_path = details_dir / f"solution_{solution_id}.json"
        payload = {
            "solution_id": solution_id,
            "feasible": bool(result.feasible),
            "objectives": {"tlcc": float(result.tlcc), "tce": float(result.tce)},
            "violation": float(result.violation),
            "reasons": list(result.reasons),
            "raw_vector": _json_safe(vectors_by_solution.get(solution_id)),
            "repaired_vector": _json_safe(artifacts.get("repaired_vector")),
            "decision": _json_safe(artifacts.get("decision")),
            "repair_actions": _json_safe(artifacts.get("repair_actions", [])),
            "screening": _json_safe(artifacts.get("screening")),
            "diagnostics": _json_safe(artifacts.get("diagnostics", {})),
            "dispatch_path": str(dispatch_path),
        }
        solution_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        written.append(
            {
                "solution_id": solution_id,
                "solution_json_path": solution_path,
                "dispatch_path": dispatch_path,
            }
        )

    return written



def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
