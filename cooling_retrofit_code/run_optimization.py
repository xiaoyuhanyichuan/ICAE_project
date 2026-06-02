from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_results import write_report
from convergence_analysis import analyze_pymoo_convergence
from data_utils import load_config, load_input_data
from decision import DecisionSchema
from model import evaluate_solution
from nsga2 import crowding_distance, fast_non_dominated_sort
from plotting import save_convergence_plots, save_pareto_distribution_plot, save_pareto_plot
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
    rack_cols = [col for col in time_frame.columns if col.startswith("rack_it_") and col.endswith("_kw")]
    if not rack_cols or "rack_id" not in rack_metadata.columns:
        peak = float(pd.to_numeric(time_frame["it_load_kw"], errors="raise").max())
        return {zone_id: peak / max(1, len(zone_ids)) for zone_id in zone_ids}

    zone_set = set(zone_ids)
    metadata = rack_metadata.set_index("rack_id")
    zone_series = {zone_id: pd.Series(0.0, index=time_frame.index) for zone_id in zone_ids}
    fallback = zone_ids[0] if zone_ids else "z1"
    for col in rack_cols:
        rack_id = col.removesuffix("_kw")
        zone_id = fallback
        if rack_id in metadata.index:
            row = metadata.loc[rack_id]
            for candidate_col in ("ac_unit", "zone"):
                if candidate_col in metadata.columns:
                    candidate = str(row[candidate_col])
                    if candidate in zone_set:
                        zone_id = candidate
                        break
        zone_series[zone_id] = zone_series[zone_id] + pd.to_numeric(time_frame[col], errors="raise")

    return {zone_id: float(series.max()) for zone_id, series in zone_series.items()}


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
        "front": front,
        "reasons": ";".join(result.reasons),
        "raw_vector_json": _vector_json(vector),
        "repaired_vector_json": _vector_json(artifacts.get("repaired_vector")),
        "config_by_zone_json": config_by_zone_json,
        "zone_capacity_json": zone_capacity_json,
        "system_capacity_json": system_capacity_json,
    }


def run(config_path: str | Path) -> dict[str, object]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    data = load_input_data(config)
    time_index = build_time_index(data.hourly, config["typical_days"])

    zone_ids = infer_zone_ids(data)
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

    evaluations = _evaluate_population(population, schema, config, time_index.frame, peak_load_by_zone_kw)
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
        offspring_evaluations = _evaluate_population(
            offspring,
            schema,
            config,
            time_index.frame,
            peak_load_by_zone_kw,
        )
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
        "scenario_growth_age_path": scenario_growth_age_path,
    }


def _evaluate_population(
    population: list[np.ndarray],
    schema: DecisionSchema,
    config: dict,
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
) -> list:
    return [
        evaluate_solution(
            vector,
            schema,
            config,
            time_frame,
            peak_load_by_zone_kw,
        )
        for vector in population
    ]


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
