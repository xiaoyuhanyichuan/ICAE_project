from __future__ import annotations

import cProfile
import copy
import io
import json
import os
import pstats
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import numpy as np
import pandas as pd

from data_utils import ensure_time_series, load_config
from convergence_analysis import analyze_pymoo_convergence
from model import (
    build_typical_day_data,
    evaluate_solution,
    get_decision_schema,
    get_profile_counters,
    make_cache_context,
    reset_profile_counters,
)
from nsga2 import run_nsga2
from pymoo_adapter import run_pymoo_nsga2


ROOT = Path(__file__).resolve().parents[1]
_WORKER_CFG: Dict[str, Any] | None = None
_WORKER_TS: pd.DataFrame | None = None
_WORKER_TD = None
_WORKER_CACHE: Dict[str, Any] | None = None


def _safe_float(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _format_duration_hms(seconds: float) -> str:
    sec = max(0.0, float(seconds))
    total = int(round(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _series_stats(values: list[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _mean_from_reports(reports: list[dict[str, float]], key: str) -> float:
    vals = np.asarray([_safe_float(r.get(key, float("nan"))) for r in reports], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _generation_weighted_stats(
    reports: list[dict[str, float]],
    value_prefix: str,
) -> Dict[str, float]:
    mins: list[float] = []
    maxs: list[float] = []
    weighted_sum = 0.0
    total_count = 0.0
    for row in reports:
        count = _safe_float(row.get("eval_count", float("nan")))
        vmin = _safe_float(row.get(f"{value_prefix}_min", float("nan")))
        vmean = _safe_float(row.get(f"{value_prefix}_mean", float("nan")))
        vmax = _safe_float(row.get(f"{value_prefix}_max", float("nan")))
        if np.isfinite(vmin):
            mins.append(vmin)
        if np.isfinite(vmax):
            maxs.append(vmax)
        if np.isfinite(count) and count > 0 and np.isfinite(vmean):
            weighted_sum += count * vmean
            total_count += count
    return {
        "min": float(np.min(np.asarray(mins, dtype=float))) if mins else float("nan"),
        "mean": float(weighted_sum / total_count) if total_count > 0 else float("nan"),
        "max": float(np.max(np.asarray(maxs, dtype=float))) if maxs else float("nan"),
        "count": float(total_count),
    }


def _parallel_preflight(pool: ProcessPoolExecutor) -> None:
    # Minimal process-pool smoke test (includes initializer execution).
    vals = list(pool.map(abs, [-1, -2], chunksize=1))
    if vals != [1, 2]:
        raise RuntimeError(f"Parallel preflight failed: unexpected map result {vals}")


def _init_worker(cfg: Dict[str, Any], ts: pd.DataFrame) -> None:
    global _WORKER_CFG, _WORKER_TS, _WORKER_TD, _WORKER_CACHE
    _WORKER_CFG = cfg
    _WORKER_TS = ts
    reset_profile_counters()
    _WORKER_TD = build_typical_day_data(cfg, ts, count_build=True)
    _WORKER_CACHE = make_cache_context(cfg, ts)


def _worker_eval(x: np.ndarray):
    return evaluate_solution(x, _WORKER_CFG, _WORKER_TS, td=_WORKER_TD, cache_ctx=_WORKER_CACHE)


def _build_bounds(cfg: Dict[str, Any]):
    schema = get_decision_schema(cfg)
    return schema.names, schema.bounds, schema.int_idx, schema.bin_idx


def _narrow_bounds(
    bounds: list[tuple[float, float]],
    pareto: list,
    int_idx: list[int],
    pad_ratio: float,
    abs_pad_ratio: float,
) -> list[tuple[float, float]]:
    if not pareto:
        return bounds
    xs = np.array([ind.x for ind in pareto], dtype=float)
    narrowed: list[tuple[float, float]] = []
    for i, (lo, hi) in enumerate(bounds):
        xlo = float(np.min(xs[:, i]))
        xhi = float(np.max(xs[:, i]))
        span = hi - lo
        pad = max(span * abs_pad_ratio, (xhi - xlo) * pad_ratio, 1e-6)
        nlo = max(lo, xlo - pad)
        nhi = min(hi, xhi + pad)
        if i in int_idx:
            nlo = float(np.floor(nlo))
            nhi = float(np.ceil(nhi))
        if nlo > nhi:
            nlo, nhi = lo, hi
        narrowed.append((nlo, nhi))
    return narrowed


def _collect_pandas_numpy_hotspots(stats_obj: pstats.Stats, top_k: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (filename, lineno, func), (cc, nc, tt, ct, _) in stats_obj.stats.items():
        f_low = filename.replace("\\", "/").lower()
        fn_low = str(func).lower()
        hit = (
            ("pandas" in f_low)
            or ("numpy" in f_low)
            or ("dataframe" in fn_low)
            or ("to_numpy" in fn_low)
            or ("concatenate" in fn_low)
            or ("mean" in fn_low)
            or ("std" in fn_low)
        )
        if not hit:
            continue
        rows.append(
            {
                "file": filename,
                "line": int(lineno),
                "func": str(func),
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "tottime_s": float(tt),
                "cumtime_s": float(ct),
            }
        )
    rows.sort(key=lambda r: (r["cumtime_s"], r["total_calls"]), reverse=True)
    return rows[:top_k]


def _typical_day_build_calls_in_cprofile(stats_obj: pstats.Stats) -> int:
    calls = 0
    for (filename, _, func), (_, nc, _, _, _) in stats_obj.stats.items():
        f_low = filename.replace("\\", "/").lower()
        if "typical_days.py" in f_low and str(func) == "build":
            calls += int(nc)
    return calls


def _profile_cfg_run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg_run = copy.deepcopy(cfg)
    p_cfg = cfg_run.get("solver", {}).get("profiling", {})
    if not bool(p_cfg.get("enabled", False)):
        return cfg_run

    if bool(p_cfg.get("small_run", True)):
        solver = cfg_run["solver"]
        solver["population_size"] = int(p_cfg.get("small_population_size", min(24, int(solver.get("population_size", 24)))))
        solver["generations"] = int(p_cfg.get("small_generations", min(8, int(solver.get("generations", 8)))))
        if bool(p_cfg.get("disable_two_stage", True)):
            solver.setdefault("two_stage", {})
            solver["two_stage"]["enabled"] = False

    if bool(p_cfg.get("force_single_process", True)):
        cfg_run["solver"].setdefault("parallel", {})
        cfg_run["solver"]["parallel"]["enabled"] = False
    return cfg_run


def _run_with_optional_profiler(
    run_callable,
    cfg_run: Dict[str, Any],
    generation_reports: list[dict[str, float]],
    cache_ctx: Dict[str, Any] | None,
):
    p_cfg = cfg_run.get("solver", {}).get("profiling", {})
    if not bool(p_cfg.get("enabled", False)):
        return run_callable()

    profile_dir = ROOT / str(p_cfg.get("profile_output_dir", "results/profile"))
    if not profile_dir.is_absolute():
        profile_dir = ROOT / profile_dir
    profile_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    lp = None
    if bool(p_cfg.get("enable_line_profiler", False)):
        try:
            from line_profiler import LineProfiler

            lp = LineProfiler()
            lp.add_function(evaluate_solution)
        except Exception:
            lp = None

    profiler.enable()
    if lp is not None:
        lp.enable_by_count()
    out = run_callable()
    if lp is not None:
        lp.disable_by_count()
    profiler.disable()

    stats_obj = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    s = io.StringIO()
    stats_obj.stream = s
    stats_obj.print_stats(int(p_cfg.get("sample_count", 120)))
    (profile_dir / "cprofile_top.txt").write_text(s.getvalue(), encoding="utf-8")
    stats_obj.dump_stats(str(profile_dir / "cprofile.prof"))

    if lp is not None:
        lp.dump_stats(str(profile_dir / "line_profile.lprof"))

    counters = get_profile_counters()
    hot = _collect_pandas_numpy_hotspots(stats_obj, top_k=int(p_cfg.get("pandas_numpy_top_k", 40)))
    cache_stats = (cache_ctx or {}).get("stats", {}) if cache_ctx is not None else {}
    cache_unique = len((cache_ctx or {}).get("l2", {})) if cache_ctx is not None else 0
    l2_h = int(cache_stats.get("l2_hit", 0))
    l2_m = int(cache_stats.get("l2_miss", 0))
    hit_rate = float(l2_h) / max(l2_h + l2_m, 1)

    profile_summary = {
        "typical_day_build_count": int(counters.get("typical_day_build_count", 0)),
        "typical_day_build_count_cprofile": _typical_day_build_calls_in_cprofile(stats_obj),
        "pandas_numpy_call_hotspots": hot,
        "cache_hit_rate": hit_rate,
        "cache_stats": {k: int(v) for k, v in cache_stats.items()},
        "cache_unique_keys": int(cache_unique),
        "generation_cache_reports": generation_reports,
    }
    (profile_dir / "profile_summary.json").write_text(json.dumps(profile_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if generation_reports:
        pd.DataFrame(generation_reports).to_csv(profile_dir / "generation_cache_report.csv", index=False, encoding="utf-8-sig")

    return out


def main() -> None:
    t_main = perf_counter()
    cfg = load_config()
    ts = ensure_time_series(cfg)
    cfg_run = _profile_cfg_run(cfg)
    reset_profile_counters()

    keys, bounds, int_idx, bin_idx = _build_bounds(cfg_run)
    solver_cfg = cfg_run["solver"]
    outer_engine = str(solver_cfg.get("outer_engine", "custom")).lower()
    parallel_cfg = solver_cfg.get("parallel", {})
    two_stage_cfg = solver_cfg.get("two_stage", {})

    enable_parallel = bool(parallel_cfg.get("enabled", True))
    workers = int(parallel_cfg.get("workers", max(1, (os.cpu_count() or 2) - 1)))
    chunksize = int(parallel_cfg.get("chunksize", 8))
    strict_parallel = bool(parallel_cfg.get("strict", False))

    # Hoist typical-day building outside NSGA-II evaluation loop.
    td_single = build_typical_day_data(cfg_run, ts, count_build=True)
    cache_single = make_cache_context(cfg_run, ts)
    generation_reports: list[dict[str, float]] = []
    pymoo_population_rows: list[dict[str, float]] = []
    pymoo_snapshot_generation = 0
    parallel_fallback_triggered = False
    parallel_failure_reason = ""
    cache_cfg = solver_cfg.get("cache", {})
    adaptive_quant = bool(cache_cfg.get("adaptive_quantize_on_low_hit", True))
    min_hit_rate_target = float(cache_cfg.get("min_hit_rate_target", 0.5))
    low_hit_quant_scale = float(cache_cfg.get("low_hit_quant_scale", 1.25))
    max_auto_quant_updates = int(cache_cfg.get("max_auto_quant_updates", 3))
    auto_quant_updates = 0

    def eval_fn(x: np.ndarray):
        return evaluate_solution(x, cfg_run, ts, td=td_single, cache_ctx=cache_single)

    def generation_callback(gen: int, report: Dict[str, float]) -> None:
        nonlocal auto_quant_updates
        row = {"generation": float(gen)}
        row.update({k: float(v) for k, v in report.items()})
        if (
            (not enable_parallel)
            and adaptive_quant
            and bool(cache_single.get("enabled", False))
            and auto_quant_updates < max_auto_quant_updates
            and float(report.get("cache_hit_rate", 1.0)) < min_hit_rate_target
        ):
            steps = dict(cache_single.get("quant_steps_active", {}))
            for key in list(steps.keys()):
                steps[key] = float(steps[key]) * low_hit_quant_scale
            cache_single["quant_steps_active"] = steps
            cache_single["l2"].clear()
            row["cache_quant_step_scaled"] = float(low_hit_quant_scale)
            auto_quant_updates += 1
        generation_reports.append(row)

    def pymoo_population_callback(gen: int, rows: list[dict[str, float]]) -> None:
        nonlocal pymoo_snapshot_generation
        if not rows:
            return
        for row in rows:
            out = dict(row)
            out["generation"] = float(pymoo_snapshot_generation)
            pymoo_population_rows.append(out)
        pymoo_snapshot_generation += 1

    def _run_outer_engine(
        eval_batch_fn,
        run_cfg: Dict[str, Any],
        run_bounds,
        seed: int,
    ):
        if outer_engine == "custom":
            return run_nsga2(
                eval_fn,
                run_bounds,
                int_idx,
                bin_idx,
                run_cfg,
                seed=seed,
                eval_batch_fn=eval_batch_fn,
                generation_callback=generation_callback,
            )
        if outer_engine == "pymoo":
            return run_pymoo_nsga2(
                eval_fn,
                run_bounds,
                int_idx,
                bin_idx,
                run_cfg,
                seed=seed,
                eval_batch_fn=eval_batch_fn,
                generation_callback=generation_callback,
                population_callback=pymoo_population_callback,
            )
        raise RuntimeError(f"Unsupported solver.outer_engine='{outer_engine}'. Expected 'custom' or 'pymoo'.")

    def solve_with(eval_batch_fn):
        if bool(two_stage_cfg.get("enabled", True)):
            stage1_cfg = dict(solver_cfg)
            stage1_cfg.update(two_stage_cfg.get("stage1", {}))
            stage2_cfg = dict(solver_cfg)
            stage2_cfg.update(two_stage_cfg.get("stage2", {}))

            _, pareto_stage1 = _run_outer_engine(
                eval_batch_fn,
                stage1_cfg,
                bounds,
                cfg_run["project"]["random_seed"],
            )
            bounds_stage2 = _narrow_bounds(
                bounds,
                pareto_stage1,
                int_idx,
                float(two_stage_cfg.get("pad_ratio", 0.35)),
                float(two_stage_cfg.get("abs_pad_ratio", 0.06)),
            )
            _, out_pareto = _run_outer_engine(
                eval_batch_fn,
                stage2_cfg,
                bounds_stage2,
                cfg_run["project"]["random_seed"] + 1,
            )
            return out_pareto

        _, out_pareto = _run_outer_engine(
            eval_batch_fn,
            solver_cfg,
            bounds,
            cfg_run["project"]["random_seed"],
        )
        return out_pareto

    def do_optimize():
        nonlocal parallel_fallback_triggered, parallel_failure_reason, pymoo_snapshot_generation
        pareto = None
        if enable_parallel:
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_worker,
                    initargs=(cfg_run, ts),
                ) as pool:
                    _parallel_preflight(pool)

                    def eval_batch_fn(arrs: list[np.ndarray]):
                        return list(pool.map(_worker_eval, arrs, chunksize=chunksize))

                    pareto = solve_with(eval_batch_fn)
            except Exception as e:
                if outer_engine == "pymoo":
                    pymoo_population_rows.clear()
                    pymoo_snapshot_generation = 0
                msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                parallel_fallback_triggered = True
                parallel_failure_reason = msg
                log_path = ROOT / "results" / "parallel_fallback.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "\n".join(
                        [
                            "Parallel evaluation failed; fallback to single-process.",
                            f"reason={msg}",
                            "",
                            "traceback:",
                            tb,
                        ]
                    ),
                    encoding="utf-8",
                )
                if strict_parallel:
                    raise RuntimeError(
                        "Parallel evaluation failed and solver.parallel.strict=true. "
                        f"reason={msg}. details={log_path}"
                    ) from e
                gurobi_related = any(
                    s in msg.lower()
                    for s in [
                        "gurobi",
                        "gurobipy",
                        "grblicense",
                        "model_init_error",
                        "strict inner solver requires gurobi",
                    ]
                )
                if gurobi_related:
                    raise RuntimeError(msg) from e
                print(
                    "Parallel evaluation unavailable, fallback to single-process. "
                    f"reason={type(e).__name__}: {e}. "
                    f"Details: {log_path}"
                )

        if pareto is None:
            pareto = solve_with(None)
        return pareto

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pareto = _run_with_optional_profiler(
        do_optimize,
        cfg_run,
        generation_reports,
        cache_single if not enable_parallel else None,
    )

    out_rows = []
    for ind in pareto:
        row = {}
        if len(keys) <= 64:
            row.update({k: float(v) for k, v in zip(keys, ind.x)})
        row["decision_vector_json"] = json.dumps(np.asarray(ind.x, dtype=float).tolist())
        row.update(ind.meta)
        row["rank"] = ind.rank
        row["crowding"] = ind.crowding
        out_rows.append(row)

    if not out_rows:
        raise RuntimeError("No Pareto solution found.")

    df = pd.DataFrame(out_rows).sort_values(["tlcc_yuan", "tce_kgco2"]).reset_index(drop=True)
    df.to_csv(results_dir / "pareto_solutions.csv", index=False, encoding="utf-8-sig")

    run_time_s = perf_counter() - t_main
    pareto_eval_stats = _series_stats([_safe_float(v) for v in pd.to_numeric(df.get("eval_total_s", pd.Series(dtype=float)), errors="coerce").tolist()]) if "eval_total_s" in df.columns else _series_stats([])
    pareto_build_stats = _series_stats([_safe_float(v) for v in pd.to_numeric(df.get("inner_build_total_s", pd.Series(dtype=float)), errors="coerce").tolist()]) if "inner_build_total_s" in df.columns else _series_stats([])
    pareto_opt_stats = _series_stats([_safe_float(v) for v in pd.to_numeric(df.get("inner_optimize_s", pd.Series(dtype=float)), errors="coerce").tolist()]) if "inner_optimize_s" in df.columns else _series_stats([])
    gen_eval_stats = _generation_weighted_stats(generation_reports, "eval_total_s")
    gen_build_stats = _generation_weighted_stats(generation_reports, "inner_build_total_s")
    gen_opt_stats = _generation_weighted_stats(generation_reports, "inner_optimize_s")
    runtime_summary = {
        "optimization_runtime_s": float(run_time_s),
        "optimization_runtime_hms": _format_duration_hms(run_time_s),
        "parallel_enabled": bool(enable_parallel),
        "outer_engine": outer_engine,
        "parallel_workers": int(workers),
        "parallel_chunksize": int(chunksize),
        "parallel_fallback_triggered": bool(parallel_fallback_triggered),
        "parallel_failure_reason": parallel_failure_reason,
        "inner_threads": int(cfg_run.get("solver", {}).get("inner", {}).get("threads", 0)),
        "persistent_template_enabled": bool(cfg_run.get("solver", {}).get("inner", {}).get("persistent_template", False)),
        "two_stage_enabled": bool(two_stage_cfg.get("enabled", False)),
        "typical_day_scenarios": int(td_single.n_scenarios),
        "pareto_solution_count": int(len(df)),
        "eval_time_stats_pareto_s": pareto_eval_stats,
        "inner_build_time_stats_pareto_s": pareto_build_stats,
        "inner_optimize_time_stats_pareto_s": pareto_opt_stats,
        "eval_time_stats_all_evals_s": gen_eval_stats,
        "inner_build_time_stats_all_evals_s": gen_build_stats,
        "inner_optimize_time_stats_all_evals_s": gen_opt_stats,
        "warm_start_hit_rate_all_evals": _mean_from_reports(generation_reports, "warm_start_hit_rate"),
        "warm_start_exact_hit_rate_all_evals": _mean_from_reports(generation_reports, "warm_start_exact_hit_rate"),
        "warm_start_neighbor_hit_rate_all_evals": _mean_from_reports(generation_reports, "warm_start_neighbor_hit_rate"),
        "template_reuse_hit_rate_all_evals": _mean_from_reports(generation_reports, "template_reuse_hit_rate"),
        "template_initial_build_rate_all_evals": _mean_from_reports(generation_reports, "template_initial_build_rate"),
        "infeasible_memory_hit_rate_all_evals": _mean_from_reports(generation_reports, "infeasible_memory_hit_rate"),
        "diagnostic_relaxation_rate_all_evals": _mean_from_reports(generation_reports, "diagnostic_relaxation_rate"),
        "diagnostic_repair_rate_all_evals": _mean_from_reports(generation_reports, "diagnostic_repair_rate"),
        "diagnostic_resolve_success_rate_all_evals": _mean_from_reports(generation_reports, "diagnostic_resolve_success_rate"),
        "template_update_time_stats_all_evals_s": _generation_weighted_stats(generation_reports, "template_update_s"),
    }
    convergence_cfg = cfg_run.get("solver", {}).get("convergence", {})
    convergence_enabled = bool(convergence_cfg.get("enabled", True))
    convergence_summary: dict[str, Any] = {
        "enabled": bool(convergence_enabled),
        "available": False,
        "reason": "outer_engine_not_pymoo" if outer_engine != "pymoo" else "not_collected",
    }
    if outer_engine == "pymoo" and convergence_enabled:
        convergence_summary = analyze_pymoo_convergence(pymoo_population_rows, results_dir, cfg_run)
    runtime_summary["pymoo_convergence"] = convergence_summary
    (results_dir / "runtime_summary.json").write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if generation_reports:
        pd.DataFrame(generation_reports).to_csv(
            results_dir / "generation_timing_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

    p_cfg = cfg_run.get("solver", {}).get("profiling", {})
    if bool(p_cfg.get("enabled", False)):
        profile_dir = ROOT / str(p_cfg.get("profile_output_dir", "results/profile"))
        if not profile_dir.is_absolute():
            profile_dir = ROOT / profile_dir
        profile_dir.mkdir(parents=True, exist_ok=True)
        summary_path = profile_dir / "profile_summary.json"
        try:
            summary_obj = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        except Exception:
            summary_obj = {}
        ratio_col = "inner_build_vs_optimize_ratio"
        if ratio_col in df.columns:
            ratio_arr = pd.to_numeric(df[ratio_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            summary_obj["inner_build_vs_optimize_ratio"] = {
                "mean": float(np.mean(ratio_arr)) if ratio_arr.size else 0.0,
                "p50": float(np.percentile(ratio_arr, 50)) if ratio_arr.size else 0.0,
                "p90": float(np.percentile(ratio_arr, 90)) if ratio_arr.size else 0.0,
                "max": float(np.max(ratio_arr)) if ratio_arr.size else 0.0,
            }
        summary_obj["pareto_size"] = int(len(df))
        summary_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    finite_mask = np.isfinite(pd.to_numeric(df["tlcc_yuan"], errors="coerce")) & np.isfinite(pd.to_numeric(df["tce_kgco2"], errors="coerce"))
    if not bool(np.any(finite_mask)):
        print("Warning: all Pareto points are non-finite (inf/nan). Summary will use first-row fallback.")
        min_cost = df.iloc[0]
        min_carbon = df.iloc[0]
        knee = df.iloc[0]
    else:
        dff = df.loc[finite_mask].copy().reset_index(drop=True)
        min_cost = dff.iloc[dff["tlcc_yuan"].idxmin()]
        min_carbon = dff.iloc[dff["tce_kgco2"].idxmin()]
        f1n = (dff["tlcc_yuan"] - dff["tlcc_yuan"].min()) / max(dff["tlcc_yuan"].max() - dff["tlcc_yuan"].min(), 1e-9)
        f2n = (dff["tce_kgco2"] - dff["tce_kgco2"].min()) / max(dff["tce_kgco2"].max() - dff["tce_kgco2"].min(), 1e-9)
        knee_idx = ((f1n**2 + f2n**2) ** 0.5).idxmin()
        knee = dff.iloc[knee_idx]

    summary = pd.DataFrame([min_cost, min_carbon, knee], index=["min_cost", "min_carbon", "knee"])
    summary.to_csv(results_dir / "representative_solutions.csv", encoding="utf-8-sig")

    print("Optimization completed.")
    print(f"Pareto size: {len(df)}")
    print("Saved:")
    print(results_dir / "pareto_solutions.csv")
    print(results_dir / "representative_solutions.csv")


if __name__ == "__main__":
    main()
