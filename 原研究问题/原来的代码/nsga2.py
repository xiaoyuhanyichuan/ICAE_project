from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

import numpy as np


@dataclass
class Individual:
    x: np.ndarray
    f1: float
    f2: float
    meta: Dict[str, float]
    rank: int = 0
    crowding: float = 0.0


def _constraint_violation(ind: Individual) -> float:
    v = ind.meta.get("constraint_violation", 0.0)
    try:
        fv = float(v)
    except Exception:
        fv = float("inf")
    if not np.isfinite(fv):
        return float("inf")
    return max(0.0, fv)


def _is_feasible(ind: Individual) -> bool:
    if "is_feasible" in ind.meta:
        try:
            return bool(ind.meta.get("is_feasible"))
        except Exception:
            return False
    return bool(np.isfinite(ind.f1) and np.isfinite(ind.f2) and _constraint_violation(ind) <= 1e-9)


def dominates(a: Individual, b: Individual) -> bool:
    a_feas = _is_feasible(a)
    b_feas = _is_feasible(b)
    if a_feas and (not b_feas):
        return True
    if (not a_feas) and b_feas:
        return False
    if (not a_feas) and (not b_feas):
        va = _constraint_violation(a)
        vb = _constraint_violation(b)
        return va < vb - 1e-12
    return (a.f1 <= b.f1 and a.f2 <= b.f2) and (a.f1 < b.f1 or a.f2 < b.f2)


def fast_non_dominated_sort(pop: List[Individual]) -> List[List[int]]:
    n = len(pop)
    s = [set() for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                s[p].add(q)
            elif dominates(pop[q], pop[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in s[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    pop[q].rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(pop: List[Individual], front: List[int]) -> None:
    if not front:
        return
    for i in front:
        pop[i].crowding = 0.0
    if len(front) <= 2:
        for i in front:
            pop[i].crowding = float("inf")
        return

    for key in ["f1", "f2"]:
        front_sorted = sorted(front, key=lambda idx: getattr(pop[idx], key))
        pop[front_sorted[0]].crowding = float("inf")
        pop[front_sorted[-1]].crowding = float("inf")

        vmin = getattr(pop[front_sorted[0]], key)
        vmax = getattr(pop[front_sorted[-1]], key)
        if abs(vmax - vmin) < 1e-12:
            continue

        for k in range(1, len(front_sorted) - 1):
            prev_v = getattr(pop[front_sorted[k - 1]], key)
            next_v = getattr(pop[front_sorted[k + 1]], key)
            pop[front_sorted[k]].crowding += (next_v - prev_v) / (vmax - vmin)


def tournament_select(pop: List[Individual], rng: np.random.Generator) -> Individual:
    i, j = rng.integers(0, len(pop), size=2)
    a, b = pop[i], pop[j]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding >= b.crowding else b


def sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    eta_c: float,
    prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    c1 = p1.copy()
    c2 = p2.copy()
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1 = min(p1[i], p2[i])
        x2 = max(p1[i], p2[i])
        lower, upper = lb[i], ub[i]

        rand = rng.random()
        beta = 1.0 + 2.0 * (x1 - lower) / max(x2 - x1, 1e-12)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + 2.0 * (upper - x2) / max(x2 - x1, 1e-12)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        c1[i] = np.clip(child1, lower, upper)
        c2[i] = np.clip(child2, lower, upper)

    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    eta_m: float,
    prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() > prob:
            continue
        yl, yu = lb[i], ub[i]
        if yl == yu:
            continue
        delta1 = (y[i] - yl) / (yu - yl)
        delta2 = (yu - y[i]) / (yu - yl)
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - val ** mut_pow

        y[i] = np.clip(y[i] + deltaq * (yu - yl), yl, yu)
    return y


def run_nsga2(
    eval_fn: Callable[[np.ndarray], Tuple[float, float, Dict[str, float]]],
    bounds: List[Tuple[float, float]],
    int_idx: List[int],
    bin_idx: List[int],
    cfg: Dict[str, Any],
    seed: int = 42,
    eval_batch_fn: Callable[[List[np.ndarray]], List[Tuple[float, float, Dict[str, float]]]] | None = None,
    generation_callback: Callable[[int, Dict[str, float]], None] | None = None,
) -> Tuple[List[Individual], List[Individual]]:
    pop_size = int(cfg["population_size"])
    generations = int(cfg["generations"])
    pc = float(cfg["crossover_prob"])
    pm = float(cfg["mutation_prob"])
    eta_c = float(cfg["eta_c"])
    eta_m = float(cfg["eta_m"])
    pm_bin = float(cfg.get("mutation_prob_binary", pm))

    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    def decode(x: np.ndarray) -> np.ndarray:
        y = np.clip(x, lb, ub)
        for idx in int_idx:
            y[idx] = np.round(y[idx])
        return y

    bin_set = set(bin_idx)
    cont_idx = [i for i in range(len(bounds)) if i not in bin_set]

    def make_inds(xs: List[np.ndarray]) -> Tuple[List[Individual], Dict[str, float]]:
        decoded = [decode(x) for x in xs]
        if eval_batch_fn is not None:
            out = eval_batch_fn(decoded)
        else:
            out = [eval_fn(xx) for xx in decoded]
        inds: List[Individual] = []
        cache_hits = 0
        cache_l1_hits = 0
        warm_start_hits = 0
        warm_start_exact_hits = 0
        warm_start_neighbor_hits = 0
        feasible_count = 0
        violation_sum = 0.0
        template_reuse_hits = 0
        template_initial_builds = 0
        infeasible_memory_hits = 0
        diagnostic_relaxations = 0
        diagnostic_repairs = 0
        diagnostic_resolve_successes = 0
        template_update_vals: List[float] = []
        eval_total_vals: List[float] = []
        build_total_vals: List[float] = []
        optimize_vals: List[float] = []
        for xx, (f1, f2, meta) in zip(decoded, out):
            ind = Individual(xx, f1, f2, meta)
            inds.append(ind)
            if bool(meta.get("cache_hit", False)):
                cache_hits += 1
            if bool(meta.get("cache_l1_hit", False)):
                cache_l1_hits += 1
            if bool(meta.get("warm_start_used", False)):
                warm_start_hits += 1
                source = str(meta.get("warm_start_source", ""))
                if source == "exact":
                    warm_start_exact_hits += 1
                if source == "neighbor":
                    warm_start_neighbor_hits += 1
            if bool(meta.get("template_reuse_hit", False)):
                template_reuse_hits += 1
            if str(meta.get("build_mode", "")) == "template_initial_build":
                template_initial_builds += 1
            if bool(meta.get("infeasible_memory_hit", False)):
                infeasible_memory_hits += 1
            if bool(meta.get("diagnostic_relaxation_used", False)):
                diagnostic_relaxations += 1
            if bool(meta.get("diagnostic_repair_applied", False)):
                diagnostic_repairs += 1
            if bool(meta.get("diagnostic_resolve_success", False)):
                diagnostic_resolve_successes += 1
            if _is_feasible(ind):
                feasible_count += 1
            violation_sum += _constraint_violation(ind)
            try:
                eval_total_vals.append(float(meta.get("eval_total_s", np.nan)))
            except Exception:
                pass
            try:
                build_total_vals.append(float(meta.get("inner_build_total_s", np.nan)))
            except Exception:
                pass
            try:
                optimize_vals.append(float(meta.get("inner_optimize_s", np.nan)))
            except Exception:
                pass
            try:
                template_update_vals.append(float(meta.get("template_update_s", np.nan)))
            except Exception:
                pass
        n = max(1, len(inds))
        eval_arr = np.asarray(eval_total_vals, dtype=float)
        build_arr = np.asarray(build_total_vals, dtype=float)
        optimize_arr = np.asarray(optimize_vals, dtype=float)
        template_update_arr = np.asarray(template_update_vals, dtype=float)
        eval_arr = eval_arr[np.isfinite(eval_arr)]
        build_arr = build_arr[np.isfinite(build_arr)]
        optimize_arr = optimize_arr[np.isfinite(optimize_arr)]
        template_update_arr = template_update_arr[np.isfinite(template_update_arr)]
        report = {
            "eval_count": float(len(inds)),
            "cache_hit_rate": float(cache_hits) / float(n),
            "cache_l1_hit_rate": float(cache_l1_hits) / float(n),
            "warm_start_hit_rate": float(warm_start_hits) / float(n),
            "warm_start_exact_hit_rate": float(warm_start_exact_hits) / float(n),
            "warm_start_neighbor_hit_rate": float(warm_start_neighbor_hits) / float(n),
            "template_reuse_hit_rate": float(template_reuse_hits) / float(n),
            "template_initial_build_rate": float(template_initial_builds) / float(n),
            "infeasible_memory_hit_rate": float(infeasible_memory_hits) / float(n),
            "diagnostic_relaxation_rate": float(diagnostic_relaxations) / float(n),
            "diagnostic_repair_rate": float(diagnostic_repairs) / float(n),
            "diagnostic_resolve_success_rate": float(diagnostic_resolve_successes) / float(n),
            "feasible_rate": float(feasible_count) / float(n),
            "mean_constraint_violation": float(violation_sum) / float(n),
            "eval_total_s_min": float(np.min(eval_arr)) if eval_arr.size else float("nan"),
            "eval_total_s_mean": float(np.mean(eval_arr)) if eval_arr.size else float("nan"),
            "eval_total_s_max": float(np.max(eval_arr)) if eval_arr.size else float("nan"),
            "inner_build_total_s_min": float(np.min(build_arr)) if build_arr.size else float("nan"),
            "inner_build_total_s_mean": float(np.mean(build_arr)) if build_arr.size else float("nan"),
            "inner_build_total_s_max": float(np.max(build_arr)) if build_arr.size else float("nan"),
            "inner_optimize_s_min": float(np.min(optimize_arr)) if optimize_arr.size else float("nan"),
            "inner_optimize_s_mean": float(np.mean(optimize_arr)) if optimize_arr.size else float("nan"),
            "inner_optimize_s_max": float(np.max(optimize_arr)) if optimize_arr.size else float("nan"),
            "template_update_s_min": float(np.min(template_update_arr)) if template_update_arr.size else float("nan"),
            "template_update_s_mean": float(np.mean(template_update_arr)) if template_update_arr.size else float("nan"),
            "template_update_s_max": float(np.max(template_update_arr)) if template_update_arr.size else float("nan"),
        }
        return inds, report

    pop_x = [lb + rng.random(len(bounds)) * (ub - lb) for _ in range(pop_size)]
    pop, init_report = make_inds(pop_x)
    if generation_callback is not None:
        generation_callback(-1, init_report)

    for gen in range(generations):
        fronts = fast_non_dominated_sort(pop)
        for front in fronts:
            crowding_distance(pop, front)

        offspring_x: List[np.ndarray] = []
        while len(offspring_x) < pop_size:
            p1 = tournament_select(pop, rng)
            p2 = tournament_select(pop, rng)
            c1, c2 = sbx_crossover(p1.x, p2.x, lb, ub, eta_c, pc, rng)
            # Binary genes: uniform crossover + bit-flip mutation
            for b in bin_idx:
                c1[b] = p1.x[b] if rng.random() < 0.5 else p2.x[b]
                c2[b] = p1.x[b] if rng.random() < 0.5 else p2.x[b]
                if rng.random() < pm_bin:
                    c1[b] = 1.0 - np.round(c1[b])
                if rng.random() < pm_bin:
                    c2[b] = 1.0 - np.round(c2[b])

            # Continuous / integer genes still use polynomial mutation
            if cont_idx:
                c1c = polynomial_mutation(c1[cont_idx], lb[cont_idx], ub[cont_idx], eta_m, pm, rng)
                c2c = polynomial_mutation(c2[cont_idx], lb[cont_idx], ub[cont_idx], eta_m, pm, rng)
                c1[cont_idx] = c1c
                c2[cont_idx] = c2c
            offspring_x.append(c1)
            if len(offspring_x) < pop_size:
                offspring_x.append(c2)
        offspring, offspring_report = make_inds(offspring_x)
        if generation_callback is not None:
            generation_callback(gen, offspring_report)

        combined = pop + offspring
        fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            crowding_distance(combined, front)

        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(combined[i] for i in front)
            else:
                sorted_front = sorted(front, key=lambda i: combined[i].crowding, reverse=True)
                remain = pop_size - len(new_pop)
                new_pop.extend(combined[i] for i in sorted_front[:remain])
                break

        pop = new_pop

    final_fronts = fast_non_dominated_sort(pop)
    for front in final_fronts:
        crowding_distance(pop, front)

    pareto = [pop[i] for i in final_fronts[0]] if final_fronts else []
    return pop, pareto
