# pymoo Convergence Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pymoo-style convergence analysis, recommended iteration count, and Pareto solution distribution plots to the current optimization result workflow.

**Architecture:** Keep the existing custom NSGA-II optimizer unchanged. Add a post-processing convergence module that consumes `all_evaluations.csv` / in-memory DataFrames, computes generation-level HV and IGD+ when pymoo is installed, falls back to a local 2D implementation when it is not, then writes metrics, summary JSON, and figures. Reports read those artifacts and include them in `results/report.md`.

**Tech Stack:** Python, pandas, numpy, matplotlib, optional pymoo indicators.

---

### Task 1: Convergence Metrics Module

**Files:**
- Create: `D:\paper\cooling_retrofit_code\convergence_analysis.py`
- Test: `D:\paper\cooling_retrofit_code\tests\test_convergence_analysis.py`

- [ ] Implement generation-level history nondominated fronts from feasible `tlcc` / `tce`.
- [ ] Compute normalized 2D Hypervolume and IGD+ using pymoo if available, with deterministic local fallback.
- [ ] Recommend an iteration count once HV improvement and IGD+ improvement stay below configured thresholds for a patience window.
- [ ] Write `results/convergence/pymoo_convergence_metrics.csv` and `results/convergence/pymoo_convergence_summary.json`.

### Task 2: Plotting

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\plotting.py`
- Test: `D:\paper\cooling_retrofit_code\tests\test_reporting.py`

- [ ] Add `save_convergence_plots(metrics, output_dir)`.
- [ ] Add `save_pareto_distribution_plot(pareto, output_dir, config)`.
- [ ] Keep `save_pareto_plot` compatible with existing tests while adding `pareto_front.png` as the canonical output.

### Task 3: Workflow Integration

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\run_optimization.py`

- [ ] Run convergence analysis after CSV export.
- [ ] Generate convergence and distribution figures when data exists.
- [ ] Return figure paths in the result dictionary.

### Task 4: Report Integration

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\analyze_results.py`

- [ ] Read convergence summary JSON.
- [ ] Add report section for HV, IGD+, final feasible count, recommendation, and artifact paths.
- [ ] Add output index entries for convergence plots and Pareto distribution plot.

### Task 5: Verification

**Files:**
- Run tests under `D:\paper\cooling_retrofit_code\tests`.

- [ ] Run targeted tests for convergence and reporting.
- [ ] Run syntax checks for touched modules.
- [ ] Regenerate report from existing result CSVs and verify new sections appear.
