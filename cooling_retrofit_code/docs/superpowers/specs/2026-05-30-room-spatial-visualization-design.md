# Room Spatial Visualization Design

## Context

The project already produces Pareto solutions for the cooling retrofit planning model. The new reporting need is to make the selected retrofit plan spatially interpretable: every rack should be shown at its room coordinate, each rack group should show the selected retrofit configuration, and the report should explain how business load growth and old AC capacity aging relate to configuration choices.

The modeling document defines the relevant inputs:

- Rack business load growth intensity: `rho_i_load`, corresponding to $\rho_i^{load}$ in Section 5.1.
- Old AC capacity aging rate: `alpha_z_age`, corresponding to $\alpha_z^{age}$ in Section 5.2.
- The relationship analysis uses `alpha_z_age` as the primary AC aging metric.

The default visualization target is the Pareto knee solution, selected from the feasible Pareto front by normalized distance to the ideal TLCC/TCE point.

## Scope

This feature adds static reporting artifacts only. It does not change the MILP, NSGA-II search, repair, screening, or objective functions in this implementation step.

Included outputs:

- `results/figures/room_layout_knee_solution.png`
- `results/figures/config_choice_vs_growth_age.png`
- `results/spatial/knee_solution_rack_layout.csv`
- `results/spatial/config_choice_relationship.csv`
- A short spatial-analysis section in the generated result report.

Excluded for now:

- Browser-based interactive visualization.
- Generating room layout plots for every Pareto solution.
- Re-estimating `rho_i_load` or `alpha_z_age` from raw operating data. If these values are missing, the code may use documented fallback values from configuration, but the report must make that clear.

## Data Sources

The spatial layout uses `rack_metadata.csv`:

- `rack_id`
- `row`
- `col`
- `ac_unit`
- `hotspot_risk`

The selected retrofit configuration uses the Pareto result table:

- `results/pareto_solutions.csv`
- `config_by_zone_json`
- `tlcc`
- `tce`
- `solution_id`

Business and aging parameters come from configuration:

- Rack-level or group-level `rho_i_load` values.
- AC-unit-level `alpha_z_age` values.

If rack-level `rho_i_load` is not provided, the implementation should allow an AC-unit-level fallback mapped to racks through `rack_metadata.ac_unit`. If `alpha_z_age` is missing for an AC group, the fallback should be explicit in configuration and reported as a fallback, not as measured data.

## Visualization Design

### Room Layout

The room layout plot is a static scatter plot similar to the user-provided reference image.

Each rack is drawn at:

- x-axis: `col`
- y-axis: `row`

Encoding:

- Marker fill color: selected retrofit configuration of the rack's `ac_unit`.
- Marker text: compact rack number parsed from `rack_id`, such as `0`, `1`, `48`.
- Marker outline or small adjacent label: AC group, such as `cdz1`.
- Optional thin red outline: racks with high `hotspot_risk`, if the metadata contains a meaningful high-risk flag.

The title includes:

- `solution_id`
- `TLCC`
- `TCE`
- A note that the displayed solution is the Pareto knee solution.

The legend maps configuration IDs to configuration names from `config.json`.

### Growth-Aging Relationship

The relationship plot is a zone-level scatter plot.

Each point represents one AC group `z`.

Encoding:

- x-axis: zone average business growth intensity $\bar\rho_z^{load}$.
- y-axis: old AC capacity aging rate $\alpha_z^{age}$.
- Marker fill color: selected retrofit configuration for the zone in the Pareto knee solution.
- Marker size: zone peak IT load.
- Marker label: AC group ID, such as `cdz1`.

This chart is descriptive. The report should avoid claiming causality because the number of AC groups is small and the selected solution is an optimization outcome shaped by multiple constraints and objectives.

## Analysis Tables

`knee_solution_rack_layout.csv` contains one row per rack:

- `solution_id`
- `rack_id`
- `row`
- `col`
- `ac_unit`
- `config_id`
- `config_name`
- `rho_load`
- `alpha_age`
- `hotspot_risk`

`config_choice_relationship.csv` contains one row per AC group:

- `solution_id`
- `ac_unit`
- `config_id`
- `config_name`
- `rack_count`
- `rho_load_mean`
- `alpha_age`
- `zone_peak_it_kw`
- `zone_mean_hotspot_risk`
- `zone_max_hotspot_risk`

If feasible, the table may also include Pareto-front selection frequencies per AC group as supplemental columns, but the main report text should focus on the knee solution.

## Code Organization

Add a small, focused module for this feature, for example `spatial_analysis.py`.

Responsibilities:

- Select the Pareto knee solution from the feasible front.
- Build rack-level and zone-level analysis frames.
- Save the two static figures.
- Return artifact paths and summary rows to reporting.

Integrate the module into the existing result-generation path after Pareto results are written and before the Markdown report is finalized.

Keep plotting helpers local to the module unless a function is generally useful for existing plots. This avoids making `plotting.py` too broad.

## Error Handling

If required spatial columns are missing, skip the room layout plot and add a report warning.

If Pareto results are missing or no feasible Pareto solution exists, skip both plots and add a report warning.

If `rho_i_load` or `alpha_z_age` is missing, use only documented configuration fallbacks. The generated CSV should still include the values actually used, and the report should state that fallback values were used.

## Testing

Add focused tests without generating many persistent files:

- Build a tiny rack metadata frame and Pareto frame, then verify rack-to-config mapping.
- Verify knee solution selection.
- Verify zone-level aggregation of `rho_load_mean`, `alpha_age`, and peak IT load.
- Verify plotting functions write only the expected two image files into a temporary directory.

Do not add broad end-to-end test artifacts for every Pareto solution.

## Acceptance Criteria

- A normal optimization/reporting run creates the two expected figures and two expected CSV files.
- The room layout figure labels all racks by spatial position and selected configuration.
- The relationship figure shows $\bar\rho_z^{load}$ against $\alpha_z^{age}$, colored by selected configuration.
- The generated report includes the two figures and a short descriptive interpretation.
- Missing parameter fallbacks are explicit in CSV/report output.
