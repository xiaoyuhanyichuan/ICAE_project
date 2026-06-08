# Assumptions

This document records implementation assumptions that should be reviewed before
formal paper experiments.

## Fitted Parameters

The following fitted parameters were provided by the researcher and are used as
model inputs:

- RDHX electric coefficient: `0.0265 kWe/kWth`
- Chiller COP: `3.0`
- AC terminal fan coefficient: `0.2137 kWe/kWth`

## Heating Demand Input

`heating_demand_kw` is loaded directly from the heating-demand data and is not
scaled to IT waste heat. In the dispatch model it is treated as an upper bound
on useful heat delivery, so recovered heat can be credited only up to demand.

## Placeholder Inputs

The economic parameters, embodied carbon parameters, default COP values,
equipment weights, and representative zone floor area in `config.json` are
placeholder assumptions. Before formal paper experiments, they must either be
replaced with final study inputs or supported with citations.

## Load-Bearing Constraint

The floor load-bearing constraint is implemented as a hard repair/screening
constraint. A tiny absolute/relative tolerance from `weight.hard_constraint` is
used only to avoid floating-point boundary misclassification; overweight is not
converted into an objective penalty.

The current implemented heating side uses WSHP heat recovery only. The WSHP
source can use recoverable heat from CDU and RDHX/backplate paths. AC and RDHX
remain on the chiller cooling path when their heat is not routed to WSHP.

RDHX is assumed to absorb 100% of IT heat in selected RDHX zones. The RDHX path
therefore no longer includes an auxiliary ventilation or room-cooling branch.

BESS is represented as one integrated capacity variable. Its cost and embodied
carbon are calculated once from that integrated capacity; charge/discharge power
limits are derived by a configured power-to-capacity multiplier instead of a
separate planning capacity.

## Fixed Retrofit Cost and Carbon

Fixed retrofit cost and fixed retrofit carbon represent only construction,
interfaces, commissioning, downtime, and retrofit work. They exclude capacity
equipment CAPEX and capacity equipment embodied carbon, which are calculated
separately from installed capacity. This avoids double-counting capacity
equipment cost or embodied carbon.

For cold-plate/CDU retrofits, the capacity equipment terms are decomposed into
CDU equipment capacity plus cold-plate IT-side retrofit capacity:
`Cap_z^{CDU} * C^{CDU} + Cap_z^{CP} * C^{cold_plate}` for CAPEX, and the same
structure for embodied carbon. `Cap_z^{CP}` is derived from the cold-plate/CDU
capacity in selected cold-plate configurations. The current default cold-plate
values are `2500 yuan/kW_IT` and `6 kgCO2e/kW_IT`.
