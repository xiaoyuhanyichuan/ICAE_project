# Assumptions

This document records implementation assumptions that should be reviewed before
formal paper experiments.

## Fitted Parameters

The following fitted parameters were provided by the researcher and are used as
model inputs:

- RDHX electric coefficient: `0.0265 kWe/kWth`
- Chiller COP: `3.0`
- AC terminal fan coefficient: `0.2137 kWe/kWth`

## Heating Demand Scaling

The raw building heating demand peak is about `8 MW`, while the available IT
waste heat peak is below `1 MW`. For the current waste-heat recovery scenario,
`heating_demand_kw` is therefore scaled at data-loading time so its peak equals
`0.9 * peak(it_load_kw)`. The original values are retained in
`heating_demand_raw_kw`, and the applied multiplier is retained in
`heating_demand_scale_factor`.

## Placeholder Inputs

The economic parameters, embodied carbon parameters, default COP values,
equipment weights, and representative zone floor area in `config.json` are
placeholder assumptions. Before formal paper experiments, they must either be
replaced with final study inputs or supported with citations.

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
