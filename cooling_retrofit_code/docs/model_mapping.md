# Model Mapping

This document maps the paper-level planning notation to the current runnable
code symbols. It intentionally reflects the implemented cooling retrofit
planning model only.

| Model notation | Code symbol | Notes |
| --- | --- | --- |
| `s_z` | `PlanningDecision.s_z` | Zone-level retrofit option selected by the outer decision. |
| `u_{z,k}` | derived from `s_z` | Binary option indicators are derived from the selected zone option. |
| `Cap_z^{AC,new}` | `cap_ac_new[z]` | New AC capacity by zone. |
| `Cap_z^{RDHX}` | `cap_rdhx[z]` | RDHX capacity by zone. |
| `Cap_z^{CDU}` | `cap_cdu[z]` | CDU capacity by zone. |
| `Cap^{BESS}` | `cap_bess` | Integrated BESS capacity; cost and embodied carbon are not split into power and energy parts. |
| WSHP capacity | `cap_wshp` | System-level WSHP heat recovery capacity. |
| Added chiller capacity | `delta_cap_chiller` | Incremental chiller capacity above existing capacity. |
| Added tower capacity | `delta_cap_tower` | Incremental tower capacity above existing capacity. |
| `P_t^{Chiller}` | `p_chiller_kw` | Chiller electric power in kW. |
| `Q_t^{Tower}` | implicit tower heat expression | Chiller condenser heat plus CDU direct rejection. |
| `s_{z,t}^{air}` | `q_air_kw` | Air-side cooling supplied in kW thermal. |
| `TLCC` | `EvaluationResult.tlcc` | Total life-cycle cost objective value. |
| `TCE` | `EvaluationResult.tce` | Total carbon emissions objective value. |

## Heat Flow Model

The current dispatch uses the following core heat-flow relationships:

- `q_rdhx = q_rdhx_to_wshp + q_rdhx_to_chiller`
- RDHX is modeled as absorbing 100% of the IT heat in its selected zone; no auxiliary ventilation/cooling branch is modeled for RDHX zones.
- `q_chiller_evap = q_air + q_rdhx_to_chiller`
- `q_cdu = q_cdu_to_wshp + q_cdu_to_free`
- `q_cdu_to_wshp + q_rdhx_to_wshp` provides the WSHP source heat requirement.
- Tower heat is `q_chiller_evap + p_chiller + q_cdu_to_free`.
- `q_heat` is WSHP delivered heat. It is not direct waste-heat heating.

Dispatch result columns include:

| Code symbol | Meaning |
| --- | --- |
| `q_chiller_evap_kw` | Chiller evaporator cooling output in kW thermal. |
| `q_cdu_to_wshp_kw` | CDU heat routed to WSHP source side. |
| `q_cdu_to_free_kw` | CDU heat routed to direct/free rejection. |
| `q_rdhx_to_wshp_kw` | RDHX/backplate heat routed to WSHP source side. |
| `q_rdhx_to_chiller_kw` | RDHX/backplate heat routed to the chiller cooling path. |
| `p_tower_kw` | Cooling tower electric power. |
| `p_wshp_kw` | WSHP electric power. |
| `p_grid_kw` | Total grid electric power used by the dispatch. |
| `q_heat_kw` | WSHP delivered heat. |

AC, RDHX, and CDU electric power are not separate dispatch result columns in
the current implementation. They are accounted for inside `p_grid_kw` by
multiplying the relevant heat flows by their configured electric coefficients.
