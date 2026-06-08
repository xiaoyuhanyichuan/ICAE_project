# Pareto / Dominated 多样性诊断报告

## 1. 样本概况

- 可行解总数：3803
- Pareto 解数：8，占比 0.21%
- Dominated 解数：3795
- Pareto 构型签名数量：8
- Dominated 构型签名数量：3793

## 2. 关键结论

- 构型差异最大的是 `4:cold_plate_cdu_wshp`，Pareto 份额比 dominated 高/低 +8.62%。
- `soft_violation_kg` 的 Pareto 均值为 55,169.2，dominated 均值为 122,514.9，差异为 -67,345.8。
- `operating_cost_yuan_per_year` 的 Pareto 均值为 1.700e+07，dominated 均值为 1.706e+07，差异为 -55,234.7。
- `operating_carbon_kg_per_year` 的 Pareto 均值为 1.093e+07，dominated 均值为 1.195e+07，差异为 -1,021,322.0。

## 3. 构型选择差异

| config_id | config_label | dominated | pareto | pareto_minus_dominated_share |
| --- | --- | --- | --- | --- |
| 4 | cold_plate_cdu_wshp | 0.2055 | 0.2917 | 0.0862 |
| 5 | rdhx_wshp | 0.1983 | 0.2542 | 0.05587 |
| 2 | new_ac_with_ashp | 0.1572 | 0.1167 | -0.04053 |
| 6 | cold_plate_cdu_wshp_plus_air | 0.09987 | 0.0625 | -0.03737 |
| 1 | new_ac | 0.08579 | 0.05 | -0.03579 |
| 3 | new_ac_high_efficiency | 0.09347 | 0.05833 | -0.03514 |
| 0 | existing_ac | 0.1599 | 0.1667 | 0.006755 |

## 4. 容量组合差异

| metric | pareto_mean | dominated_mean | pareto_minus_dominated | relative_diff | standardized_diff |
| --- | --- | --- | --- | --- | --- |
| sum_cap_ac_new_kw | 11,017.9 | 25,797.2 | -14,779.3 | -0.5729 | -2.141 |
| cap_bess | 1,600.5 | 8,814.3 | -7,213.9 | -0.8184 | -0.8411 |
| delta_cap_chiller_kw | 3,451.9 | 6,849.1 | -3,397.2 | -0.496 | -0.7316 |
| cap_tes | 22,926.8 | 17,958.1 | 4,968.7 | 0.2767 | 0.4828 |
| delta_cap_tower_kw | 11,323.6 | 9,617.3 | 1,706.3 | 0.1774 | 0.3019 |

## 5. 承重惩罚、运行成本与运行碳差异

| metric | pareto_mean | dominated_mean | pareto_minus_dominated | relative_diff | standardized_diff |
| --- | --- | --- | --- | --- | --- |
| annualized_embodied_kg_per_year | 232,213.5 | 387,968.9 | -155,755.4 | -0.4015 | -1.862 |
| weight_soft_penalty_cost_yuan_per_year | 2.758e+07 | 6.126e+07 | -3.367e+07 | -0.5497 | -1.791 |
| soft_violation_kg | 55,169.2 | 122,514.9 | -67,345.8 | -0.5497 | -1.791 |
| operating_carbon_kg_per_year | 1.093e+07 | 1.195e+07 | -1,021,322.0 | -0.08546 | -1.245 |
| annualized_capex_yuan_per_year | 8,668,133.4 | 1.084e+07 | -2,173,091.2 | -0.2004 | -1.202 |
| operating_cost_yuan_per_year | 1.700e+07 | 1.706e+07 | -55,234.7 | -0.003238 | -0.03518 |

## 6. 目标与容量/惩罚相关性

| metric | objective | pearson_corr |
| --- | --- | --- |
| soft_violation_kg | tlcc | 0.9966 |
| operating_carbon_kg_per_year | tce | 0.9957 |
| cap_bess | tce | 0.8326 |
| sum_cap_ac_new_kw | tlcc | 0.6614 |
| sum_cap_cdu_kw | tlcc | 0.5105 |
| operating_cost_yuan_per_year | tce | -0.4104 |
| sum_cap_rdhx_kw | tlcc | 0.337 |
| sum_cap_ac_new_kw | tce | 0.3235 |
| sum_cap_rdhx_kw | tce | -0.2906 |
| sum_cap_cdu_kw | tce | -0.2298 |
| delta_cap_chiller_kw | tce | 0.1811 |
| cap_wshp_kw | tlcc | 0.1733 |

## 7. 图件

- scatter: `pareto_vs_dominated_scatter.png`
- capacity: `capacity_distribution.png`
- components: `objective_components.png`
- configs: `config_distribution.png`
- penalty: `weight_penalty_vs_objectives.png`

## 8. 输出文件

- `diversity_enriched_evaluations.csv`：逐个可行解的构型、容量、承重惩罚和目标组成。
- `diversity_group_summary.csv`：Pareto 与 dominated 的分组统计。
- `config_distribution.csv`：构型选择份额。
- `capacity_quantiles.csv`：容量分位数。
- `objective_capacity_correlations.csv`：目标和容量/惩罚的相关性。

## 9. 读数提示

如果 Pareto 与 dominated 在构型和容量上高度重叠，但运行成本/运行碳差异很小，说明前沿点少更可能来自目标尺度或目标相关性；如果 Pareto 构型明显集中在少数签名，说明外层搜索和 repair 后的有效多样性不足；如果 Pareto 主要沿承重惩罚单调变化，说明软承重惩罚正在强烈塑造前沿。
