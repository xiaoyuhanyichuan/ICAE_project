# 优化结果报告

## 1. 求解概览

- 总评估方案数：2952
- 可行方案数：2952
- 可行率：100.00%
- Pareto 方案数：10
- 代数范围：0 - 40
- Pareto 年化成本范围：31,855,336.28 yuan/year - 33,675,137.32 yuan/year
- Pareto 年化碳排范围：8,520,614.64 kgCO2/year - 8,804,042.27 kgCO2/year

## 2. 模型与求解口径

- 方法论继承原研究问题：外层 NSGA-II 规划搜索，repair/screening 预筛查，内层 Gurobi MILP 运行调度，最后从全部历史可行解筛选 Pareto 前沿。
- 当前代码口径已同步：RDHX 链路不再设置辅助风冷容量；BESS 不再拆分功率容量和能量容量；供热输出以原始供热需求为可消纳上限，不再设置未供热松弛罚金。
- RDHX 按可吸收其负责区域 100% IT 热量建模；未进入 WSHP 的 RDHX 热量进入 Chiller。
- BESS 仅使用整体容量 `cap_bess`，充放电功率上限由 `technology.bess.c_rate_per_hour` 派生。
- 典型日：K=8；append_peak_day=True。
- NSGA-II：population_size=72；generations=40；mutation_probability=0.15。

## 2.1 求解加速统计

- build mode 分布：template_reuse: 2952
- parallel fallback rate: 0.00%
- parallel workers: 2
- parallel chunksize: 1
- Gurobi NodefileStart GB: 0.5
- Gurobi MIPFocus: 1
- Gurobi NumericFocus: 1
- Gurobi OutputFlag: 0
- 平均 fresh 建模/模板更新总耗时：0.097 s
- 平均模板更新时间：0.096 s
- 平均 Gurobi 求解时间：0.143 s
- 平均模板首次构建耗时：0.000 s
- warm start 命中率：0.03%
- warm start 实际写入率：0.03%；平均写入变量数：11.8
- warm start exact/neighbor/miss: exact=1, neighbor=0, miss=1480
- template reuse 命中率：99.93%
- persistent template backend：gurobi_template: 2952

## 3. 收敛性分析

- 指标后端：pymoo；参考前沿：history_nondominated
- 最终代数：40；历史可行解：2952；最终前沿点数：10
- 收敛性分析观测代数：41
- 最终 Hypervolume：1.20
- 最终 IGD+：0.00
- 推荐判断：未检测到稳定平台期。

## 4. 模型-代码同步提示

- 未检测到 RDHX 辅助风冷、BESS 双容量或供热松弛罚金字段。

## 5. Pareto 与代表方案

### 最低成本解
- solution_id：2440
- generation：33
- TLCC：31,855,336.28 yuan/year
- TCE：8,804,042.27 kgCO2/year
- 区域构型计数：1 new_ac: 1；2 new_ac_with_ashp: 1；3 new_ac_high_efficiency: 2；4 cold_plate_cdu_wshp: 12；5 rdhx_wshp: 3；6 cold_plate_cdu_wshp_plus_air: 11
- 容量配置：
  - 新空调容量: 2,534.81 kW
  - RDHX 容量: 159.33 kW
  - CDU 容量: 3,282.33 kW
  - ASHP 容量: 1,450.34 kW
  - WSHP 容量: 121.16 kW
  - BESS 整体容量: 6,198.64 kWh
  - TES 容量: 5,091.70 kWh
  - Chiller 扩容: 1,837.34 kW
  - Cooling Tower 扩容: 225.81 kW

### 最低碳排解
- solution_id：2574
- generation：35
- TLCC：33,675,137.32 yuan/year
- TCE：8,520,614.64 kgCO2/year
- 区域构型计数：2 new_ac_with_ashp: 1；4 cold_plate_cdu_wshp: 9；5 rdhx_wshp: 4；6 cold_plate_cdu_wshp_plus_air: 16
- 容量配置：
  - 新空调容量: 3,289.28 kW
  - RDHX 容量: 531.93 kW
  - CDU 容量: 3,158.88 kW
  - ASHP 容量: 2,131.36 kW
  - WSHP 容量: 3,116.27 kW
  - BESS 整体容量: 6,198.64 kWh
  - TES 容量: 24,896.19 kWh
  - Chiller 扩容: 686.90 kW
  - Cooling Tower 扩容: 19,815.38 kW

### 折中解
- solution_id：2536
- generation：35
- TLCC：32,797,315.60 yuan/year
- TCE：8,634,455.82 kgCO2/year
- 区域构型计数：2 new_ac_with_ashp: 2；3 new_ac_high_efficiency: 1；4 cold_plate_cdu_wshp: 9；5 rdhx_wshp: 5；6 cold_plate_cdu_wshp_plus_air: 13
- 容量配置：
  - 新空调容量: 3,311.23 kW
  - RDHX 容量: 603.68 kW
  - CDU 容量: 3,012.22 kW
  - ASHP 容量: 2,131.36 kW
  - WSHP 容量: 1,417.24 kW
  - BESS 整体容量: 6,198.64 kWh
  - TES 容量: 17,533.12 kWh
  - Chiller 扩容: 686.90 kW
  - Cooling Tower 扩容: 2,703.71 kW

## 6. 可行性、repair 与不可行原因

- 历史可行解数量：2952
- 历史非支配筛选后 Pareto 解数量：10
- repair 后向量与原始向量不同的候选解数量：2952
- 本轮未记录不可行原因，或所有候选解均通过筛查。

## 7. 机房空间可视化

- 机柜空间布局图：5 张独立机房图
- 构型选择关系图：`D:\paper\cooling_retrofit_code\results\figures\config_choice_vs_growth_age.png`
- 机柜级空间数据：`D:\paper\cooling_retrofit_code\results\spatial\knee_solution_rack_layout.csv`
- 空调组级关系数据：`D:\paper\cooling_retrofit_code\results\spatial\config_choice_relationship.csv`
![room_layout_knee_solution_room01](D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room01.png)
![room_layout_knee_solution_room02](D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room02.png)
![room_layout_knee_solution_room03](D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room03.png)
![room_layout_knee_solution_room04](D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room04.png)
![room_layout_knee_solution_room05](D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room05.png)
![Configuration choice vs growth and aging](D:\paper\cooling_retrofit_code\results\figures\config_choice_vs_growth_age.png)
- room01_cdz1：构型=rdhx_wshp；平均业务爬升率=16.56%；空调老化率=2.93%；区域峰值 IT 负荷=292.06 kW
- room01_cdz2：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=15.28%；空调老化率=2.51%；区域峰值 IT 负荷=195.99 kW
- room01_cdz3：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=13.04%；空调老化率=2.90%；区域峰值 IT 负荷=58.95 kW
- room01_cdz4：构型=rdhx_wshp；平均业务爬升率=12.76%；空调老化率=2.37%；区域峰值 IT 负荷=115.61 kW
- room01_cdz5：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=15.17%；空调老化率=1.24%；区域峰值 IT 负荷=330.32 kW
- room01_cdz6：构型=cold_plate_cdu_wshp；平均业务爬升率=22.69%；空调老化率=2.01%；区域峰值 IT 负荷=21.58 kW
- room02_cdz1：构型=cold_plate_cdu_wshp；平均业务爬升率=12.68%；空调老化率=1.61%；区域峰值 IT 负荷=289.87 kW
- room02_cdz2：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=17.79%；空调老化率=1.57%；区域峰值 IT 负荷=196.91 kW
- room02_cdz3：构型=cold_plate_cdu_wshp；平均业务爬升率=14.25%；空调老化率=1.54%；区域峰值 IT 负荷=59.11 kW
- room02_cdz4：构型=cold_plate_cdu_wshp；平均业务爬升率=17.77%；空调老化率=2.47%；区域峰值 IT 负荷=116.71 kW
- room02_cdz5：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=13.25%；空调老化率=1.71%；区域峰值 IT 负荷=329.27 kW
- room02_cdz6：构型=new_ac_high_efficiency；平均业务爬升率=19.37%；空调老化率=2.34%；区域峰值 IT 负荷=21.45 kW
- room03_cdz1：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=14.55%；空调老化率=2.59%；区域峰值 IT 负荷=290.92 kW
- room03_cdz2：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=14.19%；空调老化率=2.70%；区域峰值 IT 负荷=195.56 kW
- room03_cdz3：构型=cold_plate_cdu_wshp；平均业务爬升率=15.13%；空调老化率=1.59%；区域峰值 IT 负荷=59.12 kW
- room03_cdz4：构型=rdhx_wshp；平均业务爬升率=12.64%；空调老化率=1.09%；区域峰值 IT 负荷=115.55 kW
- room03_cdz5：构型=cold_plate_cdu_wshp；平均业务爬升率=12.31%；空调老化率=2.65%；区域峰值 IT 负荷=328.60 kW
- room03_cdz6：构型=cold_plate_cdu_wshp；平均业务爬升率=12.60%；空调老化率=1.54%；区域峰值 IT 负荷=21.17 kW
- room04_cdz1：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=15.54%；空调老化率=2.38%；区域峰值 IT 负荷=291.36 kW
- room04_cdz2：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=13.30%；空调老化率=2.11%；区域峰值 IT 负荷=195.33 kW
- room04_cdz3：构型=cold_plate_cdu_wshp；平均业务爬升率=11.84%；空调老化率=1.49%；区域峰值 IT 负荷=58.81 kW
- room04_cdz4：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=17.38%；空调老化率=1.63%；区域峰值 IT 负荷=116.65 kW
- room04_cdz5：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=14.56%；空调老化率=1.59%；区域峰值 IT 负荷=330.08 kW
- room04_cdz6：构型=rdhx_wshp；平均业务爬升率=20.93%；空调老化率=1.01%；区域峰值 IT 负荷=21.51 kW
- room05_cdz1：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=15.25%；空调老化率=2.10%；区域峰值 IT 负荷=291.18 kW
- room05_cdz2：构型=cold_plate_cdu_wshp；平均业务爬升率=15.03%；空调老化率=1.43%；区域峰值 IT 负荷=195.74 kW
- room05_cdz3：构型=rdhx_wshp；平均业务爬升率=13.18%；空调老化率=2.56%；区域峰值 IT 负荷=58.96 kW
- room05_cdz4：构型=new_ac_with_ashp；平均业务爬升率=16.12%；空调老化率=2.61%；区域峰值 IT 负荷=116.31 kW
- room05_cdz5：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=14.01%；空调老化率=1.16%；区域峰值 IT 负荷=329.93 kW
- room05_cdz6：构型=new_ac_with_ashp；平均业务爬升率=8.80%；空调老化率=1.85%；区域峰值 IT 负荷=21.01 kW

## 8. 三方案对比实验

- 三方案对比数据表：`D:\paper\cooling_retrofit_code\results\benchmark_comparison.csv`
- 三方案对比图：`D:\paper\cooling_retrofit_code\results\figures\benchmark_comparison.png`
![三方案对比图](D:\paper\cooling_retrofit_code\results\figures\benchmark_comparison.png)

| 方案 | 可行性 | TLCC (yuan/year) | TCE (kgCO2/year) | 运行成本 (yuan/year) | 运营碳 (kgCO2/year) | 余热回收率 | PUE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline 1：不改造（既有冗余） | True | 38,093,221.75 | 12,073,027.21 | 38,093,221.75 | 12,073,027.21 | 0.00% | 1.57 |
| Baseline 2：整体激进改造 | True | 36,032,779.22 | 8,868,473.61 | 28,930,203.45 | 8,766,091.75 | 12.43% | 1.22 |
| 本文方案：精细化改造 | True | 32,797,315.60 | 8,634,455.82 | 28,470,973.92 | 8,533,291.57 | 20.15% | 1.25 |

| 对比对象 | TLCC | TCE | 运行成本 | 运营碳 | 余热回收率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本文方案相对 Baseline 1（不改造/既有冗余） | 降低 13.90% | 降低 28.48% | 降低 25.26% | 降低 29.32% | 提高 20.15 个百分点 |
| 本文方案相对 Baseline 2（整体激进改造） | 降低 8.98% | 降低 2.64% | 降低 1.59% | 降低 2.66% | 提高 7.72 个百分点 |

- 投资追平年限均以 Baseline 1 为参照；固定成本增量和隐含碳增量由年化 TLCC/TCE 结果层扣除年度运行成本/运营碳后估算。
- baseline_1_no_retrofit (Baseline 1：不改造（既有冗余）): 可行=True, TLCC=38,093,221.75 yuan/year, TCE=12,073,027.21 kgCO2/year, PUE=1.57, 运行成本=38,093,221.75 yuan/year, 运营碳=12,073,027.21 kgCO2/year, 余热回收率=0.00%, 相对 B1 运行成本节省=0.00 yuan/year, 相对 B1 运营碳节省=0.00 kgCO2/year, 固定成本追平年限=0.00 年, 隐含碳追平年限=0.00 年
- baseline_2_aggressive_retrofit (Baseline 2：整体激进改造): 可行=True, TLCC=36,032,779.22 yuan/year, TCE=8,868,473.61 kgCO2/year, PUE=1.22, 运行成本=28,930,203.45 yuan/year, 运营碳=8,766,091.75 kgCO2/year, 余热回收率=12.43%, 相对 B1 运行成本节省=9,163,018.30 yuan/year, 相对 B1 运营碳节省=3,306,935.46 kgCO2/year, 固定成本追平年限=0.78 年, 隐含碳追平年限=0.03 年
- proposed_refined_knee (本文方案：精细化改造): 可行=True, TLCC=32,797,315.60 yuan/year, TCE=8,634,455.82 kgCO2/year, PUE=1.25, 运行成本=28,470,973.92 yuan/year, 运营碳=8,533,291.57 kgCO2/year, 余热回收率=20.15%, 相对 B1 运行成本节省=9,622,247.83 yuan/year, 相对 B1 运营碳节省=3,539,735.64 kgCO2/year, 固定成本追平年限=0.45 年, 隐含碳追平年限=0.03 年
- ![image-20260605150841803](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20260605150841803.png)
  1. RDHX 参数过优：100% 吸热、0.0265 kWe/kWth、低 CAPEX、低隐含碳。
  2. 电价与碳因子正相关：成本调度和低碳调度方向一致。
  3. 设备 CAPEX 与 embodied carbon 同向：投资侧缺少“便宜但高碳 / 贵但低碳”的技术对照。
  4. 冷板 embodied carbon 只有 6 kgCO2/kW_IT，但 RDHX 更便宜，导致冷板没有形成足够强的替代冲突。
  5. 容量上界过宽，产生大量同时高成本高碳的 dominated 解。

## 9. 输出文件索引

- 全部历史评估：`D:\paper\cooling_retrofit_code\results\all_evaluations.csv`
- Pareto 解集：`D:\paper\cooling_retrofit_code\results\pareto_solutions.csv`
- 三方案对比：`D:\paper\cooling_retrofit_code\results\benchmark_comparison.csv`
- 三方案对比图：`D:\paper\cooling_retrofit_code\results\figures\benchmark_comparison.png`
- Pareto 前沿图：`D:\paper\cooling_retrofit_code\results\figures\pareto.png`
- Pareto 决策变量分布：`D:\paper\cooling_retrofit_code\results\figures\pareto_solution_distribution.png`
- 机房空间布局图：`D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room*.png`
- 构型选择与业务爬升/空调老化关系图：`D:\paper\cooling_retrofit_code\results\figures\config_choice_vs_growth_age.png`
- pymoo 收敛总览：`D:\paper\cooling_retrofit_code\results\figures\pymoo_convergence_summary.png`
- 收敛指标明细：`D:\paper\cooling_retrofit_code\results\convergence\pymoo_convergence_metrics.csv`
- Pareto 详情目录：`D:\paper\cooling_retrofit_code\results\pareto_details`
- 本报告：`D:\paper\cooling_retrofit_code\results\report.md`

## 10. 参数来源与假设

- `economic_defaults`：source=D:\paper\原研究问题\可参考数据.md plus D:\paper\原研究问题\原来的代码\config.json fallback，confidence=medium，User-fitted coefficients override these values; cost/carbon defaults are placeholders until project-specific vendor quotations are available.
- `fitted_coefficients`：source=fitted，confidence=high，RDHX, chiller COP, and AC fan coefficients supplied by the researcher.

## 11. 后续审核重点

- 正式论文实验前，应继续替换成本、隐含碳、重量、寿命、FOM/VOM 等占位参数。
- 当前供热需求使用原始输入值；供热侧结论表示余热可利用/可消纳量，不代表必须满足建筑完整供热需求。
