# 优化结果报告

## 1. 求解概览

- 总评估方案数：3936
- 可行方案数：154
- 可行率：3.91%
- Pareto 方案数：7
- 代数范围：0 - 40
- Pareto 年化成本范围：20,819,924.61 yuan/year - 25,451,480.24 yuan/year
- Pareto 年化碳排范围：11,632,148.57 kgCO2/year - 13,282,752.27 kgCO2/year

## 2. 模型与求解口径

- 方法论继承原研究问题：外层 NSGA-II 规划搜索，repair/screening 预筛查，内层 Gurobi MILP 运行调度，最后从全部历史可行解筛选 Pareto 前沿。
- 当前代码口径已同步：RDHX 链路不再设置辅助风冷容量；BESS 不再拆分功率容量和能量容量；供热需求为硬约束，不再设置未供热松弛罚金。
- RDHX 按可吸收其负责区域 100% IT 热量建模；未进入 WSHP 的 RDHX 热量进入 Chiller。
- BESS 仅使用整体容量 `cap_bess`，充放电功率上限由 `technology.bess.c_rate_per_hour` 派生。
- 典型日：K=8；append_peak_day=True。
- NSGA-II：population_size=96；generations=80；mutation_probability=0.15。
- 供热需求缩放：enabled=True；method=peak_ratio_to_it_load；target_peak_fraction_of_it_load=0.9。
- 缩放原因：Raw building heat demand peak is about 8 MW while IT waste-heat peak is below 1 MW; scale heat demand proportionally as the model input scenario.

## 3. 收敛性分析

- 指标后端：pymoo；参考前沿：history_nondominated
- 最终代数：40；历史可行解：154；最终前沿点数：7
- 收敛性分析观测代数：26
- 最终 Hypervolume：1.03
- 最终 IGD+：0.00
- 推荐判断：未检测到稳定平台期。

## 4. 模型-代码同步提示

- 未检测到 RDHX 辅助风冷、BESS 双容量或供热松弛罚金字段。

## 5. Pareto 与代表方案

### 最低成本解
- solution_id：3055
- generation：31
- TLCC：20,819,924.61 yuan/year
- TCE：13,282,752.27 kgCO2/year
- 区域构型计数：1 new_ac: 3；2 new_ac_with_ashp: 5；3 new_ac_high_efficiency: 3；4 cold_plate_cdu_wshp: 8；5 rdhx_wshp: 5；6 cold_plate_cdu_wshp_plus_air: 6
- 容量配置：
  - 新空调容量: 18,171.14 kW
  - RDHX 容量: 5,452.43 kW
  - CDU 容量: 18,888.89 kW
  - ASHP 容量: 4,294.81 kW
  - WSHP 容量: 2,729.41 kW
  - BESS 整体容量: 36,110.42 kWh
  - TES 容量: 11,849.93 kWh
  - Chiller 扩容: 14,522.52 kW
  - Cooling Tower 扩容: 6,415.42 kW
- 解详情：`D:\paper\cooling_retrofit_code\results\pareto_details\run_2905cbdc71ee4706a3625e4730c01544\solution_3055.json`
- 调度明细：`D:\paper\cooling_retrofit_code\results\pareto_details\run_2905cbdc71ee4706a3625e4730c01544\dispatch_solution_3055.csv`

### 最低碳排解
- solution_id：3601
- generation：37
- TLCC：25,451,480.24 yuan/year
- TCE：11,632,148.57 kgCO2/year
- 区域构型计数：1 new_ac: 2；2 new_ac_with_ashp: 5；3 new_ac_high_efficiency: 2；4 cold_plate_cdu_wshp: 6；5 rdhx_wshp: 6；6 cold_plate_cdu_wshp_plus_air: 9
- 容量配置：
  - 新空调容量: 16,416.48 kW
  - RDHX 容量: 8,201.01 kW
  - CDU 容量: 14,380.74 kW
  - ASHP 容量: 14,494.29 kW
  - WSHP 容量: 7,485.52 kW
  - BESS 整体容量: 8,339.21 kWh
  - TES 容量: 26,770.72 kWh
  - Chiller 扩容: 15,889.40 kW
  - Cooling Tower 扩容: 1,247.84 kW
- 解详情：`D:\paper\cooling_retrofit_code\results\pareto_details\run_2905cbdc71ee4706a3625e4730c01544\solution_3601.json`
- 调度明细：`D:\paper\cooling_retrofit_code\results\pareto_details\run_2905cbdc71ee4706a3625e4730c01544\dispatch_solution_3601.csv`

### 折中解
- solution_id：3135
- generation：32
- TLCC：23,412,862.37 yuan/year
- TCE：12,349,379.79 kgCO2/year
- 区域构型计数：1 new_ac: 5；2 new_ac_with_ashp: 2；3 new_ac_high_efficiency: 4；4 cold_plate_cdu_wshp: 6；5 rdhx_wshp: 9；6 cold_plate_cdu_wshp_plus_air: 4
- 容量配置：
  - 新空调容量: 19,198.47 kW
  - RDHX 容量: 11,906.82 kW
  - CDU 容量: 13,620.20 kW
  - ASHP 容量: 3,857.65 kW
  - WSHP 容量: 7,253.80 kW
  - BESS 整体容量: 16,815.27 kWh
  - TES 容量: 13,809.28 kWh
  - Chiller 扩容: 15,846.11 kW
  - Cooling Tower 扩容: 399.33 kW
- 解详情：`D:\paper\cooling_retrofit_code\results\pareto_details\run_2905cbdc71ee4706a3625e4730c01544\solution_3135.json`
- 调度明细：`D:\paper\cooling_retrofit_code\results\pareto_details\run_2905cbdc71ee4706a3625e4730c01544\dispatch_solution_3135.csv`

## 6. 可行性、repair 与不可行原因

- 历史可行解数量：154
- 历史非支配筛选后 Pareto 解数量：7
- repair 后向量与原始向量不同的候选解数量：3936
- `room03_cdz2:weight_margin_exceeded:1.8189894035458565e-12`: 400 次
- `room04_cdz4:weight_margin_exceeded:1.8189894035458565e-12`: 391 次
- `room01_cdz3:weight_margin_exceeded:1.8189894035458565e-12`: 359 次
- `room02_cdz3:weight_margin_exceeded:1.8189894035458565e-12`: 324 次
- `room03_cdz3:weight_margin_exceeded:1.8189894035458565e-12`: 306 次
- `room01_cdz1:weight_margin_exceeded:1.8189894035458565e-12`: 297 次
- `inner_dispatch_infeasible`: 296 次
- `room01_cdz5:weight_margin_exceeded:1.8189894035458565e-12`: 284 次

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
- room01_cdz1：构型=new_ac_high_efficiency；平均业务爬升率=16.56%；空调老化率=24.40%；区域峰值 IT 负荷=292.06 kW
- room01_cdz2：构型=cold_plate_cdu_wshp；平均业务爬升率=15.28%；空调老化率=20.82%；区域峰值 IT 负荷=195.99 kW
- room01_cdz3：构型=new_ac_high_efficiency；平均业务爬升率=13.04%；空调老化率=24.11%；区域峰值 IT 负荷=58.95 kW
- room01_cdz4：构型=rdhx_wshp；平均业务爬升率=12.76%；空调老化率=19.64%；区域峰值 IT 负荷=115.61 kW
- room01_cdz5：构型=cold_plate_cdu_wshp；平均业务爬升率=15.17%；空调老化率=10.05%；区域峰值 IT 负荷=330.32 kW
- room01_cdz6：构型=new_ac；平均业务爬升率=22.69%；空调老化率=16.54%；区域峰值 IT 负荷=21.58 kW
- room02_cdz1：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=12.68%；空调老化率=13.20%；区域峰值 IT 负荷=289.87 kW
- room02_cdz2：构型=rdhx_wshp；平均业务爬升率=17.79%；空调老化率=12.80%；区域峰值 IT 负荷=196.91 kW
- room02_cdz3：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=14.25%；空调老化率=12.61%；区域峰值 IT 负荷=59.11 kW
- room02_cdz4：构型=rdhx_wshp；平均业务爬升率=17.77%；空调老化率=20.52%；区域峰值 IT 负荷=116.71 kW
- room02_cdz5：构型=rdhx_wshp；平均业务爬升率=13.25%；空调老化率=14.05%；区域峰值 IT 负荷=329.27 kW
- room02_cdz6：构型=rdhx_wshp；平均业务爬升率=19.37%；空调老化率=19.38%；区域峰值 IT 负荷=21.45 kW
- room03_cdz1：构型=new_ac_with_ashp；平均业务爬升率=14.55%；空调老化率=21.52%；区域峰值 IT 负荷=290.92 kW
- room03_cdz2：构型=cold_plate_cdu_wshp；平均业务爬升率=14.19%；空调老化率=22.46%；区域峰值 IT 负荷=195.56 kW
- room03_cdz3：构型=new_ac_high_efficiency；平均业务爬升率=15.13%；空调老化率=12.98%；区域峰值 IT 负荷=59.12 kW
- room03_cdz4：构型=rdhx_wshp；平均业务爬升率=12.64%；空调老化率=8.79%；区域峰值 IT 负荷=115.55 kW
- room03_cdz5：构型=new_ac；平均业务爬升率=12.31%；空调老化率=22.06%；区域峰值 IT 负荷=328.60 kW
- room03_cdz6：构型=new_ac_with_ashp；平均业务爬升率=12.60%；空调老化率=12.55%；区域峰值 IT 负荷=21.17 kW
- room04_cdz1：构型=rdhx_wshp；平均业务爬升率=15.54%；空调老化率=19.70%；区域峰值 IT 负荷=291.36 kW
- room04_cdz2：构型=rdhx_wshp；平均业务爬升率=13.30%；空调老化率=17.40%；区域峰值 IT 负荷=195.33 kW
- room04_cdz3：构型=new_ac；平均业务爬升率=11.84%；空调老化率=12.14%；区域峰值 IT 负荷=58.81 kW
- room04_cdz4：构型=new_ac；平均业务爬升率=17.38%；空调老化率=13.39%；区域峰值 IT 负荷=116.65 kW
- room04_cdz5：构型=cold_plate_cdu_wshp；平均业务爬升率=14.56%；空调老化率=12.97%；区域峰值 IT 负荷=330.08 kW
- room04_cdz6：构型=cold_plate_cdu_wshp；平均业务爬升率=20.93%；空调老化率=8.10%；区域峰值 IT 负荷=21.51 kW
- room05_cdz1：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=15.25%；空调老化率=17.37%；区域峰值 IT 负荷=291.18 kW
- room05_cdz2：构型=rdhx_wshp；平均业务爬升率=15.03%；空调老化率=11.68%；区域峰值 IT 负荷=195.74 kW
- room05_cdz3：构型=new_ac_high_efficiency；平均业务爬升率=13.18%；空调老化率=21.25%；区域峰值 IT 负荷=58.96 kW
- room05_cdz4：构型=cold_plate_cdu_wshp_plus_air；平均业务爬升率=16.12%；空调老化率=21.67%；区域峰值 IT 负荷=116.31 kW
- room05_cdz5：构型=cold_plate_cdu_wshp；平均业务爬升率=14.01%；空调老化率=9.33%；区域峰值 IT 负荷=329.93 kW
- room05_cdz6：构型=new_ac；平均业务爬升率=8.80%；空调老化率=15.19%；区域峰值 IT 负荷=21.01 kW

## 8. 输出文件索引

- 全部历史评估：`D:\paper\cooling_retrofit_code\results\all_evaluations.csv`
- Pareto 解集：`D:\paper\cooling_retrofit_code\results\pareto_solutions.csv`
- Pareto 前沿图：`D:\paper\cooling_retrofit_code\results\figures\pareto.png`
- Pareto 决策变量分布：`D:\paper\cooling_retrofit_code\results\figures\pareto_solution_distribution.png`
- 机房空间布局图：`D:\paper\cooling_retrofit_code\results\figures\room_layout_knee_solution_room*.png`
- 构型选择与业务爬升/空调老化关系图：`D:\paper\cooling_retrofit_code\results\figures\config_choice_vs_growth_age.png`
- pymoo 收敛总览：`D:\paper\cooling_retrofit_code\results\figures\pymoo_convergence_summary.png`
- 收敛指标明细：`D:\paper\cooling_retrofit_code\results\convergence\pymoo_convergence_metrics.csv`
- Pareto 详情目录：`D:\paper\cooling_retrofit_code\results\pareto_details`
- 本报告：`D:\paper\cooling_retrofit_code\results\report.md`

## 9. 参数来源与假设

- `economic_defaults`：source=D:\paper\原研究问题\可参考数据.md plus D:\paper\原研究问题\原来的代码\config.json fallback，confidence=medium，User-fitted coefficients override these values; cost/carbon defaults are placeholders until project-specific vendor quotations are available.
- `fitted_coefficients`：source=fitted，confidence=high，RDHX, chiller COP, and AC fan coefficients supplied by the researcher.

## 10. 后续审核重点

- 正式论文实验前，应继续替换成本、隐含碳、重量、寿命、FOM/VOM 等占位参数。
- 当前供热需求已按 IT 余热量级缩放，供热侧结论应解释为余热消纳场景，不能直接外推到原始 8 MW 建筑供热峰值。
