# Data Center Cooling Retrofit Optimization

本项目用于论文新研究问题的基础代码实现：在业务负荷爬升与既有空调老化背景下，对数据中心冷却系统进行改造规划，并通过“外层 NSGA-II 规划搜索 + 内层 MILP 运行调度”的双层流程求解年化总成本和年化总碳排放的 Pareto 解。

## 1. 当前模型口径

本项目以 `D:\paper\论文研究问题_建模.md` 为准。原研究问题只作为算法流程和代码组织方式参考，不沿用旧模型变量。

关键口径如下：

- `psi_t^{amb}=1`，本阶段不考虑环境工况对风冷容量的修正。
- RDHX 背板按可吸收负责机柜 100% IT 热量建模，不再设置 RDHX 辅助风冷或房间冷却容量。
- BESS 作为整体容量变量 `cap_bess` 建模，成本、隐含碳和重量均按整体容量计入，不再拆分功率容量和能量容量。
- 供热需求作为输入数据处理：已按 IT 余热规模缩放。内层模型中供热平衡为硬约束，不设置未供热松弛罚金。
- 冷板/CDU 热量不进入 chiller，只能进入 WSHP 或自然冷却排热。
- 空调和 RDHX 未被热泵回收的热量进入 chiller。
- 冷却塔承担 chiller 冷凝侧排热和 CDU 自然冷却直接排热。
- 不考虑机房空间约束，只保留承重约束。

## 2. 求解流程

代码继承原研究问题的方法论流程：

1. 读取配置、真实数据、机柜元数据、热耦合矩阵和代表日数据。
2. 用 K=8 k-means 质心代表日法生成运行时序，并保留 `representative_day_id` 与 `day_weight`。
3. 外层 NSGA-II 生成规划个体。
4. `repair.py` 对规划解进行先验修复，包括构型互斥、容量联动、冷机/冷却塔上界和承重约束。
5. `FeasibilityOracle` 做廉价结构筛查，并缓存明显不可行规划解。
6. `InnerDispatchMILP` 求解代表日运行调度。
7. `model.py` 汇总投资、固定运维、运行成本、运行碳、隐含碳和诊断指标。
8. 从全部历史可行解中筛选非支配 Pareto 前沿。
9. 输出报告、调度明细、Pareto 图、收敛性分析图和机房空间改造图。

## 3. 外层规划变量

区域级变量：

- `config_z`：空调组/机柜组 `z` 的改造构型。
- `cap_ac_new_z`：新增空调容量。
- `cap_rdhx_z`：RDHX 背板容量。
- `cap_cdu_z`：冷板/CDU 容量。

系统级变量：

- `cap_ashp`：空气源热泵容量。
- `cap_wshp`：水源热泵容量。
- `cap_bess`：BESS 整体容量。
- `cap_tes`：TES 容量。
- `delta_cap_chiller`：新增 chiller 容量。
- `delta_cap_tower`：新增 cooling tower 容量。

## 4. 内层 MILP 运行模型

内层模型完整描述代表日运行过程：

- 机柜级 IT 热负荷按业务爬升后的输入曲线进入模型。
- 旧空调有效容量和末端单位冷量耗电按空调老化率修正。
- 热量分为空气侧、冷板/CDU、RDHX 三条链路。
- 空气侧可进入 ASHP，未回收部分进入 chiller。
- RDHX 可进入 WSHP，未回收部分进入 chiller。
- 冷板/CDU 可进入 WSHP，未回收部分进入自然冷却排热。
- TES 与 BESS 均采用代表日内循环 SOC。
- 热安全通过 `H_matrix` 和机柜元数据建立热点代理约束。
- 电力平衡包含 IT 负荷、冷却设备耗电、热泵耗电、BESS 充放电。
- 供热平衡要求热泵与 TES 放热满足缩放后的供热需求。

## 5. 参数来源

参数优先级为：

1. 用户明确给定或拟合值，例如 `e_z^{RDHX}=0.0265 kWe/kWth`、`COP^{Chiller}=3`、空调末端风机单位冷量耗电系数 `0.2137 kWe/kWth`。
2. 当前项目 `data` / `real_data` 中的真实数据。
3. `D:\paper\原研究问题\可参考数据.md`。
4. `D:\paper\原研究问题\原来的代码\config.json` 中的 fallback。

业务爬升率和空调老化率目前按固定随机种子模拟，并输出到结果目录，后续可替换为真实实证数据。

## 6. 主要入口

运行优化：

```powershell
python main.py
```

或直接运行：

```powershell
python run_optimization.py
```

运行聚焦测试：

```powershell
pytest tests/test_planning_model.py
```

## 7. 结果输出

主要结果文件包括：

| 文件 | 含义 |
| --- | --- |
| `results/all_evaluations.csv` | 全部历史候选解、可行性、目标值、repair 后变量和诊断 |
| `results/pareto_solutions.csv` | 全部历史可行解中的非支配 Pareto 解 |
| `results/pareto_details/run_*/solution_*.json` | Pareto 解的完整决策与诊断 |
| `results/pareto_details/run_*/dispatch_solution_*.csv` | Pareto 解的代表日调度明细 |
| `results/report.md` | 自动结果报告 |
| `results/figures/pareto_front.png` | Pareto 前沿图 |
| `results/figures/pymoo_convergence_summary.png` | 收敛性分析图 |
| `results/figures/rack_retrofit_layout.png` | 机房空间改造图 |
| `results/figures/growth_age_config_relation.png` | 业务爬升率、空调老化率与构型选择关系图 |

## 8. 结果报告重点

`analyze_results.py` 会汇总：

- 总评估数量、可行率和 Pareto 解数量。
- TLCC/TCE 范围。
- 最优 Pareto 解的构型选择与容量配置。
- repair / screening 诊断。
- 运行调度中的电力平衡、供热平衡、SOC、热点松弛和冷源流向残差。
- 业务爬升率、空调老化率与最终改造构型选择之间的统计关系。

