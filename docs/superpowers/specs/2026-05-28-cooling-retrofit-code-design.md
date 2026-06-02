# Cooling Retrofit Code Design

## 1. 目标与范围

本项目在 `D:\paper\cooling_retrofit_code\` 下新建独立代码工程，用于实现“面向既有风冷数据中心低碳改造的协同规划模型”的基础求解代码。原问题代码目录 `D:\paper\原研究问题\原来的代码\` 仅作为算法和接口参考，不直接修改，也不全量复制。

第一版目标是完整保留原项目的基本求解框架：外层 NSGA-II 搜索规划解，内层 Gurobi MILP 评估运行调度，输出 `TLCC` 与 `TCE` 的 Pareto 前沿，并生成 CSV/JSON、图表和中文 Markdown 分析报告。

第一版不追求复刻原项目所有缓存、诊断、自动修复和性能技巧，而是优先保证新研究问题的模型语义清晰、变量不重复、代码边界明确、结果可解释。

## 2. 已确认需求

- 新建独立项目目录：`D:\paper\cooling_retrofit_code\`。
- 使用 Gurobi 作为内层 MILP 求解器。
- 输入采用混合模式：结构数据、负荷数据和已整理的小时级运行数据必须来自 `D:\paper\real_data`，设备经济性和缺失工程参数允许从配置默认值读取，并在输出中标记为 assumed。
- 外层完整实现七类改造构型 `k=0..6`。
- 内层完整包含 Chiller、Tower、ASHP、WSHP、TES、BESS、grid purchase、heat supply、冷热量分流和 SOC 约束。
- 时序支持 `typical_days` 与 `full_timeseries` 两种模式，默认使用 k-means 质心代表日法，`K=8`。
- 目标函数保持双目标：年化总成本 `TLCC` 和年化总碳排 `TCE`。
- 输出包含 CSV/JSON、图表和中文 Markdown 分析报告。
- 新项目保留算法求解框架，但不照搬旧项目冗余代码；必须清理旧模型遗留变量、函数和结果字段。
- 可行性处理采用“外层 repair + MILP 前廉价 screening + 正式 MILP”三层结构，第一版不实现 LP relaxation screening，仅预留后续增强接口。

## 3. 项目结构

拟创建以下文件：

```text
D:\paper\cooling_retrofit_code\
  README.md
  config.json
  main.py
  data_utils.py
  typical_days.py
  decision.py
  repair.py
  feasibility_oracle.py
  inner_dispatch.py
  model.py
  nsga2.py
  run_optimization.py
  analyze_results.py
  plotting.py
  docs\
    model_mapping.md
    assumptions.md
  tests\
    test_data_utils.py
    test_typical_days.py
    test_decision_schema.py
    test_repair.py
    test_inner_dispatch_small.py
    test_evaluate_solution.py
```

核心职责如下：

- `main.py`：一键运行优化、图表生成和中文报告。
- `config.json`：集中管理路径、场景、典型日、Gurobi、NSGA-II、技术参数、经济参数、碳参数和承重参数。
- `data_utils.py`：读取和校验 `D:\paper\real_data` 中的真实数据，处理时间对齐和 assumed 参数标记。
- `typical_days.py`：实现 k-means 质心代表日法和全时序模式。
- `decision.py`：定义外层决策编码、边界、解码和派生构型状态。
- `repair.py`：实现个体生成后的先验 repair，以及正式 MILP 前的规则 screening。
- `feasibility_oracle.py`：包装 screening 结果、约束违反度和不可行诊断。
- `inner_dispatch.py`：构建并求解 Gurobi 内层 MILP。
- `model.py`：连接 planning decision、repair、screening、MILP 和目标函数汇总。
- `nsga2.py`：实现 NSGA-II 进化流程，接入 Deb 约束处理思想。
- `run_optimization.py`：读取配置、构建数据、运行外层优化、保存结果。
- `analyze_results.py` 与 `plotting.py`：生成 CSV/JSON、图表和中文 Markdown 报告。

## 4. 旧代码取舍原则

保留或改写：

- NSGA-II 的初始化、交叉、变异、非支配排序、拥挤度和 Pareto 输出思想。
- “外层规划解 -> 内层 Gurobi MILP 评估 -> 返回 TLCC/TCE”的求解框架。
- 典型日和结果分析的接口思想。

清理掉：

- 旧模型主变量 `y_i`、`x_ig`、设备组液冷接入和 `n_crac_remove`。
- 与新热量流不一致的旧 CDU/HP/CRAC/Tower 自动修复逻辑。
- 旧项目中复杂的缓存、retry、诊断扩展和不再适用的结果字段。
- 与新论文模型无关的函数、配置段和中间变量。

新代码统一通过 `PlanningDecision` 和 `EvaluationResult` 在模块间传递信息。构型派生状态只在解码阶段生成，不在多个文件重复计算。

## 5. 外层变量设计

外层个体采用紧凑编码：

- `s_z ∈ {0,...,6}`：空调机组区域 `z` 的构型选择，解码为 `u_{z,k}`。
- 区域容量变量：
  - `Cap_z_AC_new`
  - `Cap_z_vent`
  - `Cap_z_RDHX`
  - `Cap_z_CDU`
- 系统容量变量：
  - `Cap_ASHP`
  - `Cap_WSHP`
  - `Cap_BESS_E`
  - `Cap_BESS_P`
  - `Cap_TES`
  - `Delta_Cap_Chiller`
  - `Delta_Cap_Tower`

派生状态包括：

- `o_z`：保留旧空调。
- `n_z`：使用新空调。
- `p_z`：使用冷板/CDU。
- `r_z`：使用 RDHX。
- `a_z`：空气源热泵回收路径可用。
- `w_z`：水源热泵回收路径可用。

派生状态只由 `s_z` 决定，不作为外层独立变量。

## 6. 配置设计

`config.json` 分为以下段落：

- `paths`：输入数据目录 `D:\paper\real_data`、输出目录。
- `scenario`：复制机房数量、负荷爬升模式、时间模式。
- `typical_days`：启用状态、聚类天数 `K=8`、特征列、随机种子、是否附加峰值日。
- `solver`：Gurobi 参数和 NSGA-II 参数。
- `technology`：构型定义、容量上限、COP、泵耗、热捕获比例。
- `economics`：CAPEX、FOM、VOM、热价、折现率和寿命。
- `carbon`：电网碳因子、替代供热碳因子、隐含碳参数。
- `weight`：机柜重量、设备重量、楼板承重上限和区域面积。
- `assumptions`：缺失参数的默认值、来源描述和置信等级。

配置中的缺失工程参数必须携带来源标记，例如 `source: "assumed"`、`source: "reference_data"`、`source: "fitted"` 或 `source: "measured"`，并进入报告。

## 7. 典型日设计

默认使用 k-means 质心代表日法：

1. 将全时序按自然日切分为 `n_days × 24`。
2. 为每一天构造特征向量，候选特征包括电价、碳因子、IT 总负荷、rack 负荷统计量、室外温度或湿球温度、供热需求。
3. 标准化日特征。
4. 用 k-means 聚类为 `K=8` 个日簇。
5. 对每个簇选择距离质心最近的真实日作为代表日。
6. 代表日权重等于该簇包含的天数。
7. 可选添加极端峰值日用于容量校验；如果峰值日已经被选中，则不重复添加。
8. 所有运行成本、运行碳排和供热量按代表日权重还原为年化统计。

实现上不直接使用虚拟质心日曲线，以避免 24 小时序列失真。

## 8. 内层 MILP 设计

内层 Gurobi MILP 接收已经解码和 repair 后的 `PlanningDecision`，不再决定构型。

热量流口径：

- 空气侧：
  - `s_air[z,t]` 表示空调机组区域 `z` 的有效空气侧冷却量。
  - `s_air[z,t]` 只受可用空气侧容量限制，不强制等于空气侧热负荷。
  - 空气侧未被 ASHP 回收的热量进入 Chiller 蒸发侧。
- RDHX：
  - 仅构型 5 启用 `Cap_z_RDHX`。
  - RDHX 冷源由 Chiller 提供。
  - RDHX 水侧热量可进入 WSHP，未回收部分由 Chiller 负担。
  - RDHX 耗电使用整体系数 `e_z_RDHX`，不拆分 terminal 和 pump。
- 冷板/CDU：
  - 仅构型 4 和 6 启用冷板与 `Cap_z_CDU`。
  - 冷板热量经 CDU 后分为 WSHP 热源和自然冷却直接排热两路。
  - 冷板/CDU 不进入 Chiller 负荷。
  - CDU 容量限制冷板侧可承接热量。
- Chiller：
  - 只承担空气侧未回收热量和 RDHX 未由 WSHP 回收的热量。
  - `P_Chiller[t] = Q_Chiller_evap[t] / COP_Chiller[t]`。
  - 容量为 `Cap_Chiller_old + Delta_Cap_Chiller`。
- Cooling tower：
  - 承担 Chiller 冷凝侧排热。
  - 承担冷板/CDU 未进入 WSHP 的自然冷却直接排热。
  - `Q_Tower[t] = Q_Chiller_evap[t] + P_Chiller[t] + Q_CDU_to_free[t]`。
- 热泵与供热：
  - ASHP 只用于空气侧余热升级供热。
  - WSHP 用于 RDHX 和冷板/CDU 水侧余热升级供热。
  - 不允许余热直接供热。
- TES/BESS：
  - TES 接供热侧，平衡热泵供热与外部供热需求。
  - BESS 接电力侧，参与购电、充放电和 SOC 约束。
- 电力平衡：
  - 购电覆盖 Chiller、Tower、terminal fan、RDHX、CDU pump、WSHP pump、ASHP/WSHP 电耗、TES 辅助电耗和 BESS 充电，扣除 BESS 放电。
- 热安全：
  - 使用 `H_matrix`、rack IT load、空气侧有效冷却量和供风温度参数约束 rack 入口温度。
  - 热安全松弛默认允许但惩罚很高，用于诊断而不是常规优化。

约束构建函数保持少量且有清晰边界：

- `_add_air_side_constraints()`
- `_add_liquid_heat_constraints()`
- `_add_chiller_tower_constraints()`
- `_add_heat_pump_storage_constraints()`
- `_add_power_balance()`
- `_add_thermal_safety_constraints()`

同一物理量只在一个函数中创建变量或约束，避免重复定义。

## 9. 目标函数设计

`TLCC` 包括：

- 固定改造成本年化：`CRF_k * C_zk_retro * u_zk`。
- 容量设备投资年化：新空调、辅助通风、RDHX、CDU、ASHP、WSHP、Chiller、Tower、TES、BESS。
- 年运行成本：购电、可变运维、固定运维和热安全松弛惩罚。
- 供热收益或替代供热成本抵扣：只计算经 ASHP/WSHP 升级后的有效供热量，并受 `Q_heat_demand[t]` 限制。

`TCE` 包括：

- 固定改造隐含碳年化：`EI_zk_retro / n_k * u_zk`。
- 容量设备隐含碳年化：各设备单位容量隐含碳除以寿命。
- 运行碳排：购电量乘逐时电网碳因子，扣除替代供热碳减排。

配置中控制是否允许运行碳抵扣后为负。默认不允许总运行碳小于 0。

固定改造成本和隐含碳只包含拆改、接口、施工、调试、停机损失等固定项，不包含新空调、RDHX、CDU、热泵、Chiller、Tower、TES 和 BESS 等容量设备，避免重复计算。

## 10. 可行性加速设计

可行性处理分三层。

### 10.1 外层个体 repair

在初始化、交叉和变异之后执行：

- 构型互斥 repair：清零不适用于当前构型的容量。
- 构型最低容量 repair：选择新空调、RDHX、冷板/CDU 时，容量低于区域负荷下界则抬升到最小可行容量。
- 设备配比 repair：匹配冷板与 CDU、热泵容量与热源上界、BESS 能量与功率最小时长关系。
- Chiller/Tower 链条 repair：根据空气侧和 RDHX 峰值下界估算 Chiller 扩容，根据 Chiller 冷凝侧排热和 CDU 直接排热估算 Tower 扩容。
- 承重 repair 或截断：容量变量导致承重越界时优先下调相关容量；构型本身导致 rack 静载不可行时标记为不可修复。
- repair 后投影回上下界并记录 `repair_actions`。

### 10.2 MILP 前廉价 screening

正式调用 Gurobi 前执行：

- 峰值供冷 screening。
- 热安全上界 screening。
- Chiller/Tower 能量链 screening。
- WSHP/ASHP/供热上界 screening。
- TES/BESS 功率和能量上界 screening。

第一版不实现 LP relaxation screening，但保留函数入口 `run_lp_relaxation_screening()`，默认返回 skipped 状态。

### 10.3 正式 MILP 与约束处理

只有通过 repair 和 screening 的个体进入完整 Gurobi MILP。MILP 不可行时输出诊断，不做自动修复重试。

NSGA-II 使用 Deb 的约束处理思想：

- 可行解优先于不可行解。
- 两个可行解按 `TLCC/TCE` 非支配关系比较。
- 两个不可行解按约束违反度比较，违反度更小者更优。

每代输出 infeasible 统计，包括 repair 次数、screening 拦截原因、MILP infeasible 原因和平均内层求解时间。

## 11. 输出设计

输出目录包括：

- `results/pareto_solutions.csv`：非支配解、目标值、构型分布、容量配置和可行性。
- `results/all_evaluations.csv`：全部评估个体、repair 记录、screening 结果和求解时间。
- `results/solutions/*.json`：代表解完整规划变量和调度统计。
- `results/dispatch/*.csv`：代表解的典型日或全时序调度结果。
- `results/figures/*.png`：Pareto 图、构型分布图、容量配置图、冷热量流图、成本分解图、碳分解图。
- `results/report.md`：中文结果摘要，说明参数来源、关键发现和不可行诊断。

报告必须明确标注 `measured`、`fitted`、`reference_data` 和 `assumed` 参数。

## 12. 测试设计

测试文件和目标：

- `tests/test_data_utils.py`：验证真实数据读取、形状检查、时间对齐、小时级 AC 运行数据读取和 assumed 参数标记。
- `tests/test_typical_days.py`：验证 k-means `K=8` 生成 8 个真实代表日，权重总和正确。
- `tests/test_decision_schema.py`：验证七类构型编码、容量边界、one-hot 解码和派生状态。
- `tests/test_repair.py`：验证互斥容量清零、冷板/CDU 配比、Chiller/Tower 下界和承重 screening。
- `tests/test_inner_dispatch_small.py`：用小样例验证 Gurobi MILP 可建模、可求解，且 Chiller、Tower、RDHX、CDU 热量流不混淆。
- `tests/test_evaluate_solution.py`：验证一个规划解能完整经过 repair、screening、MILP 和目标汇总。

测试使用已整理的小时级 AC 运行数据 `m100_22-07_ac_unit_hourly.csv` 做轻量读取校验；原始 15 分钟级 AC telemetry 文件只作为可追溯数据源，不进入常规测试流程。

## 13. 文档设计

新项目文档包括：

- `README.md`：项目目标、环境依赖、运行方式、数据要求和输出说明。
- `docs/model_mapping.md`：论文符号到代码字段的映射。
- `docs/assumptions.md`：假设参数、来源、置信等级和待补充数据。
- `results/report.md`：每次实验自动生成的中文结果报告。

## 14. 实施边界

本规格通过后，下一步是编写详细 implementation plan，不直接写代码。

第一版明确不做：

- 不修改 `D:\paper\原研究问题\原来的代码\`。
- 不全量复制旧项目。
- 不保留旧模型主变量和旧修复体系。
- 不实现 LP relaxation screening。
- 不把原始 15 分钟级 AC telemetry 文件放入常规测试流程；小时级文件 `m100_22-07_ac_unit_hourly.csv` 可以进入基础数据读取校验。

第一版必须做到：

- 新项目目录可独立运行。
- 能读取现有真实结构和负荷数据。
- 能使用配置假设补齐缺失工程参数。
- 能运行 NSGA-II + Gurobi MILP 双层求解。
- 能输出 Pareto、调度结果、图表和中文报告。
