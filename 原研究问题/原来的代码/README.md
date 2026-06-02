# 数据中心改造成本-碳多目标优化（NSGA-II + 典型日 MILP）

本项目用于求解“数据中心改造背景下成本与隐含碳的多目标优化”问题。  
外层采用 NSGA-II 做规划优化，内层采用 Gurobi MILP 做运行调度优化，并用“典型日 + 峰值日”替代全年 8760 小时全量 MILP。

优化目标：
- 目标 1：最小化年化总成本 `TLCC`
- 目标 2：最小化年化总碳排 `TCE`

> 协作提醒：修改代码、约束、参数、算法流程或数学说明后，必须同步更新对应文档。  
> 当前项目的协作约定见 [AGENTS.md](/d:/Myproject/AGENTS.md)，审查文档主要在 `source_md/` 下。

## 1. 项目结构（完整）

```text
Myproject/
├─ data/
│  ├─ config.json                # 全部参数配置（经济/技术/碳/模型2/求解器）
│  ├─ time_series.csv            # 年度小时序列（推荐含 rack_it_*_kw / price_grid / cf_grid / heat_demand_kw / t_wb）
│  └─ H_matrix.csv               # 机柜级热拓扑矩阵 H_ji（必需）
├─ src/
│  ├─ main.py                    # 一键入口：优化 + 可视化 + 分析
│  ├─ run_optimization.py        # NSGA-II 外层优化入口
│  ├─ nsga2.py                   # NSGA-II 算法实现（混合编码）
│  ├─ model.py                   # 外层评估器：解码 + 典型日 + 调 inner MILP + 年化目标
│  ├─ feasibility_oracle.py      # 外层快速可行性筛查与投影修复（CDU / HP / CRAC / Chiller / Tower）
│  ├─ inner_dispatch.py          # 内层 Gurobi MILP（动态业务调度/热拓扑/储能/功热平衡）
│  ├─ typical_days.py            # 典型日构建（4季典型日 + 1峰值日）
│  ├─ data_utils.py              # 配置与时序数据加载/校验
│  ├─ visualize.py               # 图表输出
│  ├─ analyze_results.py         # 结果分析报告输出
│  └─ result_utils.py            # 结果解码等共享工具
├─ results/
│  ├─ pareto_solutions.csv       # 帕累托解集
│  ├─ representative_solutions.csv
│  ├─ runtime_summary.json       # 本轮优化总体运行时间与单点评估耗时统计
│  ├─ generation_timing_report.csv # 每代评估耗时/缓存/可行率统计
│  ├─ figures/                   # 前沿图、分布图、调度图
│  ├─ dispatch/                  # 代表解调度时序导出
│  ├─ convergence/               # pymoo 收敛性快照与 HV/IGD+ 指标
│  ├─ archive/                   # 历史结果快照
│  ├─ logs/                      # 临时运行日志
│  ├─ cache/                     # 本地缓存，不作为论文结果
│  └─ README.md                  # results 目录约定
├─ reports/
│  └─ result_analysis.md         # 自动分析报告
├─ source_docx/                  # 原始文档
├─ source_pdfs/                  # 原始 PDF
├─ source_md/                    # 模型说明与审查文档
│  ├─ modelexplanation.md        # 数学模型主说明（权威口径）
│  ├─ algorithm_process.md       # 算法执行流程说明
│  ├─ feasibility_oracle_visualization_prompt.md # Oracle 论文配图 prompt
│  ├─ innerconstraitcheck.md     # 内层约束审查稿
│  ├─ outerconstraintcheck.md    # 外层约束审查稿
│  ├─ parameter_catalog.md       # 参数目录
│  ├─ real_data_replacement_checklist.md # 真实数据替换清单
│  ├─ data-center-dataset-processing-skill/ # 真实数据处理 Skill 文档
│  └─ parallel_benchmark_manual.md # 并行 workers / inner threads 对比实验手册
├─ AGENTS.md                     # 协作约定与文档联动规则
├─ config.md                     # config.json 中文参数说明
├─ requirements.txt              # Python 依赖
└─ README.md
```

## 2. 运行环境要求

当前项目以本机环境为准，推荐使用 `Python 3.14` 与 `gurobipy==13.0.1`：
- Python: `3.14.x`
- Gurobi Python 包: `gurobipy==13.0.1`
- 有效 Gurobi License（13.x，且已与本机安装匹配）

> 说明：旧版 `Python 3.11 + gurobipy 12.0.3` 要求仅适用于旧电脑；本项目当前默认环境已切换为 `Python 3.14 + gurobipy 13.0.1`。

## 3. 快速开始

在 `D:\Myproject` 目录执行：

```powershell
# 1) 创建 Python 3.14 虚拟环境
py -3.14 -m venv .venv314

# 2) 安装依赖
D:\Myproject\.venv314\Scripts\python.exe -m pip install -r requirements.txt

# 3) 一键运行
D:\Myproject\.venv314\Scripts\python.exe src/main.py
```

## 4. 算法执行过程（核心）

### Step 1: 读取配置与数据
- 读取 `data/config.json`
- 读取/校验 `data/time_series.csv`
- 推荐主输入字段为：`rack_it_*_kw`、`price_grid`、`cf_grid`、`heat_demand_kw`、`t_wb`
- 若缺少 `rack_it_*_kw` 且 `require_explicit_rack_cols=false`，会按规则生成异构机柜 IT 负荷曲线
- 读取并校验 `data/H_matrix.csv` 尺寸是否为 `|R| x |R|`

### Step 2: 典型日降维
- 将全年小时数据按天处理
- 构建 `4` 个季节典型日 + `1` 个峰值日（每个场景 24 小时）
- 生成场景权重 `W_k`，用于将运行成本/碳加权回全年
- 典型日特征默认基于 `price_grid`、`cf_grid`、`heat_demand_kw`、`t_wb`、`it_total_kw` 和机柜负荷 `p10/p50/p90/std`

### Step 3: 外层 NSGA-II 生成规划解
- 染色体采用混合编码：
  - 二进制：`y_i`（机柜改造）、`x_ig`（设备组冷板接入）
  - 整数：`N_CRAC_remove`
  - 连续：`Cap_CDU, Cap_HP, Cap_BESS^E, Cap_BESS^P, Cap_TES, Cap_Tower, Cap_HX`
- 外层算法引擎可通过 `solver.outer_engine` 切换：
  - `custom`：项目内置 NSGA-II 实现，默认 fallback
  - `pymoo`：使用 `pymoo` 的 NSGA-II 框架，仍复用本项目的 `evaluate_solution()`、Oracle、worker cache 与内层 Gurobi MILP
- 进化过程：
  - 非支配排序 + 拥挤距离
  - 二进制位：均匀交叉 + bit-flip 变异
  - 连续位：SBX + 多项式变异
- 在进入内层 MILP 前，会先调用 `feasibility_oracle.py` 对候选规划解做快速可行性筛查与必要投影修复

### Step 4: 内层 MILP 运行调度评估（每个候选解）
- 外层给定规划变量后，内层用 Gurobi 建模并求解：
  - 已知机柜 IT 负荷输入 `rack_it_*_kw`
  - IT 热映射与液冷/风冷分摊
  - 机柜级热拓扑约束 `theta_{i,k,t}`（含 `H_ji`）
  - BESS/TES 动态、充放互斥、SOC 边界
  - 电力平衡、供热平衡、CRAC/Chiller/Tower/CDU 容量与冗余
  - CDU/Tower 变工况功率 PWL 线性化
- 默认并行策略为“外层进程并行 + 内层单模型少线程”：
  - `solver.parallel.workers = 2`
  - `solver.inner.threads = 1`
  - `solver.inner.persistent_template = true`
  - `solver.inner.mip_focus = 1`
  - `solver.inner.nodefile_start_gb = 0.5`
- 内层会对相同或相邻量化容量、且离散拓扑一致的候选解尝试复用上一轮调度结果作为 `MIP start`
- 当 `solver.inner.persistent_template = true` 时，每个 worker 会复用持久 Gurobi 模板模型，只更新候选解相关系数、RHS 与 warm start，而不是每次从零建模

### Step 5: 目标函数回传外层
- `TLCC = 年化CAPEX + 年化FOM + 运行成本 + 罚项`
- `TCE = 年化隐含碳 + 运行碳排 + 电池动态碳退化 - 供热替代碳抵扣 + 罚项`
- NSGA-II 基于 `(TLCC, TCE)` 迭代，最终输出帕累托前沿。

### Step 6: 结果输出与后处理
- `results/pareto_solutions.csv`
- `results/representative_solutions.csv`
- `results/runtime_summary.json`
- `results/generation_timing_report.csv`
- `results/figures/*.png`
- `results/dispatch/*_dispatch_timeseries.csv`
- `reports/result_analysis.md`
- `results/figures/pareto_front_pymoo.png`：基于 `pymoo` 的 Pareto 前沿散点图
- `results/figures/solution_distribution_pymoo.png`：基于 `pymoo` PCP 的多目标解集分布图
- 当 `solver.outer_engine = "pymoo"` 且 `solver.convergence.enabled = true` 时，额外输出 `results/convergence/pymoo_generation_population.csv`、`results/convergence/pymoo_convergence_metrics.csv` 和 `results/figures/pymoo_convergence_*.png`
- 调度图当前按 `winter / spring / summer / autumn / peak_day` 分开绘制，避免多个典型日混在线图中难以辨认
- `results/README.md` 记录结果目录整理规则：主结果留根目录，历史/日志/缓存放入子目录

## 5. 关键建模约定

- 当前主模型不再区分“高密机柜 / 低密机柜”两类口径；所有机柜统一由外部机柜级 IT 负荷曲线驱动，并由 `y_i` 决定是否进行液冷改造。
- `existing.n_racks_total` 是当前总机柜数唯一口径，模型不再保留高密/低密机柜数量拆分。
- Chiller 不参与改造（无新增 CAPEX/隐含碳），但参与运行约束与运行电耗计算。
- BESS 电芯隐含碳采用“按放电吞吐折算”的动态惩罚，不做静态一次性计入。
- 若内层 MILP 不可行，`evaluate_solution()` 会先按配置尝试 `INF_OR_UNBD` 复判、诊断松弛、可映射容量投影和一次重求；仍不可行时返回大罚值，并可写入 infeasible memory 避免重复评估同类无效候选解。

## 6. 常见问题

1. 报错 `Version number is X.Y, license is for version Z.W`
- 原因：`gurobipy` 与 License 主版本不一致。
- 处理：确认当前环境使用 `Python 3.14 + gurobipy==13.0.1`，并检查本机 Gurobi License 是否为 13.x。

2. 报错 `time_series missing required rack IT load columns`
- 原因：`time_series.csv` 缺少 `rack_it_000_kw` 等机柜级 IT 负荷列，且 `model2.rack_load.require_explicit_rack_cols=true`。
- 处理：补齐 `rack_it_*_kw` 列，或将 `require_explicit_rack_cols` 设为 `false` 以启用规则生成。

3. 报错 `H matrix shape mismatch`
- 原因：`H_matrix.csv` 尺寸与机柜数不一致。
- 处理：保证 `H_matrix.csv` 为 `|R| x |R|`。
