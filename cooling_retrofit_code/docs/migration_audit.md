# 新研究问题模型迁移审核

本项目的代码迁移原则是继承原研究问题的算法外壳和求解流程，只替换规划决策、运行约束、目标函数和冷热量流。新模型的准则文件是 `D:\paper\论文研究问题_建模.md`，方法论参考 `D:\paper\原研究问题` 下的原算法、原内外层约束和原代码。

## 方法论继承映射

| 原问题组件 | 新项目实现 | 状态 |
| --- | --- | --- |
| 外层 NSGA-II 生成规划个体 | `run_optimization.py`、`nsga2.py` | 已继承；最终 Pareto 从全部历史可行解筛选 |
| 个体 repair | `repair.py` | 已独立维护；负责构型互斥、容量联动、冷机/冷却塔、承重约束 |
| MILP 前廉价筛查 | `feasibility_oracle.py` | 已恢复 Oracle 组织；LP relaxation 仅保留接口，作为后续增强 |
| 不可行记忆 | `FeasibilityOracle` | 已加入 repaired-vector 签名缓存，避免重复送入同类不可行结构 |
| 内层 MILP 调度 | `inner_dispatch.py` | 已替换为新研究问题热量流和电热平衡模型 |
| 目标回传 | `model.py` | 已回传 TLCC/TCE，并保留 repair、screening、dispatch、diagnostics artifacts |
| 典型日 | `typical_days.py` | 已使用 K=8 k-means 代表日；可按配置追加峰值日 |
| 结果输出 | `run_optimization.py`、`analyze_results.py` | 已输出全历史评估、Pareto、调度明细、solution JSON 和报告 |

## 新模型代码准则

- 外层规划变量包含区域构型、空气侧新增容量、RDHX、CDU、ASHP、WSHP、BESS 整体容量、TES、冷机增容和冷却塔增容。
- RDHX 背板按可吸收其负责机柜 100% IT 热量建模，不再设置辅助风冷/房间冷却链路；冷板/CDU 热量不进入 chiller，只能进入 WSHP 或自然冷却排热；空调和 RDHX 未回收热量进入 chiller。
- 冷却塔承担冷机冷凝侧排热，以及冷板/CDU 未进入 WSHP 的自然冷却直接排热。
- 内层完整调度空气侧、冷板/CDU、RDHX、ASHP、WSHP、Chiller、Cooling Tower、TES、BESS、供热平衡、电力平衡和热点代理。
- `s_air` 表示区域有效空气侧冷却量，必须覆盖空气侧负荷且不超过该区域空气侧容量。
- BESS 不再拆分功率容量和能量容量；成本、隐含碳和承重按整体容量计入，充放电功率由整体容量和固定倍率派生。
- 固定改造成本/隐含碳表示施工、接口和停机等固定项；新增设备按容量单独计入，避免重复。

## 参数来源优先级

1. 用户拟合或明确给定的参数：RDHX 单位冷量耗电系数 `0.0265 kWe/kWth`、Chiller COP `3`、空调末端风机系数 `0.2137 kWe/kWth`。
2. `D:\paper\real_data` 中可直接读取的数据。
3. `D:\paper\原研究问题\可参考数据.md` 中的参考成本、碳和设备参数。
4. `D:\paper\原研究问题\原来的代码\config.json` 中的原项目默认值。
5. 暂无真实值的占位参数，必须保留在 `config.json` 的 `assumptions` 中，后续用报价或实测数据替换。

## 已知后续增强

- LP relaxation screening 尚未实现，本轮只保留接口，避免引入额外求解时间和调试面。
- 当前不可行诊断以结构筛查、Gurobi 状态和调度残差为主；如果后续需要定位具体冲突约束，可加入原项目式 relaxation/elastic IIS 报告。
- 成本、隐含碳、FOM/VOM 仍以参考数据和原项目 fallback 为主，论文定稿前需要替换为明确来源的实证或报价数据。
