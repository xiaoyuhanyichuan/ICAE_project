# `model.py` / `feasibility_oracle.py` 外层约束审查整理

## Update: Post-MILP Infeasibility Handling

Current implementation adds a second infeasibility handling layer after the pre-MILP Oracle:

- `FeasibilityOracle.assess()` remains the pre-MILP screening and projection layer. It classifies reasons such as `cdu_capacity`, `crac_capacity`, `chiller_capacity_fixed_infrastructure`, `tower_capacity`, `topology_envelope_failed`, and `projection_bound_hit`.
- If the formal inner MILP returns infeasible or `INF_OR_UNBD`, `evaluate_solution()` may rerun a fresh diagnostic model with `DualReductions=0`.
- If infeasibility is confirmed, a diagnostic relaxation model can enlarge the existing diagnostic slack bounds and summarize slack by category: CDU, CRAC, Chiller, Tower, Power, Heat, and SOC.
- Only slack that maps safely back to existing outer design variables is automatically repaired: CDU slack increases `Cap_CDU`; CRAC slack first reduces `n_CRAC_remove` and then may increase `Cap_HP`; Tower slack increases `Delta Cap_Tower`.
- Chiller slack is recorded as fixed-infrastructure infeasibility. The current outer model does not add new chiller capacity, so no automatic chiller projection is performed.
- If a mapped repair is applied, the formal inner MILP is resolved once. If it still fails, the candidate receives the existing penalty treatment.
- Infeasible memory is a heuristic no-good cache keyed by discrete topology, quantized capacities, and reason. It avoids repeatedly solving very similar infeasible candidates, but it is not a formal Benders feasibility cut.

This update documents engineering behavior in `src/model.py`, `src/feasibility_oracle.py`, and `src/inner_dispatch.py`. It does not change the mathematical model definition in `source_md/modelexplanation.md`.

本文档整理当前外层规划层在代码中的**实际约束实现**，用于审核 `src/model.py`、`src/feasibility_oracle.py` 与 `src/run_optimization.py` 是否符合预期。

对应源码主位置：
- [src/model.py](/d:/Myproject/src/model.py)
- [src/feasibility_oracle.py](/d:/Myproject/src/feasibility_oracle.py)
- [src/run_optimization.py](/d:/Myproject/src/run_optimization.py)

## 1. 外层求解结构

当前外层不是单个显式 MILP/LP，而是：

1. `NSGA-II` 在给定边界内搜索决策向量
2. `model.py` 将决策向量解码为离散拓扑与连续容量
3. `feasibility_oracle.py` 对拓扑和容量做快速可行性筛查/必要投影
4. `inner_dispatch.py` 对通过筛查的方案做内层运行调度
5. 外层再叠加空间、承重等工程约束罚项，形成 `TLCC` 与 `TCE`

因此，外层“约束”主要体现在：
- 决策变量边界
- 解码时的二进制/整数/兼容性限制
- Oracle 可行性筛查
- 空间/承重后处理约束
- 最终可行性判定

## 2. 外层决策变量

代码位置：
- [src/model.py:199](/d:/Myproject/src/model.py:199)
- [src/model.py:220](/d:/Myproject/src/model.py:220)
- [src/model.py:224](/d:/Myproject/src/model.py:224)

当前外层决策向量包含三类变量：

### 2.1 机柜液冷改造二进制

$$
y_i \in \{0,1\}, \quad \forall i \in R
$$

含义：
- `y_i = 1`：机柜 `i` 进行液冷改造
- `y_i = 0`：不改造

### 2.2 机柜-设备组液冷适配二进制

$$
x_{ig} \in \{0,1\}, \quad \forall i \in R,\; g \in G
$$

含义：
- `x_{ig} = 1`：机柜 `i` 中设备组 `g` 采用液冷
- `x_{ig} = 0`：不采用液冷

### 2.3 设备拆除/容量连续变量

$$
n_{CRAC}^{remove} \in \mathbb{Z}
$$

$$
Cap_{CDU},\; Cap_{HP},\; Cap_{BESS}^E,\; Cap_{BESS}^P,\; Cap_{TES},\; \Delta Cap_{Tower},\; Cap_{HX} \in \mathbb{R}
$$

其上下界来自 `data/config.json` 中 `decision_bounds`。

## 3. 决策变量边界与解码约束

## 3.1 边界约束

代码位置：
- [src/model.py:203](/d:/Myproject/src/model.py:203)
- [src/model.py:227](/d:/Myproject/src/model.py:227)

所有外层变量首先受到 `decision_bounds` 的上下界限制：

$$
\underline{x_j} \le x_j \le \overline{x_j}
$$

特别地，`cap_tower_kw` 若配置了 `delta_cap_tower_kw`，则代码使用该边界。

## 3.2 二进制与整数化

代码位置：
- [src/model.py:239](/d:/Myproject/src/model.py:239)
- [src/model.py:246](/d:/Myproject/src/model.py:246)
- [src/model.py:257](/d:/Myproject/src/model.py:257)

解码时，外层会将连续搜索向量强制投影为工程可实施决策：

$$
y_i = \text{clip}(\text{round}(x_i), 0, 1)
$$

$$
x_{ig} = \text{clip}(\text{round}(x_{ig}), 0, 1)
$$

$$
n_{CRAC}^{remove} = \text{clip}(\text{round}(x), \underline{n}, \overline{n})
$$

$$
Cap_m = \text{clip}(x, \underline{Cap_m}, \overline{Cap_m})
$$

含义：虽然 NSGA-II 在连续空间搜索，但实际进入物理模型前会被投影到离散/有界工程变量。

## 3.3 兼容性与从属关系

代码位置：
- [src/model.py:241](/d:/Myproject/src/model.py:241)
- [src/model.py:242](/d:/Myproject/src/model.py:242)

解码后还有两条隐式拓扑约束：

$$
x_{ig} \le a_{ig}, \quad \forall i,g
$$

$$
x_{ig} \le y_i, \quad \forall i,g
$$

其中：
- `a_{ig}`：设备组兼容性参数

含义：
- 只有兼容液冷的设备组才能开启液冷
- 设备组液冷不能脱离机柜级液冷改造而独立存在

## 4. Oracle：拓扑-容量可行性筛查

外层最关键的“约束前筛”在 `FeasibilityOracle.assess()` 中实现。

代码位置：
- [src/feasibility_oracle.py:220](/d:/Myproject/src/feasibility_oracle.py:220)

## 4.1 液冷可承担热负荷包络

### 4.1.1 已知机柜 IT 负荷

代码位置：
- [src/feasibility_oracle.py](/d:/Myproject/src/feasibility_oracle.py)

$$
q^{IT}_{i}(k,t)=\widehat q^{IT}_{i}(k,t)
$$

含义：Oracle 不再求模型内负荷分配包络；每个机柜的 IT 热负荷直接来自典型日中的 `rack_it_*_kw` 时序。

### 4.1.2 液冷系数

代码位置：
- [src/feasibility_oracle.py:206](/d:/Myproject/src/feasibility_oracle.py:206)

对给定拓扑，代码构造：

$$
c_i = \sum_{g \in G} \eta_{ig} x_{ig}\lambda_{ig}
$$

则液冷热量可写成确定性机柜级映射：

$$
q_i^{LC}(k,t) = c_i \, \widehat q_i^{IT}(k,t)
$$

### 4.1.3 典型日逐时液冷包络

代码位置：
- [src/feasibility_oracle.py:118](/d:/Myproject/src/feasibility_oracle.py:118)
- [src/feasibility_oracle.py:169](/d:/Myproject/src/feasibility_oracle.py:169)

Oracle 对每个时段直接计算：

$$
q_{LC}(k,t)=\sum_{i \in R} c_i \, \widehat q_i^{IT}(k,t)
$$

当前实现中 `q_{LC,\min}` 与 `q_{LC,\max}` 都取该确定性液冷负荷序列，用于沿 CDU / HP / CRAC / Chiller / Tower 链做快速容量筛查。

这个确定性序列用于后续 CDU / HP / CRAC / Tower 的快速可行性判定。

## 4.2 CDU 约束

代码位置：
- [src/feasibility_oracle.py:233](/d:/Myproject/src/feasibility_oracle.py:233)

当前代码使用：

$$
Cap_{CDU} \ge \max_{k,t} q_{LC,\min}(k,t)
$$

若不满足且启用投影，则：

$$
Cap_{CDU} \leftarrow \text{clip}\left(\max_{k,t} q_{LC,\min}(k,t)\right)
$$

并记录违约量：

$$
v_{CDU} = \max\left(0,\; \max_{k,t} q_{LC,\min}(k,t) - Cap_{CDU}\right)
$$

## 4.3 低温链：HP + CRAC

代码位置：
- [src/feasibility_oracle.py:239](/d:/Myproject/src/feasibility_oracle.py:239)

当前口径下，低温侧原始热源基线取为外部机柜 IT 负荷求和：

$$
q_{LOW}^{base}(k,t) = q_{IT}^{total}(k,t)
$$

其中

$$
q_{IT}^{total}(k,t) = \sum_{i \in R} \widehat q_i^{IT}(k,t)
$$

不再额外叠加“低密机柜固定负荷”项。

先定义 CDU 可实际利用的液冷热量：

$$
q_{LC,used}^{max}(k,t) = \min\left(Cap_{CDU},\; q_{LC,\max}(k,t)\right)
$$

热泵蒸发侧最大可承接低品位热源：

$$
q_{LOW,HP}^{max} = Cap_{HP}\cdot \frac{COP_{HP}-1}{COP_{HP}}
$$

则 CRAC 至少要承接：

$$
q_{CRAC}^{lb}(k,t) = \max\left(0,\; q_{LOW}^{base}(k,t) - q_{LC,used}^{max}(k,t) - q_{LOW,HP}^{max}\right)
$$

并取峰值：

$$
q_{CRAC,\max}^{lb} = \max_{k,t} q_{CRAC}^{lb}(k,t)
$$

## 4.4 CRAC 拆除上限

代码位置：
- [src/feasibility_oracle.py:244](/d:/Myproject/src/feasibility_oracle.py:244)

剩余 CRAC 有效制冷能力为：

$$
Cap_{CRAC,eff} = \frac{(N_{CRAC}^{old} - n_{CRAC}^{remove}) \cdot Cap_{CRAC}^{unit}}{red_{CRAC}}
$$

则最大允许拆除台数满足：

$$
n_{CRAC}^{remove,max}
= \left\lfloor
N_{CRAC}^{old}
- \frac{red_{CRAC}\cdot q_{CRAC,\max}^{lb}}{Cap_{CRAC}^{unit}}
\right\rfloor
$$

代码中再做裁剪：

$$
0 \le n_{CRAC}^{remove,max} \le N_{CRAC}^{old}
$$

若启用投影：

$$
n_{CRAC}^{remove,fixed} = \min(n_{CRAC}^{remove,raw},\; n_{CRAC}^{remove,max})
$$

违约量：

$$
v_{CRAC} = \max(0,\; q_{CRAC,\max}^{lb} - Cap_{CRAC,eff})
$$

## 4.5 HP 补偿投影

代码位置：
- [src/feasibility_oracle.py:257](/d:/Myproject/src/feasibility_oracle.py:257)

若 `CRAC` 仍不足，代码会反向推所需热泵容量：

$$
q_{LOW,HP}^{req}
= \max_{k,t}\max\left(0,\; q_{LOW}^{base}(k,t) - q_{LC,used}^{max}(k,t) - Cap_{CRAC,eff}\right)
$$

$$
Cap_{HP}^{req} = q_{LOW,HP}^{req}\cdot \frac{COP_{HP}}{COP_{HP}-1}
$$

若启用投影：

$$
Cap_{HP} \leftarrow \max(Cap_{HP},\; Cap_{HP}^{req})
$$

## 4.6 Chiller 约束

代码位置：
- [src/feasibility_oracle.py:274](/d:/Myproject/src/feasibility_oracle.py:274)

代码按内层物理关系推导冷机最低负荷：

$$
q_{CH}^{lb}(k,t) = q_{CRAC}^{lb}(k,t)\left(1+\frac{1}{COP_{CRAC}}\right)
$$

$$
q_{CH,\max}^{lb} = \max_{k,t} q_{CH}^{lb}(k,t)
$$

对应违约量：

$$
v_{CH} = \max(0,\; q_{CH,\max}^{lb} - Cap_{CH}^{old})
$$

## 4.7 Cooling Tower 约束

代码位置：
- [src/feasibility_oracle.py:278](/d:/Myproject/src/feasibility_oracle.py:278)

塔侧最低需求：

$$
q_{TW}^{lb}(k,t) = q_{CH}^{lb}(k,t)\left(1+\frac{1}{COP_{CH}}\right)
$$

$$
q_{TW,\max}^{lb} = \max_{k,t} q_{TW}^{lb}(k,t)
$$

总塔有效能力：

$$
Cap_{TW,eff} = \frac{Cap_{TW}^{old} + \Delta Cap_{TW}}{red_{TW}}
$$

违约量：

$$
v_{TW} = \max(0,\; q_{TW,\max}^{lb} - Cap_{TW,eff})
$$

若启用投影，则要求：

$$
\Delta Cap_{TW}^{req} = \max(0,\; red_{TW}\cdot q_{TW,\max}^{lb} - Cap_{TW}^{old})
$$

并令：

$$
\Delta Cap_{TW} \leftarrow \max(\Delta Cap_{TW},\; \Delta Cap_{TW}^{req})
$$

## 4.8 Oracle 可行性判定

代码位置：
- [src/feasibility_oracle.py:291](/d:/Myproject/src/feasibility_oracle.py:291)

总违约量：

$$
V^{oracle} = v_{CDU} + v_{CRAC} + v_{CH} + v_{TW}
$$

可行判定：

$$
\text{oracle\_feasible} =
\begin{cases}
1, & V^{oracle} \le \varepsilon_{oracle} \\
0, & \text{otherwise}
\end{cases}
$$

## 5. Oracle 不可行时的外层跳过规则

代码位置：
- [src/model.py:590](/d:/Myproject/src/model.py:590)
- [src/model.py:594](/d:/Myproject/src/model.py:594)

如果：

$$
\text{oracle\_feasible} = 0
$$

且配置：

$$
\text{skip\_inner\_on\_infeasible} = 1
$$

则当前方案不会进入 `inner_dispatch`，直接返回：

$$
TLCC = +\infty,\quad TCE = +\infty
$$

同时附加罚项：

$$
Penalty = 10^9 + 10^6 \cdot V^{oracle}
$$

## 6. 外层后处理工程约束

这些约束当前不是通过独立优化器显式建模，而是在 `evaluate_solution()` 中以超限量罚项方式进入目标。

## 6.1 机房液冷辅助设备占地约束

代码位置：
- [src/model.py:647](/d:/Myproject/src/model.py:647)

当前代码写法：

$$
A^{use}
= \rho_{CDU}^{area} Cap_{CDU}
 + \rho_{HX}^{area} Cap_{HX}
 - a_{CRAC}^{unit} n_{CRAC}^{remove}
$$

$$
A^{over} = \max(0,\; A^{use} - A^{room,avail})
$$

罚项：

$$
Penalty \mathrel{+}= 2\times 10^6 \cdot A^{over}
$$

含义：新增 CDU、板换占地，拆除 CRAC 释放面积。

## 6.2 机柜静载约束

代码位置：
- [src/model.py:651](/d:/Myproject/src/model.py:651)

单机柜重量：

$$
W_i = w_i^{old} + w_i^{fix} y_i + \sum_g w_{ig}^{plate} x_{ig}
$$

机柜静载超限量：

$$
V^{rack} = \max_i \max(0,\; W_i - W_i^{rack,max})
$$

## 6.3 机柜点载约束

代码位置：
- [src/model.py:653](/d:/Myproject/src/model.py:653)

点载超限量：

$$
V^{point} = \max_i \max(0,\; W_i - W_i^{point,max})
$$

与静载一起计罚：

$$
Penalty \mathrel{+}= 2\times 10^6 \cdot (V^{rack} + V^{point})
$$

## 6.4 分区楼板承重约束

代码位置：
- [src/model.py:658](/d:/Myproject/src/model.py:658)
- [src/model.py:665](/d:/Myproject/src/model.py:665)

每个热区楼板承重上限：

$$
W_z^{floor,max} = FloorLoad_z^{max} \cdot Area_z
$$

区内总重量：

$$
W_z^{use}
= \sum_{i \in R_z}\left(
w_i^{old} + w_i^{fix} y_i + \sum_g w_{ig}^{plate} x_{ig}
\right)
$$

若 `cdu_zone = z_cdu`，则额外加入 CDU 重量：

$$
W_{z_cdu}^{use} \mathrel{+}= \rho_{CDU}^{weight} Cap_{CDU}
$$

总楼板超限量：

$$
V^{floor} = \sum_z \max(0,\; W_z^{use} - W_z^{floor,max})
$$

罚项：

$$
Penalty \mathrel{+}= 2\times 10^6 \cdot V^{floor}
$$

## 7. 外层目标中与约束有关的罚项汇总

代码位置：
- [src/model.py:636](/d:/Myproject/src/model.py:636)

当前总罚项包含：

$$
Penalty
= 10^9 \cdot \mathbb{I}[\text{inner infeasible}]
$$

$$
Penalty \mathrel{+}= 10^9 + 10^6 V^{oracle}, \quad \text{if } V^{oracle} > \varepsilon_{oracle}
$$

$$
Penalty \mathrel{+}= 2\times 10^6 A^{over}
$$

$$
Penalty \mathrel{+}= 2\times 10^6 (V^{rack} + V^{point})
$$

$$
Penalty \mathrel{+}= 2\times 10^6 V^{floor}
$$

这些罚项同时加到：

$$
TLCC = C^{annual}_{capex} + C^{annual}_{fom} + C^{annual}_{op} + Penalty
$$

$$
TCE = EI^{annual} + CO_2^{annual,op} + Penalty
$$

## 8. 最终可行性判定

代码位置：
- [src/model.py:747](/d:/Myproject/src/model.py:747)
- [src/model.py:748](/d:/Myproject/src/model.py:748)

当前输出中：

$$
ConstraintViolation = V^{oracle} + \mathbb{I}[\text{inner infeasible}]
$$

最终布尔可行性：

$$
is\_feasible =
\text{inner\_feasible}
\land \text{oracle\_feasible}
\land \text{finite}(TLCC)
\land \text{finite}(TCE)
$$

注意：  
当前 `space_over / rack_over / point_over / floor_over` 进入了罚项，但**没有**直接并入 `constraint_violation` 或 `is_feasible` 的逻辑判定。  
也就是说，它们在当前实现里是“高罚项工程约束”，不是“直接把方案标成不可行”的硬开关。

## 9. 外层审核重点

建议你重点检查以下问题：

- `x_{ig} \le a_{ig}` 与 `x_{ig} \le y_i` 是否符合你的机柜-设备组物理定义
- Oracle 中 `q_{LC,\min}` / `q_{LC,\max}` 的使用方向是否符合你对保守性的要求
- `n_{CRAC}^{remove,max}` 的推导是否与你的冷量冗余设计一致
- `Cap_{HP}`、`\Delta Cap_{Tower}` 的投影补偿是否会改变你想要的“外层自主决策”语义
- 占地/承重约束目前是罚项，不是硬不可行；如果你希望它们是绝对硬约束，需要再改实现

## 10. 备注

- 本文档描述的是**当前代码实现**，不是理论模型的标准答案。
- 外层是“启发式搜索 + 可行性筛查 + 罚项约束”的混合结构，不能直接按单一数学规划模型去理解。
- 若后续你继续修改 `model.py` 或 `feasibility_oracle.py`，建议同步更新本文件。
