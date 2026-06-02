# `inner_dispatch.py` 约束审查整理

本文档按 `src/inner_dispatch.py` 当前实现整理，用于人工审核内层调度 MILP 的约束是否与预期数学模型一致。  
除特别说明外，下文只描述**已进入 Gurobi 模型**的约束，不包含求解后才计算的诊断指标。

对应源码主位置：
- [src/inner_dispatch.py](/d:/Myproject/src/inner_dispatch.py)

## 1. 范围与索引

- `i \in R`：机柜
- `g \in G`：设备组
- `z \in Z`：热区
- `b \in B`：业务类型
- `k \in K`：典型日场景
- `t \in T`：场景内小时

当前内层模型是“典型日场景 × 小时”的调度 MILP。  
其中 `Q_{Heat,Demand}(k,t)` 来自外部时序输入参数 `heat_demand_kw`。

## 2. 主要变量

### 2.1 IT 与冷却分配

- `q^{IT}_{ikt}`：机柜 IT 热负荷
- `q^{LC}_{ikt}`：机柜液冷带走热量
- `q^{AIR}_{ikt}`：机柜风冷带走热量
- `s^{LC}_{ikt}`：机柜液冷供给能力
- `s^{AIR}_{zkt}`：热区风冷供给能力
- `\theta_{ikt}`：机柜温度状态
- `\xi_{ikt}`：热点温升软惩罚变量

### 2.2 电功率

- `p^{grid}_{kt}`：电网购电
- `p^{bch}_{kt}, p^{bdis}_{kt}`：BESS 充/放电功率
- `p^{HP}_{kt}`：热泵耗电
- `p^{CRAC}_{kt}`：CRAC 耗电
- `p^{CH}_{kt}`：冷水机组耗电
- `p^{CDU}_{kt}`：CDU 耗电
- `p^{TW}_{kt}`：冷却塔耗电

### 2.3 热量流

- `q^{HP,out}_{kt}`：热泵供热输出
- `q^{TES,ch}_{kt}, q^{TES,dis}_{kt}`：TES 充/放热
- `q^{MID,HX}_{kt}`：中品位热源进入板换部分
- `q^{MID,waste}_{kt}`：中品位热源弃热部分
- `q^{LOW,HP}_{kt}`：低品位热源进入热泵蒸发侧部分
- `q^{CRAC}_{kt}`：低品位热源转入 CRAC 部分
- `q^{CH}_{kt}`：冷机侧热负荷
- `q^{CH,TW}_{kt}`：冷机传给塔侧负荷
- `q^{TW,tot}_{kt}`：冷却塔总散热负荷
- `q^{MID,src}_{kt}`：中品位热源总量
- `q^{LOW,src}_{kt}`：低品位热源总量
- `q^{Heat,Supply}_{kt}`：用户侧供热量
- `q^{direct}_{kt}`：板换直供热量

### 2.4 储能状态

- `e^{TES}_{kt}`：TES 储热状态
- `e^{BESS}_{kt}`：BESS 电量状态
- `d^{TES,ch}_{kt}, d^{TES,dis}_{kt}`：TES 充/放热二进制
- `d^{BCH}_{kt}, d^{BDIS}_{kt}`：BESS 充/放电二进制

### 2.5 负载率/PWL 变量

- `\phi^{CDU}_{kt}`：CDU 负载率
- `\phi^{TW}_{kt}`：冷却塔负载率
- `p^{CDU,curve}_{kt}`：CDU 变负载附加功率
- `p^{TW,curve}_{kt}`：冷却塔变负载附加功率

### 2.6 诊断松弛变量（仅在 `diag_slack_enabled=True` 时存在）

- `sl^{CDU}_{kt}, sl^{CRAC}_{kt}, sl^{CH}_{kt}, sl^{TW}_{kt}`
- `sl^{P,+}_{kt}, sl^{P,-}_{kt}`

## 3. 约束整理

## 3.1 已知机柜 IT 负荷输入与热负荷映射

代码位置：
- [src/inner_dispatch.py](/d:/Myproject/src/inner_dispatch.py)

$$
q^{IT}_{ikt} = \widehat q^{IT}_{ikt}, \quad \forall i,k,t
$$

$$
q^{LC}_{ikt} = \left(\sum_{g \in G} \eta_{ig} x_{ig}\lambda_{ig}\right) q^{IT}_{ikt}, \quad \forall i,k,t
$$

$$
q^{AIR}_{ikt} = q^{IT}_{ikt} - q^{LC}_{ikt}, \quad \forall i,k,t
$$

含义：机柜 IT 热负荷由外部时序参数 `rack_it_*_kw` 固定输入，不再由内层模型优化分配；`\lambda_{ig}` 表示机柜内服务器类型/设备组热负荷占比。代码中已将设备组热负荷中间变量代数消去，直接使用合并系数计算液冷热负荷。

## 3.3 液冷支路与 CDU 容量

代码位置：
- [src/inner_dispatch.py:305](/d:/Myproject/src/inner_dispatch.py:305)
- [src/inner_dispatch.py:306](/d:/Myproject/src/inner_dispatch.py:306)
- [src/inner_dispatch.py:313](/d:/Myproject/src/inner_dispatch.py:313)

$$
q^{LC}_{ikt} \le s^{LC}_{ikt}, \quad \forall i,k,t
$$

$$
s^{LC}_{ikt} \le Cap^{branch}_i \, y_i, \quad \forall i,k,t
$$

若开启诊断松弛：
$$
\sum_{i \in R} s^{LC}_{ikt} \le Cap_{CDU} + sl^{CDU}_{kt}, \quad \forall k,t
$$

否则：
$$
\sum_{i \in R} s^{LC}_{ikt} \le Cap_{CDU}, \quad \forall k,t
$$

含义：机柜支路受本机柜液冷改造决策和支路能力限制，总液冷供给受 CDU 总容量限制。

## 3.4 风冷热区能力

代码位置：
- [src/inner_dispatch.py:319](/d:/Myproject/src/inner_dispatch.py:319)
- [src/inner_dispatch.py:320](/d:/Myproject/src/inner_dispatch.py:320)

$$
\sum_{i \in R_z} q^{AIR}_{ikt} \le s^{AIR}_{zkt}, \quad \forall z,k,t
$$

$$
s^{AIR}_{zkt} \le Cap^{AIR}_{zkt}, \quad \forall z,k,t
$$

含义：每个热区的风冷需求不能超过该热区时变风冷能力。

## 3.5 热源总量汇总

代码位置：
- [src/inner_dispatch.py:324](/d:/Myproject/src/inner_dispatch.py:324)
- [src/inner_dispatch.py:325](/d:/Myproject/src/inner_dispatch.py:325)

$$
q^{MID,src}_{kt} = \sum_{i \in R} q^{LC}_{ikt}, \quad \forall k,t
$$

$$
q^{LOW,src}_{kt} = \sum_{i \in R} q^{AIR}_{ikt}, \quad \forall k,t
$$

含义：
- 中品位热源来自液冷带走的热量
- 低品位热源来自未被液冷支路带走的 IT 总热负荷

## 3.6 热拓扑与热点约束

代码位置：
- [src/inner_dispatch.py:332](/d:/Myproject/src/inner_dispatch.py:332)
- [src/inner_dispatch.py:340](/d:/Myproject/src/inner_dispatch.py:340)
- [src/inner_dispatch.py:345](/d:/Myproject/src/inner_dispatch.py:345)

若 `use_hji=True`：
$$
\theta_{ikt}
= T^{sup}_{z(i)kt}
 + \sum_{j \in R} H_{ji} q^{AIR}_{jkt}
 - \beta_i s^{AIR}_{z(i)kt},
\quad \forall i,k,t
$$

若 `use_hji=False`：
$$
\theta_{ikt}
= T^{sup}_{z(i)kt}
 - \beta_i s^{AIR}_{z(i)kt},
\quad \forall i,k,t
$$

共同的热点约束：
$$
\theta_{ikt} \le T^{max}_i + \xi_{ikt}, \quad \forall i,k,t
$$

含义：热点温度由送风温度、风冷负荷耦合项和风量缓解项共同决定，`\xi` 为热惩罚软变量。

## 3.7 中品位热源路径：直供 + 弃热

代码位置：
- [src/inner_dispatch.py:357](/d:/Myproject/src/inner_dispatch.py:357)
- [src/inner_dispatch.py:358](/d:/Myproject/src/inner_dispatch.py:358)
- [src/inner_dispatch.py:359](/d:/Myproject/src/inner_dispatch.py:359)

$$
q^{MID,src}_{kt} = q^{MID,HX}_{kt} + q^{MID,waste}_{kt}, \quad \forall k,t
$$

$$
q^{MID,HX}_{kt} \le Cap_{HX}, \quad \forall k,t
$$

$$
q^{direct}_{kt} = \eta_{HX} \, q^{MID,HX}_{kt}, \quad \forall k,t
$$

含义：液冷中品位热源可走两条路：
- 进入板换后直供用户侧
- 不利用时进入塔侧散热

## 3.8 低品位热源路径：热泵 + CRAC

代码位置：
- [src/inner_dispatch.py:361](/d:/Myproject/src/inner_dispatch.py:361)
- [src/inner_dispatch.py:362](/d:/Myproject/src/inner_dispatch.py:362)
- [src/inner_dispatch.py:363](/d:/Myproject/src/inner_dispatch.py:363)
- [src/inner_dispatch.py:364](/d:/Myproject/src/inner_dispatch.py:364)
- [src/inner_dispatch.py:366](/d:/Myproject/src/inner_dispatch.py:366)
- [src/inner_dispatch.py:367](/d:/Myproject/src/inner_dispatch.py:367)
- [src/inner_dispatch.py:368](/d:/Myproject/src/inner_dispatch.py:368)
- [src/inner_dispatch.py:374](/d:/Myproject/src/inner_dispatch.py:374)
- [src/inner_dispatch.py:375](/d:/Myproject/src/inner_dispatch.py:375)

### 3.8.1 低品位热源分流

$$
q^{LOW,src}_{kt} = q^{LOW,HP}_{kt} + q^{CRAC}_{kt}, \quad \forall k,t
$$

### 3.8.2 热泵提升与容量

$$
q^{HP,out}_{kt} = q^{LOW,HP}_{kt} \cdot \frac{COP_{HP}}{COP_{HP}-1}, \quad \forall k,t
$$

$$
q^{HP,out}_{kt} \le Cap_{HP}, \quad \forall k,t
$$

$$
p^{HP}_{kt} = \frac{q^{HP,out}_{kt}}{COP_{HP}}, \quad \forall k,t
$$

### 3.8.3 CRAC / 冷机链

$$
p^{CRAC}_{kt} = \frac{q^{CRAC}_{kt}}{COP_{CRAC}}, \quad \forall k,t
$$

$$
q^{CH}_{kt} = q^{CRAC}_{kt} + p^{CRAC}_{kt}, \quad \forall k,t
$$

$$
p^{CH}_{kt} = \frac{q^{CH}_{kt}}{COP_{CH}}, \quad \forall k,t
$$

若开启诊断松弛：
$$
q^{CRAC}_{kt} \le \frac{Cap^{CRAC,old}}{red_{CRAC}} + sl^{CRAC}_{kt}, \quad \forall k,t
$$

$$
q^{CH}_{kt} \le \frac{Cap^{CH,old}}{red_{CH}} + sl^{CH}_{kt}, \quad \forall k,t
$$

否则：
$$
q^{CRAC}_{kt} \le \frac{Cap^{CRAC,old}}{red_{CRAC}}, \quad \forall k,t
$$

$$
q^{CH}_{kt} \le \frac{Cap^{CH,old}}{red_{CH}}, \quad \forall k,t
$$

含义：低品位热源的一部分通过热泵提温，剩余部分走风冷 CRAC-冷机链。

## 3.9 塔侧热平衡与容量

代码位置：
- [src/inner_dispatch.py:377](/d:/Myproject/src/inner_dispatch.py:377)
- [src/inner_dispatch.py:378](/d:/Myproject/src/inner_dispatch.py:378)
- [src/inner_dispatch.py:382](/d:/Myproject/src/inner_dispatch.py:382)

$$
q^{CH,TW}_{kt} = q^{CH}_{kt} + p^{CH}_{kt}, \quad \forall k,t
$$

$$
q^{TW,tot}_{kt} = q^{CH,TW}_{kt} + q^{MID,waste}_{kt}, \quad \forall k,t
$$

记：
$$
Cap^{TW,tot} = Cap^{TW,old} + Cap_{TW}
$$

若开启诊断松弛：
$$
q^{TW,tot}_{kt} \le \frac{Cap^{TW,tot}}{red_{TW}} + sl^{TW}_{kt}, \quad \forall k,t
$$

否则：
$$
q^{TW,tot}_{kt} \le \frac{Cap^{TW,tot}}{red_{TW}}, \quad \forall k,t
$$

含义：冷机与中品位弃热最终都汇入塔侧，由冷却塔总能力承接。

## 3.10 负载率定义

代码位置：
- [src/inner_dispatch.py:384](/d:/Myproject/src/inner_dispatch.py:384)
- [src/inner_dispatch.py:385](/d:/Myproject/src/inner_dispatch.py:385)

$$
q^{MID,src}_{kt} = Cap_{CDU} \, \phi^{CDU}_{kt}, \quad \forall k,t
$$

$$
q^{TW,tot}_{kt} = Cap^{TW,tot} \, \phi^{TW}_{kt}, \quad \forall k,t
$$

含义：将 CDU 和冷却塔热负荷归一化成负载率变量，为后续 PWL 功率曲线服务。

## 3.11 TES 约束

代码位置：
- [src/inner_dispatch.py:397](/d:/Myproject/src/inner_dispatch.py:397)
- [src/inner_dispatch.py:404](/d:/Myproject/src/inner_dispatch.py:404)
- [src/inner_dispatch.py:405](/d:/Myproject/src/inner_dispatch.py:405)
- [src/inner_dispatch.py:406](/d:/Myproject/src/inner_dispatch.py:406)
- [src/inner_dispatch.py:407](/d:/Myproject/src/inner_dispatch.py:407)

记 `t^-` 表示上一时段，且代码中采用循环闭合：
$$
t^- = (t-1)\bmod |T|
$$

### 3.11.1 储热状态递推

$$
e^{TES}_{kt}
= e^{TES}_{k,t^-}(1-\sigma^{TES})
 + \eta^{TES,ch} q^{TES,ch}_{kt}
 - \frac{q^{TES,dis}_{kt}}{\eta^{TES,dis}},
\quad \forall k,t
$$

### 3.11.2 储热容量上限

$$
e^{TES}_{kt} \le Cap_{TES}, \quad \forall k,t
$$

并且 `e^{TES}_{kt} \ge 0` 由变量下界给出。

### 3.11.3 充放热功率与模式

$$
q^{TES,ch}_{kt} \le \gamma^{TES} Cap_{TES} \, d^{TES,ch}_{kt}, \quad \forall k,t
$$

$$
q^{TES,dis}_{kt} \le \gamma^{TES} Cap_{TES} \, d^{TES,dis}_{kt}, \quad \forall k,t
$$

$$
d^{TES,ch}_{kt} + d^{TES,dis}_{kt} \le 1, \quad \forall k,t
$$

含义：TES 不能同一时段既充又放，充/放热功率受 C-rate 与容量共同限制。

## 3.12 BESS 约束

代码位置：
- [src/inner_dispatch.py:409](/d:/Myproject/src/inner_dispatch.py:409)
- [src/inner_dispatch.py:416](/d:/Myproject/src/inner_dispatch.py:416)
- [src/inner_dispatch.py:417](/d:/Myproject/src/inner_dispatch.py:417)
- [src/inner_dispatch.py:418](/d:/Myproject/src/inner_dispatch.py:418)
- [src/inner_dispatch.py:419](/d:/Myproject/src/inner_dispatch.py:419)
- [src/inner_dispatch.py:420](/d:/Myproject/src/inner_dispatch.py:420)

$$
e^{BESS}_{kt}
= e^{BESS}_{k,t^-}(1-\sigma^{BESS})
 + \eta^{BESS,ch} p^{bch}_{kt}
 - \frac{p^{bdis}_{kt}}{\eta^{BESS,dis}},
\quad \forall k,t
$$

$$
0.1\,Cap^{E}_{BESS} \le e^{BESS}_{kt} \le 0.9\,Cap^{E}_{BESS}, \quad \forall k,t
$$

$$
p^{bch}_{kt} \le Cap^{P}_{BESS}\, d^{BCH}_{kt}, \quad \forall k,t
$$

$$
p^{bdis}_{kt} \le Cap^{P}_{BESS}\, d^{BDIS}_{kt}, \quad \forall k,t
$$

$$
d^{BCH}_{kt} + d^{BDIS}_{kt} \le 1, \quad \forall k,t
$$

## 3.13 电功率平衡

代码位置：
- [src/inner_dispatch.py:427](/d:/Myproject/src/inner_dispatch.py:427)
- [src/inner_dispatch.py:433](/d:/Myproject/src/inner_dispatch.py:433)

先记总 IT 功率：
$$
p^{IT}_{kt} = \sum_{i \in R} q^{IT}_{ikt}
$$

总冷却功率：
$$
p^{cool}_{kt} = p^{CRAC}_{kt} + p^{CH}_{kt} + p^{CDU}_{kt} + p^{TW}_{kt}
$$

若开启诊断松弛：
$$
p^{grid}_{kt} + p^{bdis}_{kt}
= p^{IT}_{kt} + p^{cool}_{kt} + p^{HP}_{kt} + p^{bch}_{kt}
 + sl^{P,+}_{kt} - sl^{P,-}_{kt},
\quad \forall k,t
$$

否则：
$$
p^{grid}_{kt} + p^{bdis}_{kt}
= p^{IT}_{kt} + p^{cool}_{kt} + p^{HP}_{kt} + p^{bch}_{kt},
\quad \forall k,t
$$

## 3.14 供热平衡与热需求满足

代码位置：
- [src/inner_dispatch.py:435](/d:/Myproject/src/inner_dispatch.py:435)
- [src/inner_dispatch.py:439](/d:/Myproject/src/inner_dispatch.py:439)

根据当前代码实现，供热侧使用的是你文档中最新版本的两条式子：

$$
q^{direct}_{kt} + q^{HP,out}_{kt} + q^{TES,dis}_{kt}
= q^{Heat,Supply}_{kt} + q^{TES,ch}_{kt},
\quad \forall k,t
$$

$$
q^{Heat,Supply}_{kt} \ge Q_{Heat,Demand}(k,t),
\quad \forall k,t
$$

其中：
- 左侧：系统当期可用热源
- 右侧：用户侧供热量与 TES 充热消耗
- `Q_{Heat,Demand}(k,t)`：外部时序输入参数

## 3.15 CDU 与冷却塔 PWL 功率曲线

### 3.15.1 CDU PWL

代码位置：
- [src/inner_dispatch.py:468](/d:/Myproject/src/inner_dispatch.py:468)
- [src/inner_dispatch.py:469](/d:/Myproject/src/inner_dispatch.py:469)

代码使用分段线性约束：
$$
p^{CDU,curve}_{kt} = f_{CDU}(\phi^{CDU}_{kt}), \quad \forall k,t
$$

其中 `f_CDU` 由以下采样点构建：
$$
f_{CDU}(x) = P^{CDU}_{rated}\left(a_1x + a_2x^2 + a_3x^3\right)
$$

再叠加待机功率：
$$
p^{CDU}_{kt} = P^{CDU}_{idle} + p^{CDU,curve}_{kt}, \quad \forall k,t
$$

### 3.15.2 冷却塔 PWL

代码位置：
- [src/inner_dispatch.py:477](/d:/Myproject/src/inner_dispatch.py:477)
- [src/inner_dispatch.py:478](/d:/Myproject/src/inner_dispatch.py:478)

先按湿球温度修正效率：
$$
\eta^{wb}_{kt} = \max\left(0.55,\;1-\alpha_{wb}(T^{wb}_{kt}-T^{wb}_{ref})\right)
$$

再构造分段线性曲线：
$$
p^{TW,curve}_{kt} = f_{TW}(\phi^{TW}_{kt}), \quad \forall k,t
$$

其中采样母式为：
$$
f_{TW}(x) = P^{TW}_{fan,rated}\left(\frac{x}{\eta^{wb}_{kt}}\right)^3
$$

最后：
$$
p^{TW}_{kt} = P^{TW}_{idle} + p^{TW,curve}_{kt}, \quad \forall k,t
$$

## 4. 审核提醒

审核 `inner_dispatch.py` 时，建议重点看这几类是否与你的总模型说明一致：

- `3.14` 供热平衡与热需求满足的两条式子
- `3.7` 与 `3.8` 是否保持“中品位直供路径 / 低品位热泵路径”分离
- `3.11` TES 状态递推是否与你文档中的储热定义一致
- `3.15` PWL 功率曲线是否只用于耗电计算，而不改变主热量守恒
- 所有 `diag_slack_*` 是否仅为诊断松弛，不应被误读为正常物理能力

## 5. 备注

- 本文档描述的是**当前代码实现**，不是对理论模型的二次解释。
- 如果你后续继续改 `inner_dispatch.py`，建议同步更新本文件，避免“代码和审查稿”再次偏离。
