# A股配对研究方法

## 目录

1. 数据口径与时期
2. 同行业候选与协整
3. 多重检验
4. 形成期共线去重
5. 滚动信号
6. 两腿损益、成本与风控
7. 研究门控与复现
8. 现实限制

## 1. 数据口径与时期

协整只能使用复权后的正价格。默认数据源是 panda_data 的 `get_stock_daily_pre`，即前复权日线；显式选择后复权时使用 `get_stock_daily_post`。普通未复权 `close` 不进入本研究。

把时间分成三段：

- `data_period`：包含形成期缓冲与完整评估期的数据请求区间；
- `formation_period`：评估期之前最后252个有效交易日；
- `evaluation_period`：`--years` 指定的完整样本外区间。

例如截止2026-07-10、`--years 5`，评估期从2021-07-11起算。程序再向前取至少400个自然日，随后从实际交易日中选出最后252日作为形成期。绩效只计算评估期，形成期不会以零收益混入。

历史股票池使用数据起点附近可得的指数成分。行业接口没有历史日期参数，因此行业分类仍有残余前视偏差，必须在报告中披露。行业查询对瞬时错误逐只最多重试3次；最终覆盖率低于95%时停止回测，并在诊断中记录成功数、未知数和覆盖率。

## 2. 同行业候选与协整

先在形成期内按一级行业分组，再要求两只股票的对数价格相关性达到阈值。行业未知或跨行业的组合直接丢弃。

对每个候选运行 Engle–Granger 两阶段检验：

$$
\log P^A_t=\alpha+\beta\log P^B_t+\varepsilon_t
$$

第一步估计 $\alpha,\beta$；第二步检验残差 $\varepsilon_t$ 是否平稳。输出保留原始 `pvalue`，不能用调整后数值覆盖它。

用下式估计半衰期：

$$
\Delta\varepsilon_t=a+b\varepsilon_{t-1}+u_t,\qquad
t_{1/2}=\frac{\log 2}{-b}.
$$

默认只保留2–60个交易日的有限半衰期。

## 3. Benjamini–Hochberg 多重检验

同时测试数百个候选时，仅看 `p<0.05` 会积累大量偶然显著结果。对同一次形成期候选家族执行 Benjamini–Hochberg：

1. 对 $m$ 个原始p值从小到大排序；
2. 计算 $p_{(i)}m/i$；
3. 从后向前取累计最小值，得到单调的调整值；
4. 把调整值放回原顺序，记为 `pvalue_fdr`。

默认同时要求：

```text
pvalue < 0.05
pvalue_fdr <= 0.05
2 <= half_life <= 60
```

报告和代表配对文件同时保留原始p值与FDR值。

## 4. 形成期共线去重

不要在回测后用全样本收益选对。对通过统计筛选的候选，仅用形成期构造两腿收益代理：

$$
r^{pair}_t=\frac{r^A_t-\beta r^B_t}{1+|\beta|}.
$$

步骤如下：

1. 标准化各候选的形成期两腿收益；
2. 用奇异值计算主成分解释率、有效秩和条件数；
3. 用 $1-|\rho|$ 作为距离做平均连接层次聚类；
4. 每簇选择 `pvalue_fdr` 最小、再按原始p值排序的代表对；
5. 按q值从小到大贪心执行 `max_pairs_per_symbol`，默认同一股票最多进入1对。

`formation_collinearity.json` 报告代表配对、单股出现次数、最大重复次数和有效独立对数。有效独立对数不会把多个共享股票的组合机械地视为独立头寸。

## 5. 滚动信号

形成期只负责初始选对。进入评估期后，每60个交易日：

1. 用此前252个交易日重新估计协整关系；
2. 重新检查原始p值与半衰期；
3. 关系有效时计算静态或Kalman价差；
4. 在当前60日区间内交易；
5. 区间末日先关闭已有仓位，禁止当日新开随后无法计价的仓位。

默认信号：

```text
入场：|z| > 2.0
均值回归出场：|z| < 0.5
关系停止：|z| > 4.0
时间停止：持有60日
峰值回撤停止：当前净收益 - 持仓期峰值净收益 <= -5%
```

Kalman只改变信号使用的动态 $\alpha,\beta$；不得把滤波创新或价差变化直接当成可交易收益。

## 6. 两腿损益、成本与风控

入场后固定对冲比例。对数收益按总敞口归一化：

$$
r_t^{gross}=
\frac{s(r^A_t-\beta r^B_t)}{1+|\beta|},
$$

其中 $s=+1$ 表示多A、空 $\beta$ 单位B；$s=-1$ 表示相反方向。

每日分别记录A腿与B腿贡献。净收益扣除：

- 默认单边换手成本7.5bp，完整开平合计15bp；
- 空头权重对应的每日融券持有费，默认年费800bp。

每笔交易输出 `leg_a_ret`、`leg_b_ret`、`transaction_cost`、`borrow_cost`、`ret_gross`、`ret_net`、`peak_net_pnl` 和 `max_drawdown_from_peak`，保证两腿、成本和净收益可核对。

## 7. 研究门控与复现

门控只允许以下两种顶层状态：

- `NO_TRADE_*`：保持零仓位，默认输出不得显示 LONG/SHORT；
- `RESEARCH_PASS`：输出仍标为 `RESEARCH_ONLY`，只允许继续模拟。

`run_id` 绑定：

- 指数池、三段时期、全部筛选和风控参数；
- 成本、融券年费、复权口径、方法与代码版本；
- 形成期代表配对摘要；
- 最终门控结论、绩效与交易统计；
- `price_panel_sha256`、`universe_sha256`、`industry_map_sha256`。
- 可出方向代表对的共享400日价格与行业快照。

手工只改报告中的部分字段会使 `signal` 阻断方向。`run_id` 是公开的完整性摘要，不是数字签名；它能发现误改和内部不一致，但无法阻止有意同时重写整份报告及其摘要。报告必须来自可信的本地回测；若需要防恶意篡改，应使用只读归档或外部签名。不同配置使用不同 `config_id` 目录，避免结果覆盖；同日同配置重跑前应归档旧目录。

`signal` 只接收报告内已有回测记录的代表对。未经过该报告形成期FDR、聚类和重复限制的新配对不能借用整体 `RESEARCH_PASS`。信号重新获取这些代表对的同一400日价格与行业窗口；缺失代表股票、全空价格列、行业覆盖不足或指纹变化时输出 `NO_TRADE_DATA_MISMATCH`，保存无方向CSV并停止筛选。完整历史指纹只证明报告所用输入，不声称信号阶段重新下载了完整回测历史。

## 8. 现实限制

- A股个股做空依赖券源；逐日券源、召回和强平未模拟。
- T+1、涨跌停、停牌和成交冲击未完整模拟。
- 融券年费只是参数假设，不能代表真实可成交费率。
- 股指期货无法精确替代个股空腿，不应描述为等价执行。
- 历史行业分类不可得，仍有残余偏差。
- 数据供应商可能修订历史复权值；同一日期但指纹不同，视为不同研究输入。

这些限制意味着任何正收益、正夏普或 `RESEARCH_PASS` 都不构成实盘许可。

## 9. 方法来源

- Engle, R. F., & Granger, C. W. J. (1987). *Co-integration and Error Correction: Representation, Estimation, and Testing*. Econometrica, 55(2), 251–276.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society, Series B, 57(1), 289–300.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*. The Review of Financial Studies, 19(3), 797–827.

本项目根据上述公开方法独立实现，不声称复现 Jim Simons、Renaissance Technologies 或任何私有交易系统。
