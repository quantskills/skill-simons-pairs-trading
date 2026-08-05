# A 股协整配对交易研究

> 从同行业股票里寻找相对稳定的价差关系，并用滚动样本外回测检验它是否真的站得住。

![CI](https://github.com/quantskills/skill-simons-pairs-trading/actions/workflows/ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB)
![License](https://img.shields.io/badge/License-MIT-green)

## 这个仓库做什么

配对交易看起来简单：买入相对便宜的一只，同时卖出相对贵的一只，等价差回归。但真正容易出错的地方在前面——配对是不是事后挑出来的，协整关系是否还在，多次检验有没有制造“显著性”，成本和做空限制有没有漏算。

这个仓库从沪深 300 和中证 500 的历史成分开始，只在同行业内找候选，再用复权日线、Engle–Granger 检验、FDR、半衰期和共线去重缩小范围。入选配对随后进入滚动样本外回测，两条腿的收益、对冲比例、交易成本、融券费、回撤和退出原因都会单独记录。

“Simons”只表示统计套利的研究风格。这里没有复现 Renaissance Technologies 的私有系统，也不把回测结果当成投资建议或实盘许可。

## 可以拿它做什么

- 在同一指数、同一行业内寻找协整候选
- 查看原始 p 值、FDR q 值、半衰期和当前 z-score
- 运行固定形成期、滚动重估和样本外回测
- 检查单股重复出现和配对之间的共线问题
- 在通过研究门槛后生成理论信号
- 复查成本假设、数据修订和 A 股做空限制

## 数据接口

| 用途 | PandaData 接口 | 说明 |
|---|---|---|
| 指数成分 | `get_index_weights` | 使用数据起点附近的历史沪深 300、中证 500 成分 |
| 行业分类 | `get_stock_industry` | 只在同行业内配对；覆盖不足时停止 |
| 股票日线 | `get_stock_daily_pre` | 默认前复权价格 |
| 股票日线 | `get_stock_daily_post` | 可显式选择后复权价格 |

普通未复权 `get_stock_daily` 收盘价不进入协整检验，以减少分红、送转和除权造成的伪关系。行业接口没有历史日期参数，因此历史研究使用查询时的行业分类，报告会明确披露这一限制。

仓库不附带行情、指数成分、行业数据、缓存或 PandaData 安装包。你需要自行取得数据权限，并确认自己的用途符合服务条款。`panda_data==0.0.12` 的公开材料没有写明许可证和源码维护地址，详情见[第三方说明](THIRD_PARTY_NOTICES.md)。

### 联网运行条件

数据请求只会发往你明确配置的 HTTPS 地址。`PANDADATA_BASE_URL` 不能夹带账号、密码、查询参数或片段；HTTP 和重定向都会被拒绝，也没有绕过开关。

`panda_data==0.0.12` 自带的默认地址是 HTTP，不能直接使用。服务方尚未提供 HTTPS 地址时，真实登录和数据研究会停止，但离线测试与合成数据回测不受影响。

登录可以使用短期 `PANDADATA_TOKEN`，也可以用账号密码换取令牌。程序不会让 SDK 把会话写进 `user.json`，令牌只留在当前进程中；过期后重新认证并启动命令即可。

## 研究流程

### 1. 形成期选对

1. 使用评估期之前最后 252 个交易日作为形成期；
2. 只保留同行业、对数价格相关性不低于 0.80 的候选；
3. 运行 Engle–Granger 检验，保留原始 `pvalue`；
4. 对同批候选执行 Benjamini–Hochberg 校正，默认要求 `pvalue_fdr ≤ 0.05`；
5. 只保留半衰期在 2–60 个交易日的关系；
6. 用形成期两腿收益进行 PCA 共线诊断和层次聚类；
7. 每簇保留 q 值最低的代表配对；
8. 用 `max_pairs_per_symbol=1` 限制单股重复暴露。

评估期数据不参与形成期选对，避免事后选择。

### 2. 滚动回测

| 参数 | 默认值 |
|---|---:|
| 评估期 | 完整 5 年 |
| 协整关系重估 | 每 60 个交易日 |
| 入场 | `|z| > 2` |
| 均值回归退出 | `|z| < 0.5` |
| 极端偏离停止 | `|z| > 4` |
| 最长持有 | 60 个交易日 |
| 单对峰值回撤停止 | 5% |
| 单边换手成本 | 7.5 bp |
| 融券年费假设 | 800 bp |
| 价差模型 | `static` |
| Kalman 状态噪声 `delta` | `1e-4` |
| Kalman 观测噪声 `r` | `1e-3` |

滚动区间最后一日禁止新开仓。收益按两条复权价格序列、入场对冲比例和总敞口归一化计算；成本和融券费单独记账。

默认 `static` 使用每个形成期估计的固定 α/β。选择 `--method kalman` 时，状态为 `[alpha, beta]`，状态噪声协方差为 `delta × I`，观测噪声方差为 `r`；两个参数都必须为正数。`signal` 会沿用门控回测的 `method`、`kalman_delta`、`kalman_r`、z-score 窗口和阈值，不能把静态回测结果与 Kalman 信号混用。

### 3. 研究门控

| 状态 | 含义 |
|---|---|
| `NO_TRADE_NEGATIVE_EDGE` | 扣费后年化收益或夏普不为正 |
| `NO_TRADE_INSUFFICIENT_SAMPLE` | 交易少于 100 笔 |
| `NO_TRADE_WEAK_EDGE` | 夏普低于 0.5 或年化收益低于 1% |
| `NO_TRADE_INVALID_METRICS` | 指标无效 |
| `RESEARCH_PASS` | 仅允许继续样本外和纸面研究 |

`RESEARCH_PASS` 不是实盘许可。任何 `NO_TRADE` 状态下，控制台和 CSV 都不会显示 `LONG` 或 `SHORT` 理论方向。

## 快速开始

目前在 Windows、PowerShell、Python 3.11 和 3.12 下完成了测试。Python 需要 >=3.11 且 <3.13；Linux 和 macOS 暂未做完整验收。

```powershell
git clone https://github.com/quantskills/skill-simons-pairs-trading.git
Set-Location skill-simons-pairs-trading

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-lock.txt
```

先填入服务方提供的 HTTPS 地址，再选择一种认证方式。短期令牌更省事：

```text
# 示例地址，请替换为服务方提供的真实地址
PANDADATA_BASE_URL=https://data-provider.example
PANDADATA_TOKEN=你的短期令牌
```

也可以使用账号密码换取进程内令牌：

```text
PANDADATA_BASE_URL=https://data-provider.example
PANDADATA_USER=你的账号
PANDADATA_PASSWORD=你的密码
```

这些变量也可以手工放进用户目录下的 `~/.pandadata.env`，但不要放进项目目录或版本库。程序不会从 SDK 的 `user.json` 保存或恢复会话。

第一次使用先跑健康检查。它会检查 Python、SDK、DNS、TLS 证书和凭证是否就绪，但不会打印凭证内容：

```powershell
python scripts\cli.py health-check
python scripts\cli.py check-login

# 服务方排查 HTTPS 时，可暂时只检查运行环境和传输
python scripts\cli.py health-check --transport-only

$env:SIMONS_TODAY = "20260710"

python scripts\cli.py backtest `
  --indexes 000300.SH 000905.SH `
  --years 5 `
  --price-basis pre_adjusted `
  --method static

python scripts\cli.py signal `
  --indexes 000300.SH 000905.SH `
  --years 5 `
  --price-basis pre_adjusted `
  --method static
```

`signal` 是日线收盘后的研究快照，不是盘中实时信号。它必须与回测使用相同日期、价差方法和核心参数；静态模型沿用形成期 α/β，Kalman 模型沿用同一 `delta` 与 `r`。参数变化会生成不同的 `config_id` 与输出目录，未匹配的门控报告不会放行理论方向。

单对诊断：

```powershell
python scripts\cli.py diagnose `
  --pair 600519.SH,000858.SZ `
  --lookback-days 500 `
  --price-basis pre_adjusted
```

作为 Skill 使用时，可直接提出：

```text
在沪深 300 和中证 500 内筛选同行业协整配对，输出候选池和研究门控。
用前复权数据回测五年，并检查 FDR、半衰期、重复暴露和 5% 峰值回撤停止。
```

## 数据质量与复现

- 行业查询遇到瞬时错误时逐只最多重试 3 次；
- 行业覆盖率低于 95% 时停止回测；
- 信号缺少代表股票、出现全空价格列、行业覆盖不足或指纹不匹配时，返回 `NO_TRADE_DATA_MISMATCH`；
- 报告保存 `price_panel_sha256`、`universe_sha256`、`industry_map_sha256`；
- `run_id` 绑定配置、时期、代表配对、最终门控、绩效、交易统计和共享信号窗口；
- `run_id` 是复现摘要，不是数字签名。需要防止有意篡改时，应使用只读归档或外部签名。

## 输出

```text
outputs/simons_pairs/
├── backtest_YYYYMMDD_<config_id前12位>/
│   ├── report.json
│   ├── report.md
│   ├── formation_representatives.csv
│   ├── formation_collinearity.json
│   ├── formation_collinearity.md
│   ├── pairs.csv
│   ├── trades.csv
│   ├── daily_returns.csv
│   ├── pair_daily_returns.csv
│   └── equity.png
└── signals_YYYYMMDD_<config_id前12位>.csv
```

`formation_collinearity.json` 记录行业覆盖、主成分解释率、条件数、有效秩、最大配对相关性、单股出现次数和有效独立对数。`trades.csv` 分列记录两腿收益、换手成本、融券费、峰值净收益和峰值回撤。

## 已知限制

- A 股个股不能自由做空，项目没有验证逐日券源、召回和强平；
- 真实融券费率和可用数量可能与假设完全不同；
- T+1、涨跌停、停牌、成交冲击和容量未完整模拟；
- 查询时行业分类不能替代历史时点行业分类；
- 股指期货无法精确替代个股空腿；
- 数据供应商可能修订历史复权数据；
- 真实数据运行依赖 PandaData 数据服务方提供并维护受信 HTTPS 端点；供应商公开默认 HTTP 地址会被拒绝；
- 令牌只在当前进程使用，过期后不会自动重新登录，必须重新启动并认证；
- `panda_data` 的许可证、维护来源和数据使用权需由使用者自行核验；
- 协整关系会因基本面或行业结构变化而失效；
- 历史回测和测试通过都不保证未来收益。

## 验证

下面两条命令只做离线检查，不读取个人凭证，也不访问真实数据接口：

```powershell
python -m pytest tests -q
python -m compileall -q scripts tests
```

统计定义和记账口径见 [研究方法](references/methodology.md)，完整信号示例见 [每日信号示例](examples/01_daily_signal.md)。

## 维护情况

项目目前处于 beta 阶段。离线算法可以复现，真实数据仍取决于服务方 HTTPS 地址、有效账号和数据权限。

问题和改进建议请走 GitHub Issues 或 Pull Requests，初始维护者为 Duzey。

本仓库只借鉴公开的配对交易与统计套利方法，与 Jim Simons、Renaissance Technologies 或其关联机构没有合作或授权关系。

## License

本项目采用 [MIT License](LICENSE)。
