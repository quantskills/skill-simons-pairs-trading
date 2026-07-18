# A 股协整配对交易研究

> 用同行业筛选、Engle–Granger 协整、FDR 校正、共线去重和滚动回测，研究 A 股均值回归配对。

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)
![Project](https://img.shields.io/badge/QUANTSKILLS-Community%20Project-blue)

## 项目简介

本项目提供一套可复核的 A 股配对交易研究流程。它从沪深 300 和中证 500 的历史成分中构建股票池，在同行业内筛选候选，使用复权日线做协整检验，再通过 Benjamini–Hochberg FDR、半衰期、PCA 共线诊断、层次聚类和单股重复限制压缩候选池。

入选配对进入滚动样本外回测。系统记录两腿收益、对冲比例、换手成本、融券持有费、持仓期峰值回撤和退出原因，并用保守门控决定是否允许继续纸面研究。

项目借鉴公开的统计套利和均值回归方法，不声称复现 Jim Simons、Renaissance Technologies 或任何私有基金系统，也不构成投资建议或实盘许可。

## 适用场景

- 在同一指数和行业范围内寻找协整候选；
- 查看候选配对池、原始 p 值、FDR q 值、半衰期和当前 z-score；
- 运行固定形成期、滚动重估和样本外回测；
- 检查单股重复暴露与配对共线；
- 在满足完整研究门控时生成理论信号；
- 复核数据修订、成本假设和 A 股做空限制。

## 数据接口

| 用途 | PandaData 接口 | 说明 |
|---|---|---|
| 指数成分 | `get_index_weights` | 使用数据起点附近的历史沪深 300、中证 500 成分 |
| 行业分类 | `get_stock_industry` | 只在同行业内配对；覆盖不足时停止 |
| 股票日线 | `get_stock_daily_pre` | 默认前复权价格 |
| 股票日线 | `get_stock_daily_post` | 可显式选择后复权价格 |

普通未复权 `get_stock_daily` 收盘价不进入协整检验，以减少分红、送转和除权造成的伪关系。行业接口没有历史日期参数，因此历史研究使用查询时的行业分类，报告会明确披露这一限制。

仓库不附带行情数据、指数成分、行业数据、缓存或 `panda_data` 安装包。使用者需要自行取得 PandaData 访问权限，并遵守相应的数据授权和使用条款。`panda_data==0.0.12` 的公开材料没有提供可核验的许可证和维护来源，使用前应自行完成授权与供应链核查；详见 [第三方说明](THIRD_PARTY_NOTICES.md)。

### 联网运行条件

数据请求只允许发往用户明确配置的受信 HTTPS 地址。`PANDADATA_BASE_URL` 必须是完整的 `https://` 地址，不得包含账号或密码；HTTP 地址和重定向都会被拒绝，不能用开关绕过。

当前 `panda_data==0.0.12` 公开默认地址使用 HTTP，因此不能直接用于本项目。要调用真实数据，使用者必须先取得由自己或数据服务方提供的受信 HTTPS 网关，并确认网关证书、访问权限和数据授权。没有该网关时，登录检查和真实数据研究不可运行，但离线测试、合成数据回测和算法复现仍可完成。

登录可使用短期 `PANDADATA_TOKEN`，也可使用账号密码换取令牌。安全适配层会关闭 SDK 的 `user.json` 读写、磁盘自动登录和自动重新登录，并把会话限制在当前进程内。令牌到期后应结束当前进程，更新令牌或账号凭证，再重新启动命令。

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

已验收环境为 Windows、PowerShell 和 Python 3.12。Linux 与 macOS 尚未纳入正式验收。

```powershell
git clone https://github.com/quantskills/skill-simons-pairs-trading.git
Set-Location skill-simons-pairs-trading

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-lock.txt
```

先配置受信 HTTPS 网关，并在当前进程中选择一种认证方式。优先使用短期令牌：

```text
PANDADATA_BASE_URL=https://你的受信网关.example
PANDADATA_TOKEN=你的短期令牌
```

也可以使用账号密码换取进程内令牌：

```text
PANDADATA_BASE_URL=https://你的受信网关.example
PANDADATA_USER=你的账号
PANDADATA_PASSWORD=你的密码
```

这些变量也可由使用者手工保存到用户级 `~/.pandadata.env`，但不应放进项目目录或版本库。程序不会把令牌、账号或密码写入 SDK 的 `user.json`，也不会从该文件恢复会话。令牌失效时，结束进程并重新认证。

运行登录检查、回测和同日信号：

```powershell
python scripts\cli.py check-login

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
- 真实数据运行依赖用户提供受信 HTTPS 网关；供应商公开默认 HTTP 地址会被拒绝；
- 令牌只在当前进程使用，过期后不会自动重新登录，必须重新启动并认证；
- `panda_data` 的许可证、维护来源和数据使用权需由使用者自行核验；
- 协整关系会因基本面或行业结构变化而失效；
- 历史回测和测试通过都不保证未来收益。

## 验证

离线测试不会读取用户级凭证，也不会调用真实网络接口；它覆盖算法、研究契约和安全失败路径。真实数据验收只有在使用者提供受信 HTTPS 网关并具备合法数据权限时才能进行：

```powershell
python -m pytest tests -q
python -m compileall -q scripts tests
```

统计定义和记账口径见 [研究方法](references/methodology.md)，完整信号示例见 [每日信号示例](examples/01_daily_signal.md)。

## 项目状态与维护

本仓库当前为 Beta。离线算法和研究契约可复现；真实数据运行取决于受信 HTTPS 网关、有效认证及数据授权，因此验收状态为有条件可运行。

本仓库是 QUANTSKILLS Community Project，初始维护者为 Duzey；后续问题和变更通过 GitHub Issues 与 Pull Requests 处理。它不代表 QUANTSKILLS 官方认证、验证或背书。

本项目只借鉴公开的配对交易与统计套利方法，与 Jim Simons、Renaissance Technologies 或其关联机构无隶属、合作或授权关系。

## License

本项目采用 [MIT License](LICENSE)。
