---
name: simons-pairs-trading
description: "Screen and audit reproducible, research-only A-share pairs with adjusted PandaData prices, same-industry Engle-Granger tests, Benjamini-Hochberg FDR, formation-only clustering, rolling backtests, conservative gates, and z-score diagnostics. Use for cointegration screening, mean-reversion backtests, pair diagnostics, or research signals."
---

# A 股协整配对交易研究

筛选和评估 A 股均值回归配对。所有结果仅用于研究，不要把候选配对、`RESEARCH_PASS` 或理论方向写成投资建议、自动交易指令和实盘许可。

## 运行准备

使用 Python >=3.11 且 <3.13，并在 Skill 根目录安装锁定依赖：

```powershell
python -m pip install --require-hashes -r requirements-lock.txt
```

真实数据只连接 PandaData 服务方提供的 HTTPS 地址：

```text
# 示例地址，运行前替换
PANDADATA_BASE_URL=https://data-provider.example
PANDADATA_TOKEN=你的短期令牌
```

也可用账号密码换取进程内令牌：

```text
PANDADATA_BASE_URL=https://data-provider.example
PANDADATA_USER=你的账号
PANDADATA_PASSWORD=你的密码
```

`PANDADATA_BASE_URL` 必须是完整的 HTTPS 地址，不能包含凭证、查询参数或片段。拒绝 HTTP 和重定向，不尝试绕过。`panda_data==0.0.12` 的默认地址使用 HTTP；服务方没有提供 HTTPS 地址时，只运行离线测试和合成数据研究。

只从当前进程环境变量或用户目录下的 `~/.pandadata.env` 读取凭证，不读取项目目录或当前工作目录中的 `.env`。保持 SDK 的 `user.json` 读写和自动登录关闭，会话只在当前进程内有效。令牌失效后重新认证。不要在回复、日志、报告或仓库中展示凭证、令牌、完整第三方异常、私有缓存和私人研究结果。

先执行启动健康检查，再验证登录：

```powershell
python scripts\cli.py health-check
python scripts\cli.py check-login
```

## 标准流程

1. 固定研究截止日、指数池、完整样本外评估期和复权口径。
2. 用 `get_index_weights` 获取历史指数成分，用 `get_stock_industry` 获取行业分类。
3. 用 `get_stock_daily_pre` 获取前复权日线；只有明确要求时才使用 `get_stock_daily_post`。
4. 在评估期之前最后 252 个交易日完成形成期筛选和去重。
5. 每 60 个交易日重估协整关系与半衰期，执行滚动回测。
6. 计算研究门控；只有同日、同配置、完整一致的 `RESEARCH_PASS` 报告才允许显示理论方向。

仓库不附带行情、指数成分、行业数据、缓存或 PandaData 安装包。运行者自行确认软件许可和数据授权，服务方负责提供 HTTPS 地址；细节见 `THIRD_PARTY_NOTICES.md`。

常用命令：

```powershell
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

## 形成期选对约定

- 普通未复权 `get_stock_daily` 收盘价不得进入协整检验。
- 只保留同行业、形成期对数价格相关性达到 0.80 的候选。
- 使用 Engle–Granger 检验，同时保留原始 `pvalue`。
- 对同批候选执行 Benjamini–Hochberg 校正；默认要求原始 p<0.05、`pvalue_fdr ≤ 0.05`。
- 半衰期必须在 2–60 个交易日。
- 只用形成期两腿收益完成 PCA 共线诊断和层次聚类。
- 每簇保留 q 值最低的代表配对，再用 `max_pairs_per_symbol=1` 限制单股重复暴露。
- 评估期或全样本结果不得反向参与形成期选对。

## 回测与风控约定

- 每 60 个交易日滚动重估；关系失效时不开仓。
- 默认 `|z|>2` 入场、`|z|<0.5` 退出、`|z|>4` 停止。
- 最长持有 60 个交易日。
- 单对净收益相对持仓期峰值回撤达到 5% 时停止。
- 滚动区间最后一日不得新开随后被丢弃的仓位。
- 按两条真实资产腿、入场对冲比例和总敞口归一化计算收益。
- 默认单边换手成本 7.5 bp、融券年费 800 bp；两腿收益、成本和退出原因必须分列记录。
- 默认 `method=static`。选择 `method=kalman` 时使用 `kalman_delta=1e-4`、`kalman_r=1e-3`，且两项必须为正数。

## 研究门控

- 扣费后年化收益或夏普不为正：`NO_TRADE_NEGATIVE_EDGE`
- 交易少于 100 笔：`NO_TRADE_INSUFFICIENT_SAMPLE`
- 夏普低于 0.5 或年化收益低于 1%：`NO_TRADE_WEAK_EDGE`
- 指标无效：`NO_TRADE_INVALID_METRICS`
- 达到研究阈值：`RESEARCH_PASS`

任何 `NO_TRADE` 状态下，控制台和 CSV 都不得显示 `LONG` 或 `SHORT` 理论方向。`RESEARCH_PASS` 也只允许继续样本外和纸面研究。

## 门控与复现

门控必须绑定指数池、形成期、评估期、统计阈值、FDR、半衰期、方法、滚动窗口、z-score 阈值、持有期、5% 峰值回撤停止、成本、融券费、复权口径、聚类和单股重复上限。

报告同时绑定代码版本、最终门控、绩效、交易统计、代表配对摘要，以及 `price_panel_sha256`、`universe_sha256`、`industry_map_sha256`。`run_id` 是这些内容的复现摘要，不是数字签名。

`signal` 只重新评估报告内已完成形成期筛选和回测的代表配对。它必须使用与门控回测相同的价差方法：静态模型沿用形成期 α/β，Kalman 模型沿用同一 `kalman_delta` 与 `kalman_r`；z-score 窗口和阈值也必须一致。缺少代表股票、出现全空价格列、行业覆盖不足、配置不一致或指纹不匹配时，返回失败门控或 `NO_TRADE_DATA_MISMATCH`，保存无方向 CSV，并停止筛选。

`signal` 是日线收盘后的研究快照，不是盘中实时信号或自动下单接口。

## 输出约定

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

不得手改 `report.json`。同日、同配置重跑会使用同一目录；如需保留供应商修订前后的两个版本，应先归档原目录。

## 单对诊断

```powershell
python scripts\cli.py diagnose `
  --pair 600519.SH,000858.SZ `
  --lookback-days 500 `
  --price-basis pre_adjusted
```

诊断图不构成门控通过或交易建议。

## 必须披露的限制

明确说明 A 股个股做空和逐日券源未验证；T+1、涨跌停、停牌、成交冲击、融券召回、强平和容量未完整模拟；行业分类不是历史时点数据；股指期货不能精确替代个股空腿；供应商可能修订复权数据；协整关系可能失效。真实数据运行还依赖服务方 HTTPS 端点、有效认证和合法数据权限；HTTP 永久禁用，令牌过期后必须重新启动并认证。历史结果和测试通过都不保证未来收益。

## 验证

```powershell
python -m pytest tests -q
python -m compileall -q scripts tests
```

测试通过只说明程序行为符合研究约定，不说明策略有效或适合实盘。服务方 HTTPS 地址、认证或数据授权缺一时，停止真实数据命令。统计定义见 `references/methodology.md`，使用示例见 `examples/01_daily_signal.md`。
