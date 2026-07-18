# 示例：生成受门控的每日研究信号

## 场景

使用同一截止日、同一双指数池和同一参数，先完成五年样本外回测，再生成每日研究文件。示例假设回测没有达到研究门槛，因此方向必须隐藏。

## PowerShell 命令

```powershell
$skillDir = "C:\path\to\skill-simons-pairs-trading"
Set-Location $skillDir
$env:SIMONS_TODAY="20260710"

python scripts\cli.py backtest `
  --indexes 000300.SH 000905.SH `
  --years 5 `
  --price-basis pre_adjusted `
  --corr 0.80 `
  --pvalue 0.05 `
  --fdr 0.05

python scripts\cli.py signal `
  --indexes 000300.SH 000905.SH `
  --years 5 `
  --price-basis pre_adjusted `
  --corr 0.80 `
  --pvalue 0.05 `
  --fdr 0.05
```

## 预期状态

```text
NO_TRADE_WEAK_EDGE | 研究门控=NO_TRADE_WEAK_EDGE
方向已隐藏；保存无方向诊断记录，数量取决于当次代表配对
输出：...\signals_20260710_<config_id前12位>.csv
```

CSV中的关键字段：

```text
signal=NO_TRADE
research_status=NO_TRADE
strategy_gate=NO_TRADE_WEAK_EDGE
gate_report_run_id=<同日回测run_id>
gate_report_verified=True
price_basis=pre_adjusted
```

默认输出不显示理论多空方向。即使报告变为 `RESEARCH_PASS`，输出也只标记 `RESEARCH_ONLY`，仍需独立验证券源、T+1、涨跌停、停牌和实际成本。
