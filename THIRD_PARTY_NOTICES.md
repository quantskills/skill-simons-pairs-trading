# 第三方软件与数据说明

本文件说明本仓库与外部软件、数据服务和公开研究方法的边界，不替代各权利人的许可证或服务条款。

## PandaData

项目通过 Python 包 `panda_data==0.0.12` 调用指数成分、行业分类和股票复权日线接口。该包由使用者依据 `requirements-lock.txt` 自行安装，不包含在本仓库源码或发布压缩包中。

截至本次发布准备，随该包及本项目可核验材料未提供明确的软件许可证、维护主体和源码来源。本项目不据此授予任何第三方软件或数据权利。使用者在安装、访问或分发相关内容前，应自行确认：

- 软件包的许可证、来源、维护主体和供应链风险；
- PandaData 账号、接口和数据集的合法访问权限；
- 行情、指数成分、行业分类、缓存和研究输出的使用与再分发范围；
- PandaData 数据服务方提供的 HTTPS 端点、证书和服务条款。

`panda_data==0.0.12` 的公开默认服务地址使用 HTTP，本项目永久拒绝该传输方式。真实数据运行必须由 PandaData 数据服务方提供并维护受信 HTTPS 端点；否则只能运行离线测试和合成数据研究。

## 其他 Python 依赖

NumPy、pandas、SciPy、statsmodels、Matplotlib、PyArrow、tabulate 及其间接依赖由各自权利人维护，并按各自许可证提供。版本和下载文件摘要记录在 `requirements-lock.txt`。锁定摘要用于复现安装内容，不改变第三方许可证，也不代表对其来源或适用性的背书。

## 公开研究方法与名称

项目采用 Engle–Granger 协整检验、Benjamini–Hochberg FDR、PCA、层次聚类、Kalman 状态估计和均值回归回测等公开方法。方法引用见 `references/methodology.md`。

项目名称中的 “Simons” 仅表示统计套利研究风格。项目与 Jim Simons、Renaissance Technologies 或其关联机构没有隶属、合作、授权或认可关系，也不声称复现任何私有交易系统。

## 本仓库许可证

本仓库自行创作的代码和文档采用 [MIT License](LICENSE)。MIT License 不覆盖第三方软件、数据、商标、账号、接口服务或使用者自行提供的内容。
