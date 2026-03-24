# TzScoreGroupTools 评分分组分析工具使用说明

`TzScoreGroupTools` 是一个用于评分卡模型评估与分组分析的通用工具类。它封装了 **KS**、**AUC**、**LIFT**、**PSI** 等常用指标的批量计算功能，支持分组分析、交叉分析、加权计算以及多模型（多分数列）对比。

## 1. 核心功能

- **多维度指标计算**：支持同时计算 KS、AUC、Top 5% Lift、Sub 5% Lift 等指标。
- **分组分析**：支持按指定字段（如渠道、客户类型等）进行分组统计。
- **交叉分析**：支持任意两个维度的交叉组合分析（如 渠道 x 月份）。
- **PSI 监控**：提供月度 PSI 计算功能，支持全量及分组 PSI 监控。
- **自动化报告**：支持一键生成包含数据透视表、PSI 明细及分布图的 Excel 报告。
- **灵活配置**：
    - 支持样本权重 (`weight`)。
    - 支持 Score 模式（分数越低越坏）与 Probability 模式（概率越高越坏）。
    - 支持多层索引 (`MultiIndex`) 展示，便于对比多个模型表现。

## 2. 快速开始

以下是一个完整的使用示例，展示了如何初始化工具并调用核心功能。

```python
import pandas as pd
import numpy as np
from datetime import datetime
from new_tools.score_group_tools import TzScoreGroupTools

# 1. 准备数据
# 假设 df 是包含预测结果的数据框
# 必须包含：目标列(target), 时间列(month), 分组列(可选), 分数列(至少一列)
# df = pd.read_csv('your_data.csv') 

# 2. 初始化工具类
tool = TzScoreGroupTools(
    label_df=df,                                      # 数据源
    target_col='target',                              # 目标变量名 (0/1)
    month_col='month',                                # 月份列名 (格式如 '2023-01')
    groupby_list=['channel', 'product_type'],         # 默认分析的分组字段列表
    proba_list=['model_score_v1', 'model_score_v2'],  # 需要评估的分数列名列表
    if_weight=False,                                  # 是否使用权重 (若True需数据中有'weight'列)
    if_score=True,                                    # True表示分数值(越小越坏), False表示概率值(越大越坏)
    metrics_lst=['KS', 'AUC', 'TOP_5_LIFT'],          # 需要计算的指标
    multi_index=True                                  # 输出结果是否使用多层索引格式
)

# 3. 计算单维度分组指标 (如按渠道看各模型表现)
result_channel = tool.group_metrics(groupby_col='channel')
print(result_channel)

# 4. 批量计算所有默认分组的指标
result_all_groups = tool.group_metrics_many()
# 返回字典: {'channel': df_channel, 'product_type': df_product}

# 5. 交叉分析 (如 渠道 x 月份)
result_cross = tool.get_cross(['channel', 'month'], proba_col='model_score_v1')

# 6. 计算月度 PSI
psi_result = tool.get_month_psi(groupby_col='channel')

# 7. 生成完整 Excel 报告 (推荐)
tool.create_report(
    save_path='./model_report/', 
    phs_n_cols_simmple=3, 
    if_score=True
)
```

## 3. API 详解

### 3.1 初始化 `__init__`

```python
TzScoreGroupTools(
    label_df: pd.DataFrame, 
    target_col: str = 'target',
    month_col = 'month',
    groupby_list: list = None, 
    proba_list: list = None, 
    if_weight: bool = False, 
    if_score = False, 
    metrics_lst: list = None, 
    multi_index: bool = False
)
```
- **label_df**: 输入的 Pandas DataFrame。
- **proba_list**: 待评估的模型分数列名列表。
- **if_score**: 
    - `False` (默认): 输入为概率值 (Probability)，值越大代表风险越高 (Bad)。
    - `True`: 输入为评分 (Score)，值越小代表风险越高 (Bad)。**注意：** 设置正确与否会影响 AUC (是否大于0.5) 和 Lift 的方向。

### 3.2 指标计算 `group_metrics`

计算指定分组下的各项模型指标。

```python
df = tool.group_metrics(
    groupby_col='channel',  # 指定分组列, 若为 None 则只计算全量
    proba_cols=['score1', 'score2'] # 覆盖默认分数列
)
```

### 3.3 批量分组计算 `group_metrics_many`

对 `groupby_list` 中的每一个字段分别调用 `group_metrics`。

```python
# 返回一个字典，key为分组列名，value为对应的指标统计 DataFrame
dfs_dict = tool.group_metrics_many()
```

### 3.4 交叉分析 `get_cross`

对多个维度组合后进行指标计算。

```python
# 例如分析不同渠道在不同月份的表现
df_cross = tool.get_cross(
    groupby_col_cross_list=['channel', 'month'], 
    proba_col='score1'
)
```

### 3.5 PSI 计算 `get_month_psi`

计算基准月（通常为数据中最小月份）与其他月份的 PSI。

```python
# 计算按渠道分组的月度 PSI
df_psi = tool.get_month_psi(groupby_col='channel')
```

### 3.6 报告生成 `create_report`

一键生成包含多个 Sheet 的 Excel 报告文件。

```python
tool.create_report(
    save_path='D:/reports/', 
    if_simple=True,   # 生成简报
    if_detail=True,   # 生成详细分组报告
    if_psi=True,      # 生成 PSI 报告
    if_detail_plot=True # 是否在 Excel 中插入分布图 (耗时较长)
)
```
- 输出文件通常包括：
    - `simple_report.xlsx`: 全量及整体指标概览。
    - `detail_report.xlsx`:各维度详细分组指标及分布图。
    - `detail_report_psi.xlsx`: 各维度详细 PSI 数据。

## 4. 依赖项

该工具依赖以下内部模块 (需确保在 `sys.path` 或同级目录下):
- `new_tools.new_psi`
- `new_tools.excel_utils`
- `new_tools.plt_utils`

以及第三方库: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`.
