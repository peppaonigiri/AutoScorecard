# -*- coding: utf-8 -*-
import numpy as np

def proba2score(prob, pdo=30, rate=2, base_odds=35, base_score=750):
    """从概率转换到分数
    
    公式: Score = Offset + Factor * ln(odds)
    其中:
        Factor = pdo / ln(rate)
        Offset = base_score - Factor * ln(base_odds)
        odds = (1 - prob) / prob
    """
    # 避免无效对数 (prob 必须在 (0, 1) 之间)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    factor = pdo / np.log(rate)
    offset = base_score - factor * np.log(base_odds)
    return factor * (np.log(1 - prob) - np.log(prob)) + offset
