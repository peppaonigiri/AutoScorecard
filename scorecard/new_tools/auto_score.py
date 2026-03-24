import numpy as np
import pandas as pd

class ScoreCalculator:
    def __init__(self, initial_p=1000, initial_q=10, initial_pdo=400, good_weight=1, min_score=100, max_score=1000):
        self.p = initial_p
        self.q = initial_q
        self.pdo = initial_pdo
        self.good_weight = good_weight
        self.min_score = min_score
        self.max_score = max_score
        self.range_score_max = (max_score - min_score)
        self.range_score_min = (max_score - min_score) * 0.7

    def card_score(self, pred):
        odds = (1 - pred) * self.good_weight / pred
        b = -self.pdo / np.log(2)
        a = self.p + b * np.log(self.q)
        score = a - b * np.log(odds)
        return score.astype(int)

    def automatic_scoring(self, prba_arr):
        score_list = self.card_score(prba_arr)
        scor_min = min(score_list)
        scor_max = max(score_list)

        max_iterations = 500    # 最大迭代次数
        adjustments_p = 100    # p的初始调整值
        adjustments_q = 1     # q的初始调整值
        adjustments_pdo = 10   # pdo的初始调整值
        iterations = 0
        
        while (abs(scor_max - scor_min) >= self.range_score_max or (scor_max - scor_min) <= self.range_score_min) and iterations < max_iterations:
            if abs(scor_max - scor_min) >= self.range_score_max:
                self.pdo = max(0, self.pdo - adjustments_pdo)
                # self.q += adjustments_q
            elif abs(scor_max - scor_min) <= self.range_score_min:
                self.pdo += adjustments_pdo
                # self.q = max(1, self.q - adjustments_q)
                
            score_list = self.card_score(prba_arr)
            scor_min = min(score_list)
            scor_max = max(score_list)
            iterations += 1

            # 每100次迭代，调整一次pdo为原来的一半
            if iterations % 100 == 0:
                adjustments_pdo //= 2
            
            if iterations == max_iterations:
                print("注意：分数范围调整已达到最大迭代次数")
            
        iterations = 0
        while (scor_min < self.min_score or scor_max > self.max_score) and iterations < max_iterations:
            if scor_min < self.min_score:
                self.p += adjustments_p
            if scor_max > self.max_score:
                self.p = max(0, self.p - adjustments_p)

            score_list = self.card_score(prba_arr)
            scor_min = min(score_list)
            scor_max = max(score_list)
            iterations += 1

            # 每100次迭代，调整一次p为原来的一半
            if iterations % 100 == 0:
                adjustments_p //= 2

            if iterations == max_iterations:
                print("注意：最大最小分数调整已达到最大迭代次数")

        return score_list, scor_min, scor_max, self.p, self.q, self.pdo

# 示例用法
# data = pd.DataFrame({'model_prob': np.random.rand(1000)})  # 示例数据
# scorer = ScoreCalculator(max_score=1000, min_score=100)  # 实例化分数计算器
# scores, min_score, max_score, final_p, final_q, final_pdo = scorer.automatic_scoring(data['model_prob'])
# data['score'] = scores

# # 打印结果
# print(data)
# print(f"Min Score: {min_score}, Max Score: {max_score}, P: {final_p}, Q: {final_q}, PDO: {final_pdo}")
