#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUC函数差异比较测试
比较sklearn.metrics.roc_auc_score和sklearn.metrics.auc函数的差异
包含XGBoost训练和测试的完整案例
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data(n_samples=1000, n_features=10, n_informative=8, random_state=42):
    """
    生成样本数据
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # 转换为DataFrame便于处理
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names

def train_xgb_model(X_train, y_train, X_val, y_val):
    """
    训练XGBoost模型
    """
    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    return model

def compare_auc_functions(y_true, y_pred_proba):
    """
    比较roc_auc_score和auc函数的差异
    """
    print("=" * 60)
    print("AUC函数差异比较")
    print("=" * 60)
    
    # 方法1: 使用roc_auc_score直接计算AUC
    auc_score_direct = roc_auc_score(y_true, y_pred_proba)
    print(f"1. roc_auc_score直接计算: {auc_score_direct:.6f}")
    
    # 方法2: 使用roc_curve + auc计算AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score_curve = auc(fpr, tpr)
    print(f"2. roc_curve + auc计算:   {auc_score_curve:.6f}")
    
    # 检查差异
    difference = abs(auc_score_direct - auc_score_curve)
    print(f"3. 两种方法差异:         {difference:.10f}")
    
    if difference < 1e-10:
        print("4. 结论: 两种方法结果完全一致")
    else:
        print("4. 结论: 两种方法存在微小差异")
    
    print("\n详细说明:")
    print("- roc_auc_score: 直接计算ROC曲线下的面积")
    print("- auc(fpr, tpr): 基于ROC曲线的FPR和TPR计算面积")
    print("- 理论上应该完全一致，差异可能来自数值精度")
    
    return auc_score_direct, auc_score_curve, fpr, tpr

def plot_roc_curve(fpr, tpr, auc_value, title="ROC Curve"):
    """
    绘制ROC曲线
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)
    
    # 测试1: 完美分类器
    y_perfect = np.array([0, 0, 0, 1, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    auc1 = roc_auc_score(y_perfect, y_pred_perfect)
    fpr, tpr, _ = roc_curve(y_perfect, y_pred_perfect)
    auc2 = auc(fpr, tpr)
    
    print(f"完美分类器测试:")
    print(f"  roc_auc_score: {auc1:.6f}")
    print(f"  auc(fpr, tpr): {auc2:.6f}")
    print(f"  差异: {abs(auc1 - auc2):.10f}")
    
    # 测试2: 随机分类器
    y_random = np.array([0, 0, 0, 1, 1, 1])
    y_pred_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    auc1 = roc_auc_score(y_random, y_pred_random)
    fpr, tpr, _ = roc_curve(y_random, y_pred_random)
    auc2 = auc(fpr, tpr)
    
    print(f"\n随机分类器测试:")
    print(f"  roc_auc_score: {auc1:.6f}")
    print(f"  auc(fpr, tpr): {auc2:.6f}")
    print(f"  差异: {abs(auc1 - auc2):.10f}")
    
    # 测试3: 反向分类器
    y_reverse = np.array([0, 0, 0, 1, 1, 1])
    y_pred_reverse = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    
    auc1 = roc_auc_score(y_reverse, y_pred_reverse)
    fpr, tpr, _ = roc_curve(y_reverse, y_pred_reverse)
    auc2 = auc(fpr, tpr)
    
    print(f"\n反向分类器测试:")
    print(f"  roc_auc_score: {auc1:.6f}")
    print(f"  auc(fpr, tpr): {auc2:.6f}")
    print(f"  差异: {abs(auc1 - auc2):.10f}")

def main():
    """
    主函数
    """
    print("XGBoost AUC函数差异比较测试")
    print("=" * 60)
    
    # 1. 生成数据
    print("1. 生成样本数据...")
    df, feature_names = generate_sample_data(n_samples=2000, n_features=15)
    print(f"   数据形状: {df.shape}")
    print(f"   正样本比例: {df['target'].mean():.3f}")
    
    # 2. 划分训练集和测试集
    print("\n2. 划分训练集和测试集...")
    X = df[feature_names]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 3. 训练XGBoost模型
    print("\n3. 训练XGBoost模型...")
    model = train_xgb_model(X_train, y_train, X_val, y_val)
    
    # 4. 预测
    print("\n4. 模型预测...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"   预测概率范围: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"   预测准确率: {(y_pred == y_test).mean():.4f}")
    
    # 5. 比较AUC函数
    print("\n5. 比较AUC函数...")
    auc_direct, auc_curve, fpr, tpr = compare_auc_functions(y_test, y_pred_proba)
    
    # 6. 绘制ROC曲线
    print("\n6. 绘制ROC曲线...")
    plot_roc_curve(fpr, tpr, auc_direct, "XGBoost ROC Curve")
    
    # 7. 测试边界情况
    test_edge_cases()
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. roc_auc_score和auc(fpr, tpr)在理论上应该完全一致")
    print("2. 实际使用中可能存在微小的数值精度差异")
    print("3. 推荐使用roc_auc_score，因为它更直接且高效")
    print("4. 如果需要ROC曲线的详细信息，可以使用roc_curve + auc的组合")
    print("5. 两种方法都适用于二分类问题的AUC计算")

if __name__ == "__main__":
    main()
