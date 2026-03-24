from process_rules import process_rules
import reduce_mem_usage
import pandas as pd

if __name__ == '__main__':

    # 读取特征数据
    data = pd.read_csv(r'D:\jupyter\策略挖掘\墨西哥\sample-yq_gamav52_20240702.csv')

    # 读取标签数据
    y_label = pd.read_csv(r'D:\jupyter\策略挖掘\墨西哥\sample-yq-y.csv')

    # 特征数据与标签数据的匹配列名称
    merge_col = ['mobile_number_md5_v2']

    data = data.merge(y_label, on=merge_col, how='left')

    data = reduce_mem_usage.reduce_mem_usage(data)

    del y_label

    # 这里设置需要保留的特征列，例如所有特征第一个字母均为v的列，其余的都是非特征列

    keep_list = [col for col in data.columns if 'v' == col[0]]
    ex_list = [col for col in data.columns if col not in keep_list]


    # # 或者在这里单独设置需要排除的非特征列
    # ex_list = ['age', 'gender', 'income']
    # 这里设置时间列名称
    time_col = 'time'
    data[time_col] = pd.to_datetime(data[time_col]).dt.strftime('%Y-%m-%d')
    # 设置标签列名称
    y_col = 'fpd7_fz'
    # 设置结果保存路径
    fail_path = 'D:\jupyter\策略挖掘'

    # 开始自动化规则挖掘
    pr = process_rules(data, ex_list, time_col, y_col, fail_path)

    # 获取单个规则表现
    pr.get_rules_df()
    # 获取规则集在训练集、测试集以及整体数据集上的表现
    pr.get_rules_set()
    # 获取按照特定规则集捕获的样本明细,如果要指定规则集，则需要传入切断点的索引,比如下列代码为取生成规则集的前20条规则
    rule_list = pd.read_excel(r'D:\jupyter\策略挖掘\rules_set_all.xlsx')['规则'].to_list()[:20]
    pr.get_catch_df(rule_list)