
# 4. z_score
def Zscore(data):
    cols = data.columns
    # 获得数据框的列名
    for col in cols:
        # 循环读取每列
        df_col = data[col]
        # 得到每列的值
        z_score = (df_col - df_col.mean()) / df_col.std()
        # 计算每列的Z-score得分
        # 判断Z-score得分是否大于2.2，（此处2.2代表一个经验值），如果是则是True，否则为False
        data[col] = z_score.abs() > 2.2
    print(data)
    # 打印输出
    return data