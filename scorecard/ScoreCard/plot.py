import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import scorecardpy as sc


def plot_score(score_ks):
    def to_percent(temp, position):
        return '%1.0f' % (1 * temp * 100) + '%'

    plt.figure(figsize=(15, 6))

    size = score_ks.shape[0]
    x = np.arange(size)
    total_width, n = 0.6, 2
    width = total_width / n  # 每种类型的柱状图宽度
    x = x - (total_width - width) / 2  # 重新设置x轴的坐标

    plt.bar(x, score_ks['goods'].tolist(), width=width, label="goods")  # 画柱状图
    plt.bar(x + width, score_ks['bads'].tolist(), width=width, label="bads")
    plt.ylabel('num', fontsize=12)
    plt.legend(loc=1, bbox_to_anchor=(0, 0.97), borderaxespad=0.)

    # 横坐标
    m1 = [str(score_ks['min'].tolist()[i]) for i in range(size)]
    m1 = [i[:-2] if i[-1] == '0' else i for i in m1]
    m2 = [str(score_ks['max'].tolist()[i]) for i in range(size)]
    m2 = [i[:-2] if i[-1] == '0' else i for i in m2]
    date1 = [m1[i] + '-' + m2[i] for i in range(size)]
    x = list(range(len(date1)))
    plt.xticks(x, date1, rotation=45)

    # 调用plt.twinx()后可绘制次坐标轴
    plt.twinx()

    plt.plot(x, score_ks['bad_rate'].tolist(), label="bad_rate")  # 绘制次坐标轴-折线图
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))  # 次坐标轴刻度百分比显示
    plt.ylabel('bad_rate', fontsize=12)
    plt.legend(loc=2, bbox_to_anchor=(1.01, 0.97), borderaxespad=0.)  # 系坐标轴图列

    badrate = list(map(float, score_ks['bad_rate'].tolist()))
    for x, y in enumerate(badrate):
        plt.text(x, y, '%1.02f' % (1 * y * 100) + '%', fontsize=10)

    plt.title("bads_goods_badrate")

    # #图表Table显示plt.table()
    # listdata=[r_qty]+[e_qty]+[userate]+[uptime]#数据
    # table_row=['RBT_MOVE','EQP_MOVE','USE_RATE(%)','UPTIME(%)']#行标签
    # table_col=date1#列标签
    # print(listdata)
    # print(table_row)
    # print(table_col)

    # the_table=plt.table(cellText=listdata,cellLoc='center',rowLabels=table_row,colLabels=table_col,rowLoc='center',colLoc='center')
    # Table参数设置-字体大小太小，自己设置
    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(12)
    # #Table参数设置-改变表内字体显示比例，没有会溢出到表格线外面
    # the_table.scale(1,3)
    # plt.show()

    plt.legend()
    plt.show()


def plot_figure(score_ks, col_name):
    x, y = score_ks.index, score_ks[col_name]
    plt.plot(x, y, linewidth=2, c='g')
    plt.title(col_name, fontsize=20)
    plt.xlabel("bin number", fontsize=12)
    plt.ylabel(col_name, fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    plt.ylim(0.8, score_ks[col_name].max() * 1.1)

    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.show()


def plot_score_psi(data, train_set=['train', 'valid'], valid_set=['oot'], x_tick_break=50):
    score = data[['proba', 'score', 'label', 'target', ]]

    sc.perf_psi(
        score={'train-test': score[score['target'].isin(train_set)][['score']],
               'oot': score[score['target'].isin(valid_set)][['score']]},
        x_tick_break=x_tick_break,
        label={'train-test': score[score['target'].isin(train_set)][['label']],
               'oot': score[score['target'].isin(valid_set)][['label']]})

