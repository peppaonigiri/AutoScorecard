import pandas as pd
import matplotlib.pyplot as plt

def plot_lift(y_predproab, y_true):
    '''
    params:
        y_predproab:预测值概率/正样本分组
        y_true:真实值/正负样本标记label
    result:
        绘制lifi曲线
    '''
    result = pd.DataFrame([y_true, y_predproab]).T
    result.columns = ['target', 'proba']
    result = result.sort_values(['proba', 'target'], ascending=False).reset_index()
    del result['index']
    result.set_index((result.index + 1) / result.shape[0], inplace=True)
    result['bad_sum'] = result['target'].cumsum()
    result['count_sum'] = [i + 1 for i in range(result.shape[0])]
    result['rate'] = result['bad_sum'] / result['count_sum']
    result['lift'] = result['rate'] / (result['target'].sum() / result.shape[0])

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid(True, linestyle='-.')
    ax1.plot(result['rate'], color='red', label='Lift model')
    ax1.plot(result.index, [result['target'].sum() / result.shape[0]] * result.shape[0], color='blue',
             label='Lift random')
    ax1.set_title('Lift Chart', fontsize=25)
    ax1.set_ylabel('tp/(tp+fp)', fontsize=20)
    ax1.set_xlabel('data sets', fontsize=20)
    ax1.set_xticks([i / 10 for i in range(11)])
    plt.legend(loc='best')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(result['lift'], color='darkorange')
    ax2.grid(True, linestyle='-.')
    ax2.set_title('Cumulative Lift Chart', fontsize=25)
    ax2.set_ylabel('lift', fontsize=20)
    ax2.set_xlabel('data sets', fontsize=20)
    ax2.set_xticks([i / 10 for i in range(11)])

    plt.show()
