from Model.train import show
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scorecardpy as sc
import toad
from Model.train import lr_model
import json
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import roc_curve, auc, recall_score, precision_score

plt.style.use('seaborn-deep')
plt.rcParams['font.sans-serif'] = 'Times New Roman'


class ScorecardEvaluation(object):

    def __init__(self, type='信贷'):
        self.type = type
        self.split_points = None
        self.model_correction = None
        self.model_ce = None
        self.score_ks = None

    def eval_dataset(self, data_end, model, keep_lst, col_name='target'):
        time_set = list(data_end[col_name].unique())
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end[col_name] == i]
        res = show(model, keep_lst, 'label', datasets)
        return res

    def eval_month(self, data_end, model, keep_lst, col_name='month_time'):
        time_set = sorted(list(data_end[col_name].unique()))
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end[col_name] == i]
        res = show(model, keep_lst, 'label', datasets)
        self.plot_ks(res)
        return res

    def plot_ks(self, res):
        x, y = res['datasets'], res['ks']
        plt.plot(x, y, linewidth=2, c='g')
        plt.title("ks curve", fontsize=20)
        plt.xlabel("month time", fontsize=12)
        plt.ylabel("ks value", fontsize=12)
        plt.tick_params(axis='both', labelsize=10)
        plt.ylim(0, res['ks'].max() * 1.2)
        plt.show()

    def plot_psi(self, data_end):

        time_set = list(data_end['month_time'].unique())
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end['month_time'] == i]
        res = pd.DataFrame()
        for k, v in datasets.items():
            res = res.append(
                {'month_time': k, 'psi': toad.metrics.PSI(data_end[data_end['target'].isin(['train'])]['score'],
                                                          v['score'])},
                ignore_index=True)
        res.sort_values(by='month_time', inplace=True)

        x, y = res['month_time'], res['psi']
        plt.plot(x, y, linewidth=2, c='g')
        plt.title("psi curve", fontsize=20)
        plt.xlabel("month time", fontsize=12)
        plt.ylabel("psi value", fontsize=12)
        plt.tick_params(axis='both', labelsize=10)
        plt.ylim(0, res['psi'].max() * 10)

        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.show()
        return res

    def plot_distribution(self, data_end, bin_num=20, x_tick_break=60, col_name='target'):
        time_set = list(data_end[col_name].unique())
        d_proba, d_score = {}, {}
        for i in time_set:
            d_proba[i] = data_end[data_end[col_name] == i]['proba']
            d_score[i] = data_end[data_end[col_name] == i]['score']

        fig, axes = plt.subplots(2, len(time_set))
        plt.style.use('ggplot')
        fig.set_figwidth(10*len(time_set))
        fig.set_figheight(5*2)

        for i in range(len(time_set)):
            key = time_set[i]
            sns.distplot(d_proba[key], bins=bin_num, ax=axes[0, i], kde_kws={"label": key}, label=key,
                         axlabel='proba_'+key)
            sns.distplot(d_score[key], bins=bin_num, ax=axes[1, i], kde_kws={"label": key}, label=key,
                         axlabel='score_'+key)

        score = data_end[['proba', 'score', 'label', 'target']]
        if 'oot' in time_set:
            sc.perf_psi(
                score={'train-test': score[score['target'].isin(['train', 'valid'])][['score']],
                       'oot': score[score['target'] == 'oot'][['score']]},
                x_tick_break=x_tick_break,
                label={'train-test': score[score['target'].isin(['train', 'valid'])][['label']],
                       'oot': score[score['target'] == 'oot'][['label']]})
        else:
            sc.perf_psi(
                score={'train': score[score['target'].isin(['train'])][['score']],
                       'valid': score[score['target'] == 'valid'][['score']]},
                x_tick_break=x_tick_break,
                label={'train': score[score['target'].isin(['train', ])][['label']],
                       'valid': score[score['target'] == 'valid'][['label']]})

    def plot_lift(self, data_end, bin_num=20, col_name='target'):
        time_set = list(data_end[col_name].unique())
        d_lift = {}
        for i in time_set:
            d_lift[i] = toad.metrics.KS_bucket(data_end[data_end[col_name] == i]['score'],
                                               data_end[data_end[col_name] == i]['label'],
                                               bucket=bin_num,
                                               )['cum_lift']

        for i in range(len(time_set)):
            key = time_set[i]
            x = range(len(d_lift[key]))
            y = d_lift[key] if self.type != '营销' else d_lift[key][::-1]
            plt.plot(x, y, linewidth=2, label=key)
            plt.title("lift curve", fontsize=20)
            plt.xlabel("bin number", fontsize=12)
            plt.ylabel("lift value", fontsize=12)
            plt.tick_params(axis='both', labelsize=10)
            plt.legend(loc='best')
            # plt.ylim(0.8, score_ks['lift'].max() * 1.1)

        plt.show()

    def plot_badrate(self, data_end, bin_num=20, col_name='target'):
        time_set = list(data_end[col_name].unique())
        d_lift = {}
        for i in time_set:
            d_lift[i] = toad.metrics.KS_bucket(data_end[data_end[col_name] == i]['score'],
                                               data_end[data_end[col_name] == i]['label'],
                                               bucket=bin_num,
                                               )['bad_rate']

        for i in range(len(time_set)):
            key = time_set[i]
            x = range(len(d_lift[key]))
            y = d_lift[key] if self.type != '营销' else d_lift[key][::-1]
            plt.plot(x, y, linewidth=2, label=key)
            plt.title("bad rate curve", fontsize=20)
            plt.xlabel("bin number", fontsize=12)
            plt.ylabel("bad rate value", fontsize=12)
            plt.tick_params(axis='both', labelsize=10)
            plt.legend(loc='best')
            # plt.ylim(0, score_ks['bad_rate'].max() * 1.1)
        plt.show()

    def show_statistic(self, data_end, bin_num=20, q=None,if_not_score=False):
        if 'label' not in data_end.columns.tolist():
            data_end['label'] = 0
        if q is not None:
            self.score_ks = toad.metrics.KS_bucket(data_end['score'], data_end['label'], q=q)
        else:
            self.score_ks = toad.metrics.KS_bucket(data_end['score'], data_end['label'], bucket=bin_num)

        num = self.score_ks.shape[0]
        if not if_not_score:
            self.score_ks['min'] = self.score_ks.apply(lambda x: int(x['min']), axis=1)
        self.score_ks.loc[0, 'min'] = -float('inf')
        self.score_ks.loc[num - 1, 'max'] = float('inf')
        self.split_points = self.score_ks['min'].tolist()
        # print(self.split_points)
        if self.type == '信贷':
            for i in range(num - 1):
                self.score_ks.loc[i, 'max'] = self.score_ks.loc[i + 1, 'min']
            tb_min, tb_max = self.score_ks['min'].tolist(), self.score_ks['max'].tolist()

            bucket = self.score_ks['min'].tolist()
            bucket.append(float('inf'))
            print(bucket)
            data_end['bucket'] = pd.cut(data_end['score'], bucket, labels=[i for i in range(len(bucket) - 1)],
                                        include_lowest=True, right=False, duplicates='drop')
            # print(data_end['bucket'])
            self.score_ks = toad.metrics.KS_bucket(data_end['score'], data_end['label'], bucket=data_end['bucket'])

            self.score_ks['通过人群整体坏账率'] = self.score_ks.apply(
                lambda x: data_end[data_end['score'] >= x['min']]['label'].mean(), axis=1)
            self.score_ks['通过率'] = self.score_ks.apply(
                lambda x: data_end[data_end['score'] >= x['min']].shape[0] / data_end.shape[0], axis=1)
            self.score_ks['odds'] = 1 / self.score_ks['odds']
            self.score_ks['min'] = sorted(list(set(tb_min)))
            self.score_ks['max'] = sorted(list(set(tb_max)))

        if self.type == '营销':
            for i in range(num - 1):
                self.score_ks.loc[i, 'max'] = self.score_ks.loc[i + 1, 'min']
            tb_min, tb_max = self.score_ks['min'].tolist(), self.score_ks['max'].tolist()
            bucket = sorted(list(set(self.score_ks['min'].tolist())))
            bucket.append(float('inf'))

            print(bucket)
            data_end['bucket'] = pd.cut(data_end['score'], bucket, labels=bucket[:-1],
                                        include_lowest=True, right=False)
            self.score_ks = toad.metrics.KS_bucket(data_end['score'], data_end['label'], bucket=data_end['bucket'])
            self.score_ks['min'] = sorted(list(set(tb_min)))
            self.score_ks['max'] = sorted(list(set(tb_max)))
            self.score_ks.sort_values(by='min', ascending=False, inplace=True)
            self.score_ks['cum_bads_prop'] = self.score_ks['bads'].cumsum() / self.score_ks['bads'].sum()
            self.score_ks['通过率'] = self.score_ks['cum_total_prop_rev']
            self.score_ks['odds'] = 1 / self.score_ks['odds']

            self.score_ks['通过人群整体响应率'] = self.score_ks.apply(
                lambda x: data_end[data_end['score'] >= x['min']]['label'].mean(), axis=1)
        return self.score_ks

    def probability_correction(self, data_end, dep='label', fit_set='train', col_name='target'):
        time_set = list(data_end[col_name].unique())
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end[col_name] == i]

        combiner = toad.transform.Combiner()
        combiner.fit(datasets['train'][['proba']], datasets['train']['label'], method='dt', min_samples=0.05, n_bins=500,
                     exclude=[])
        for v in datasets.values():
            v['proba_bin'] = combiner.transform(v['proba'])

        transer = toad.transform.WOETransformer()
        transer.fit(datasets['train'][['proba_bin']], datasets['train']['label'], exclude=[])
        for v in datasets.values():
            v['proba_woe'] = transer.transform(v['proba_bin'])

        data_end['proba_bin'] = combiner.transform(data_end['proba'])
        data_end['proba_woe'] = transer.transform(data_end['proba_bin'])
        data_end['proba_odd'] = data_end['proba'].apply(lambda x: x / (1 - x))
        data_end['proba_log_odd'] = data_end['proba'].apply(lambda x: np.log(x / (1 - x)))
        odd_all = data_end['label'].mean()/(1-data_end['label'].mean())
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end[col_name] == i]
        print('odd_all: ', odd_all)

        def sigmoid(x):
            f_x = 1.0 / (1 + np.exp(- x.astype(float)))
            return f_x

        self.model_correction, self.model_ce = lr_model(['proba_woe'], dep, fit_set, datasets)
        data_end['proba_correction_woe'] = data_end['proba_woe'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + self.model_correction.intercept_ + np.log(odd_all)))
        data_end['proba_correction_woe_b'] = data_end['proba_woe'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + np.log(odd_all)))
        print(self.model_correction.intercept_)
        self.model_correction, self.model_ce = lr_model(['proba_odd'], dep, fit_set, datasets)
        data_end['proba_correction_odd'] = data_end['proba_odd'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + self.model_correction.intercept_ + np.log(odd_all)))

        self.model_correction, self.model_ce = lr_model(['proba_log_odd'], dep, fit_set, datasets)
        data_end['proba_correction_log_odd'] = data_end['proba_log_odd'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + self.model_correction.intercept_ + np.log(odd_all)))
        print(self.model_correction.intercept_)

    def plot_proba_correction_curve(self, data_end):
        self.score_ks['proba'] = self.score_ks.apply(
            lambda x: data_end[(data_end['score'] >= x['min']) & (data_end['score'] < x['max'])]['proba'].mean(),
            axis=1)
        self.score_ks['proba_correction_woe'] = self.score_ks.apply(
            lambda x: data_end[(data_end['score'] >= x['min']) & (data_end['score'] < x['max'])][
                'proba_correction_woe'].mean(),
            axis=1)
        self.score_ks['proba_correction_odd'] = self.score_ks.apply(
            lambda x: data_end[(data_end['score'] >= x['min']) & (data_end['score'] < x['max'])][
                'proba_correction_odd'].mean(),
            axis=1)
        self.score_ks['proba_correction_log_odd'] = self.score_ks.apply(
            lambda x: data_end[(data_end['score'] >= x['min']) & (data_end['score'] < x['max'])][
                'proba_correction_log_odd'].mean(),
            axis=1)
        self.score_ks['proba_odd'] = self.score_ks.apply(
            lambda x: data_end[(data_end['score'] >= x['min']) & (data_end['score'] < x['max'])][
                'proba_odd'].mean(),
            axis=1)
        self.score_ks['proba_correction_woe_b'] = self.score_ks.apply(
            lambda x: data_end[(data_end['score'] >= x['min']) & (data_end['score'] < x['max'])][
                'proba_correction_woe_b'].mean(),
            axis=1)
        x = range(len(self.score_ks['proba']))
        plt.figure(figsize=(20, 10))
        plt.plot(x, self.score_ks['proba'], linewidth=2, label='proba')
        plt.plot(x, self.score_ks['proba_correction_woe'], linewidth=2, label='proba_correction_woe')
        # plt.plot(x, self.score_ks['proba_correction_odd'], linewidth=2, label='proba_correction_odd')
        plt.plot(x, self.score_ks['bad_rate'], linewidth=2, label='bad_rate')
        plt.plot(x, self.score_ks['proba_correction_log_odd'], linewidth=2, label='proba_correction_log_odd')
        plt.plot(x, self.score_ks['proba_correction_woe_b'], linewidth=2, label='proba_correction_woe_b')
        plt.title("probability correction curve", fontsize=20)
        plt.xlabel("bin number", fontsize=12)
        plt.ylabel("probability", fontsize=12)
        plt.tick_params(axis='both', labelsize=10)
        plt.legend(loc='best')
        plt.show()

    def parameter_fluctuation_curve(self, file_path, threshold,model_type='xgb'):
        res = pd.read_excel(file_path)
        res = res.sort_values(by='value')
        res = res[res['value'] > threshold]

        # t_param = pd.DataFrame()
        # for i in res['model_param']:
        #     t_param = t_param.append(json.loads(i), ignore_index=True)
        t_param_list = []
        for i in res['model_param']:
            temp_df = pd.DataFrame(json.loads(i), index=[0])
            t_param_list.append(temp_df)
        t_param = pd.concat(t_param_list)

        x = range(t_param.shape[0])
        if model_type == 'xgb':
            ft_lst = ['reg_lambda', 'reg_alpha', 'n_estimators', 'max_depth', 'min_child_weight', 'learning_rate', 'gamma',
                    'subsample', "col_sample_bytree", "scale_pos_weight"]#
        elif model_type == 'lgb':
            ft_lst = ['learning_rate','n_estimators','max_depth','num_leaves','min_child_weight',
                    'min_split_gain','min_child_samples','reg_lambda','reg_alpha','colsample_bytree',
                    'max_bin','subsample_for_bin','subsample','subsample_freq']
        min_max_scaler = preprocessing.MinMaxScaler()
        t_param[ft_lst] = min_max_scaler.fit_transform(t_param[ft_lst])
        # plt.figure(figsize=(20, 10))
        # for i in ft_lst:
        #     plt.plot(x, t_param[i], linewidth=2, label=i)
        # plt.title("parameter fluctuation curve", fontsize=20)
        # plt.xlabel("round", fontsize=12)
        # plt.ylabel("probability", fontsize=12)
        # plt.tick_params(axis='both', labelsize=10)
        #
        # plt.legend(loc='best')
        # plt.show()

        plt.figure(figsize=(20, 15))
        for plt_index in range(1, 11):
            # 往画布上添加子图：按三行二列，添加到下标为plt_index的位置
            plt.subplot(3, 4, plt_index)
            key = ft_lst[plt_index-1]
            plt.plot(x, t_param[key], linewidth=2, label=key)

            plt.title(key, fontsize=15)
            plt.xlabel("round", fontsize=12)
            plt.ylabel("MinMaxScaler("+key+")", fontsize=12)
            plt.tick_params(axis='both', labelsize=10)

        plt.show()

    def get_df(self, split_points, res, col_name, num_refused, num_pass):
        num = split_points.shape[0]
        split_points.loc[0, 'min'] = -float('inf')
        split_points.loc[num - 1, 'max'] = float('inf')
        print(num)
        for i in range(num - 1):
            split_points.loc[i, 'max'] = (split_points.loc[i, 'max'] + split_points.loc[i + 1, 'min']) / 2
        for i in range(1, num - 1):
            split_points.loc[i, 'min'] = split_points.loc[i - 1, 'max']
        split_points['拒绝召回'] = split_points.apply(
            lambda x: res[(res[col_name] <= x['max']) & (res[col_name] >= x['min'])].shape[0], axis=1)
        split_points['total_cumsum'] = split_points['total'].cumsum()
        split_points['拒绝召回_cumsum'] = split_points['拒绝召回'].cumsum()
        split_points['召回后通过率'] = split_points.apply(
            lambda x: 1 - (x['total_cumsum'] + x['拒绝召回_cumsum']) / (num_refused + num_pass), axis=1)
        split_points['拒绝样本通过率'] = split_points.apply(lambda x: 1 - (x['拒绝召回_cumsum']) / (num_refused), axis=1)
        return split_points[
            ['min', 'max', 'bads', 'total', 'bad_rate', 'odds', 'cum_total_prop', 'cum_lift', '通过率', '通过人群整体坏账率',
             '拒绝样本通过率']]

    def show(self, col_name, dep, datasets):

        res = pd.DataFrame()
        for k, v in datasets.items():
            fpr_off, tpr_off, _ = roc_curve(v[dep], v[col_name])
            off_ks = abs(fpr_off - tpr_off).max()
            plt.plot(fpr_off, tpr_off, label=k)
            res = res.append({'datasets': k,
                              'auc': auc(fpr_off, tpr_off),
                              # 'precision': precision_score(v[dep], y_pred, average='micro'),
                              # 'recall': recall_score(v[dep], y_pred, average='micro'),
                              'ks': off_ks},
                             ignore_index=True)
            res = res[['datasets', 'auc', 'ks', ]]
        print(res)

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()

        return res

    def get_bad_rate(self, score, datasets):
        res = pd.DataFrame()
        for k, v in datasets.items():
            temp = v[v['score'] >= score]
            res = res.append({'dataset': k,
                              '通过率': temp.shape[0] / v.shape[0],
                              '通过人群坏账率': temp['label'].mean()},
                             ignore_index=True)
        return res

    def probability_correction_std(self, data_end, key_col, dep='label', fit_set='train', col_name='target'):
        time_set = list(data_end[col_name].unique())
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end[col_name] == i]

        combiner = toad.transform.Combiner()
        combiner.fit(datasets['train'][[key_col]], datasets['train']['label'], method='dt', min_samples=0.05, n_bins=500,
                     exclude=[])
        for v in datasets.values():
            v[key_col + '_bin'] = combiner.transform(v[key_col])

        transer = toad.transform.WOETransformer()
        transer.fit(datasets['train'][[key_col + '_bin']], datasets['train']['label'], exclude=[])
        for v in datasets.values():
            v['proba_woe'] = transer.transform(v[key_col + '_bin'])

        data_end[key_col + '_bin'] = combiner.transform(data_end[key_col])
        data_end[key_col + '_woe'] = transer.transform(data_end[key_col + '_bin'])
        data_end[key_col + '_odd'] = data_end['proba'].apply(lambda x: x / (1 - x))
        data_end[key_col + '_log_odd'] = data_end['proba'].apply(lambda x: np.log(x / (1 - x)))
        odd_all = data_end['label'].mean()/(1-data_end['label'].mean())
        datasets = {}
        for i in time_set:
            datasets[i] = data_end[data_end[col_name] == i]
        print('odd_all: ', odd_all)

        def sigmoid(x):
            f_x = 1.0 / (1 + np.exp(- x.astype(float)))
            return f_x

        self.model_correction, self.model_ce = lr_model([key_col + '_woe'], dep, fit_set, datasets)
        data_end['proba_correction_woe'] = data_end[key_col + '_woe'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + self.model_correction.intercept_ + np.log(odd_all)))
        data_end['proba_correction_woe_b'] = data_end[key_col + '_woe'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + np.log(odd_all)))
        print(self.model_correction.intercept_)
        self.model_correction, self.model_ce = lr_model([key_col + '_woe'], dep, fit_set, datasets)
        data_end['proba_correction_odd'] = data_end['proba_odd'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + self.model_correction.intercept_ + np.log(odd_all)))

        self.model_correction, self.model_ce = lr_model(['proba_log_odd'], dep, fit_set, datasets)
        data_end['proba_correction_log_odd'] = data_end['proba_log_odd'].apply(
            lambda x: sigmoid(x * self.model_correction.coef_ + self.model_correction.intercept_ + np.log(odd_all)))
        print(self.model_correction.intercept_)

    
    # def model_performance(self, data_end, keep_lst, col_name='target'):
    #     data_end['proba'] = model.predict_proba(data_end[keep_lst])[:, 1]
    #     data_end['score'] = proba2score(data_end['proba'])
    #     data_end['label'] = data_end[col_name]
    #     data_end['score'] = data_end['score'].astype(int)
    #     data_end['label'] = data_end['label'].astype(int)
    #     data_end['score'] = data_end['score'].astype(int)