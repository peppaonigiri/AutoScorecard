
import pandas as pd
import numpy as np
import scorecardpy as sc


class ShowBins(object):
    def __init__(self, data, ex_lst, dep='label'):
        data = data.fillna(-999)
        self.dev = data[data['target'] == 'train']
        self.val = data[data['target'] == 'valid']
        self.oot = data[data['target'] == 'oot']
        self.ex_lst = ex_lst
        self.ft_lst = [i for i in data.columns if i not in ex_lst]
        self.dep = dep

    def toad_show_bins(self, data, combiner, transer):
        data = combiner.transform(data)
        BINS = {}
        for i in self.ft_lst:
            count = data.groupby([i])['label'].count()
            bad = data.groupby([i])['label'].sum()
            res = pd.DataFrame({'count': count, 'bad': bad})
            res['good'] = res['count'] - res['bad']
            res['value'] = res.index
            res['variable'] = i

            bins = []
            sp_l = [float('-inf')] + list(combiner[i]) + [float('inf')]
            for j in range(len(sp_l) - 1):
                bins.append('[' + str(sp_l[j]) + ', ' + str(sp_l[j + 1]) + ')')
            n_bin = list(data[i].unique())
            n_bin.sort()
            bins_j = []

            for k in n_bin:
                bins_j.append(bins[k])
            res['bin'] = bins_j

            res['count_distr'] = res['count'] / res['count'].sum()
            res['bad_cumrate'] = res['bad'].cumsum() / res['bad'].sum()
            res['good_cumrate'] = res['good'].cumsum() / res['good'].sum()
            res['ks'] = abs(res['bad_cumrate'] - res['good_cumrate'])

            res['badprob'] = res['bad'] / (res['bad'] + res['good'])

            res['bad_prob'] = res['bad'] / res['bad'].sum()
            res['good_prob'] = res['good'] / res['good'].sum()
            res['bad_prob'] = res['bad_prob'].apply(lambda x: 0.0001 if x == 0 else x)
            res['good_prob'] = res['good_prob'].apply(lambda x: 0.0001 if x == 0 else x)

            res['iv'] = (res['bad_prob'] - res['good_prob']) * np.log(res['bad_prob'] / res['good_prob'])
            res['total_iv'] = res['iv'].sum()
            res['total_ks'] = res['ks'].max()
            woe_j = []
            for k in n_bin:
                woe_j.append(transer[i]['woe'][k])
            res['woe'] = woe_j

            BINS[i] = res[
                ['variable', 'value', 'bin', 'count', 'count_distr', 'good', 'good_prob', 'bad', 'bad_prob', 'iv',
                 'total_iv', 'woe', 'ks', 'total_ks', 'badprob']]

        return BINS

    def optbinning_show_bins(self, data, otpb_dict):
        BINS = {}
        for key, value in otpb_dict.items():
            temp = data[[key, self.dep]].copy()
            temp['indices'] = value.transform(data[key], metric="indices")
            binning_table = value.binning_table.build().iloc[:temp['indices'].max() + 1, :]

            count = temp.groupby(['indices'])['label'].count()
            bad = temp.groupby(['indices'])['label'].sum()
            res = pd.DataFrame({'count': count, 'bad': bad})
            res['good'] = res['count'] - res['bad']
            res['value'] = res.index
            res['variable'] = key

            res['bin'] = binning_table['Bin']

            res['count_distr'] = res['count'] / res['count'].sum()
            res['bad_cumrate'] = res['bad'].cumsum() / res['bad'].sum()
            res['good_cumrate'] = res['good'].cumsum() / res['good'].sum()
            res['ks'] = abs(res['bad_cumrate'] - res['good_cumrate'])

            res['badprob'] = res['bad'] / (res['bad'] + res['good'])

            res['bad_prob'] = res['bad'] / res['bad'].sum()
            res['good_prob'] = res['good'] / res['good'].sum()
            res['bad_prob'] = res['bad_prob'].apply(lambda x: 0.0001 if x == 0 else x)
            res['good_prob'] = res['good_prob'].apply(lambda x: 0.0001 if x == 0 else x)

            res['iv'] = (res['bad_prob'] - res['good_prob']) * np.log(res['bad_prob'] / res['good_prob'])
            res['total_iv'] = res['iv'].sum()
            res['total_ks'] = res['ks'].max()
            res['woe'] = binning_table['WoE']

            BINS[key] = res[
                ['variable', 'value', 'bin', 'count', 'count_distr', 'good', 'good_prob', 'bad', 'bad_prob', 'iv',
                 'total_iv', 'woe', 'ks', 'total_ks', 'badprob']]

        return BINS

    def show_bins_detail(self, data, tool_name, combiner=None, transer=None, otpb_dict=None):
        if tool_name == 'toad':
            return self.toad_show_bins(data, combiner, transer)
        elif tool_name == 'optbinning':
            return self.optbinning_show_bins(data, otpb_dict)
        elif tool_name == 'scorecardpy':
            return sc.woebin(self.dev[self.ft_lst], self.dev[self.dep])

    def datasets_binning_table(self, tool_name, combiner=None, transer=None, otpb_dict=None):
        if self.oot.shape[0] > 0:
            datasets = {'train': self.dev, 'valid': self.val, 'oot': self.oot}
        else:
            datasets = {'train': self.dev, 'valid': self.val}

        res = {}
        for key, value in datasets.items():
            res[key] = self.show_bins_detail(value, tool_name, combiner=combiner, transer=transer, otpb_dict=otpb_dict)

        return res

    def bin_plot(self, ft_lst, datasets_binning_table):
        from PIL import Image

        def fig2img(fig):
            fig.savefig('temp.png')
            return Image.open('temp.png')

        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # （解决坐标轴负数的负号显示问题）

        res = {}

        test = sc.woebin_plot(datasets_binning_table['train'][ft_lst[0]])[ft_lst[0]]
        width, height = fig2img(test).size[0], fig2img(test).size[1]

        num_datasets = len(datasets_binning_table)

        for i in ft_lst:
            concat_by_ft = Image.new("RGB", (width, height * num_datasets))
            pos = 0
            for k, v in datasets_binning_table.items():
                p = sc.woebin_plot(v[i])[i]
                concat_by_ft.paste(fig2img(p), box=(0, pos * height))
                pos += 1

            res[i] = concat_by_ft

        col, row = 4, len(res) // 4 + 1
        concat_all = Image.new("RGB", (width * col, height * num_datasets * row), (255, 255, 255))

        for i in range(len(ft_lst)):
            pos_x, pos_y = i % col, i // col
            concat_all.paste(res[ft_lst[i]], box=(pos_x * width, pos_y * num_datasets * height))

        return res, concat_all






