
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense


class MyDataset(object):
    def __init__(self, train):
        self.data = train
        os = ['OS_1', 'OS_2', 'OS_3', 'OS_4', 'OS_5', 'OS_6', 'OS_7', 'OS_8', 'OS_9', 'OS_10', 'OS_11', 'OS_12', ]
        payment = ['Payment_1', 'Payment_2', 'Payment_3', 'Payment_4', 'Payment_5', 'Payment_6', 'Payment_7',
                   'Payment_8', 'Payment_9', 'Payment_10', 'Payment_11', 'Payment_12', ]
        spend = ['Spend_1', 'Spend_2', 'Spend_3', 'Spend_4', 'Spend_5', 'Spend_6', 'Spend_7', 'Spend_8', 'Spend_9',
                 'Spend_10', 'Spend_11', 'Spend_12', ]
        delq1 = ['Delq1_1', 'Delq1_2', 'Delq1_3', 'Delq1_4', 'Delq1_5', 'Delq1_6', 'Delq1_7', 'Delq1_8', 'Delq1_9',
                 'Delq1_10', 'Delq1_11', 'Delq1_12', ]
        delq2 = ['Delq2_1', 'Delq2_2', 'Delq2_3', 'Delq2_4', 'Delq2_5', 'Delq2_6', 'Delq2_7', 'Delq2_8', 'Delq2_9',
                 'Delq2_10', 'Delq2_11', 'Delq2_12', ]
        delq3 = ['Delq3_1', 'Delq3_2', 'Delq3_3', 'Delq3_4', 'Delq3_5', 'Delq3_6', 'Delq3_7', 'Delq3_8', 'Delq3_9',
                 'Delq3_10', 'Delq3_11', 'Delq3_12']

        os = train[os]
        payment = train[payment]
        spend = train[spend]
        delq1 = train[delq1]
        delq2 = train[delq2]
        delq3 = train[delq3]

        data = pd.concat([os, payment, spend, delq1, delq2, delq3], axis=1)
        self.feature = data.values.reshape(train.shape[0], 12, 6, order='F')
        self.label = train['label_v']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


train_data = MyDataset(train_cor)
test_data = MyDataset(val_cor.reset_index())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle=False, num_workers=0)


# %%

class Lstm(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Lstm, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)  # input_size,hidden_size,num_layers
        self.linear = nn.Linear(hidden_dim, 1)  # input = 12, output = 2
        self.sigmoid = nn.Sigmoid()  # input = 2, output = 2

    def forward(self, x):
        #         x = x.sum(dim = 1)
        x = torch.tensor(x, dtype=torch.float32)
        out, _ = self.LSTM(x)
        out = out[:, -1, :]
        out = self.linear(out)
        out = self.sigmoid(out)
        return out

    def predict(self, pred):
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(t[0])
            else:
                ans.append(t[1])
        return torch.tensor(ans)

    def lstm(self, n_feature, n_month, n_hidden, n_class=2):
        """
        3个特征，12个月切片，2个隐层，2分类
        :return:
        """

        model = Lstm(n_feature, n_month, n_hidden, n_class)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 使用二分类对数损失函数
        criterion = nn.SoftMarginLoss(reduction='mean')
        opt = torch.optim.Adam(model.parameters())
        # criterion = nn.BCELoss()
        # opt = torch.optim.Adam(model.parameters())

        total_step = len(train_loader)
        total_step_test = len(test_loader)
        num_epochs = 500

        for epoch in range(num_epochs):
            train_label = []
            train_pred = []
            model.train()
            for i, (feature, labels) in enumerate(train_loader):
                feature = feature.to(device)
                labels = labels.to(device)

                # 网络训练
                out = model(feature)

                loss = criterion(out[:, 0], labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # 每一百轮打印一次
                if i % 100 == 0:
                    print('train epoch: {}/{}, round: {}/{},loss: {}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                             loss))
                    # 真实标记和预测值
                train_label.extend(labels.cpu().numpy().flatten().tolist())
                train_pred.extend(out.detach().cpu().numpy().flatten().tolist())
            # 计算真正率和假正率

            fpr_lm_train, tpr_lm_train, _ = roc_curve(np.array(train_label), np.array(train_pred))
            # 计算KS和AUC
            print('train epoch: {}/{}, KS: {}, ROC: {}'.format(
                epoch + 1, num_epochs, abs(fpr_lm_train - tpr_lm_train).max(), auc(fpr_lm_train, tpr_lm_train)))

            test_label = []
            test_pred = []

            model.eval()
            # 计算测试集上的KS值和AUC值
            for i, (feature, labels) in enumerate(test_loader):
                feature = feature.to(device)
                labels = labels.to(device)
                out = model(feature)
                loss = criterion(out[:, 0], labels)

                # 计算KS和AUC
                if i % 100 == 0:
                    print(
                        'test epoch: {}/{}, round: {}/{},loss: {}'.format(epoch + 1, num_epochs, i + 1, total_step_test,
                                                                          loss))
                test_label.extend(labels.cpu().numpy().flatten().tolist())
                test_pred.extend(out.detach().cpu().numpy().flatten().tolist())

            fpr_lm_test, tpr_lm_test, _ = roc_curve(np.array(test_label), np.array(test_pred))
            print('test epoch: {}/{}, KS: {}, ROC: {}'.format(
                epoch + 1, num_epochs, abs(fpr_lm_test - tpr_lm_test).max(), auc(fpr_lm_test, tpr_lm_test)))


os_lst = ['OS_0', 'OS_1', 'OS_2', 'OS_3', 'OS_4', 'OS_5', 'OS_6', 'OS_7', 'OS_8', 'OS_9', 'OS_10', 'OS_11', 'OS_12', ]
payment_lst = ['Payment_1', 'Payment_2', 'Payment_3', 'Payment_4', 'Payment_5', 'Payment_6', 'Payment_7', 'Payment_8',
               'Payment_9', 'Payment_10', 'Payment_11', 'Payment_12', ]
spend_lst = ['Spend_1', 'Spend_2', 'Spend_3', 'Spend_4', 'Spend_5', 'Spend_6', 'Spend_7', 'Spend_8', 'Spend_9',
             'Spend_10', 'Spend_11', 'Spend_12', ]

os = train[os_lst]
payment = train[payment_lst]
spend = train[spend_lst]
payment.insert(0, 'mask_payment', 0)  ##序列长度不相等进行补0，后续要使用mask进行屏蔽/
spend.insert(0, 'mask_spend', 0)

sequence_data = pd.concat([os, payment, spend], axis=1)
sequence_data = sequence_data.values.reshape(len(train), 13, 3, order='F')
sequence_data

os = val[os_lst]
payment = val[payment_lst]
spend = val[spend_lst]
payment.insert(0, 'mask_payment', 0)  ##序列长度不相等进行补0，后续要使用mask进行屏蔽
spend.insert(0, 'mask_spend', 0)

sequence_data_val = pd.concat([os, payment, spend], axis=1)
sequence_data_val = sequence_data_val.values.reshape(len(val), 13, 3, order='F')
sequence_data_val

os = off[os_lst]
payment = off[payment_lst]
spend = off[spend_lst]
payment.insert(0, 'mask_payment', 0)  ##序列长度不相等进行补0，后续要使用mask进行屏蔽
spend.insert(0, 'mask_spend', 0)

sequence_data_off = pd.concat([os, payment, spend], axis=1)
sequence_data_off = sequence_data_off.values.reshape(len(off), 13, 3, order='F')
sequence_data_off

# %%

inputs1 = Input(shape=(13, 3))
lstm1, state_h, state_c = LSTM(128, activation='tanh', return_sequences=True, return_state=True, recurrent_dropout=0.5)(
    inputs1)
outputs = Dense(1, activation='sigmoid')(state_h)
model_1 = Model(inputs=inputs1, outputs=[outputs])
model_1.compile(loss='binary_crossentropy', optimizer='adam')

# %%

model_1.fit(sequence_data, y, batch_size=100, epochs=50, verbose=10)

# %%

get_hidden_state = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])

# %%

results = get_hidden_state.predict(sequence_data)
results_val = get_hidden_state.predict(sequence_data_val)
results_off = get_hidden_state.predict(sequence_data_off)

# %%

tf_add = pd.DataFrame(results[2])
tf_add_val = pd.DataFrame(results_val[2])
tf_add_off = pd.DataFrame(results_off[2])

for col in tf_add.columns:
    #     X[col]=times_feature[col].values
    tf_add.rename(columns={col: 'tf_' + str(col)}, inplace=True)
    tf_add_val.rename(columns={col: 'tf_' + str(col)}, inplace=True)
    tf_add_off.rename(columns={col: 'tf_' + str(col)}, inplace=True)

# %%

tf_add_off

# %%

train_data_1 = pd.merge(trainData, tf_add, how='left', left_index=True, right_index=True)
val_data_1 = pd.merge(valData, tf_add_val, how='left', left_index=True, right_index=True)
off_data_1 = pd.merge(offData, tf_add_off, how='left', left_index=True, right_index=True)
