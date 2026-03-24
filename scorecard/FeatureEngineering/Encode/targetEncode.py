import category_encoders as ce
import pandas as pd

valid = self.New_Catagorical_Levels_in_TestData(train, valid ,label)

train, valid, dict_target = self.Target_Encode(train, valid, label, categorical_features)


def New_Catagorical_Levels_in_TestData(train, test, label, replacement_strategy='most frequent'):

    ph_train_level = pd.DataFrame(columns=train.drop(label, axis=1).select_dtypes(include="object").columns)

    for i in ph_train_level.columns:
        if replacement_strategy == "least frequent":
            ph_train_level.loc[0, i] = list(train[i].value_counts().sort_values().index)
        else:
            ph_train_level.loc[0, i] = list(train[i].value_counts().index)

    ph_test_level = pd.DataFrame(columns=test.drop(label, axis=1, errors='ignore').select_dtypes(include="object").columns)
    for i in ph_test_level.columns:
        ph_test_level.loc[0, i] = list(
            test[i].value_counts().sort_values().index)

    for i in ph_test_level.columns:
        new = list((set(ph_test_level.loc[0, i]) - set(ph_train_level.loc[0, i])))
        # now if there is a difference , only then replace it
        if len(new) > 0:
            test[i].replace(new, ph_train_level.loc[0, i][0], inplace=True)
    return test
    
    
def Target_Encode(trains, tests, label, cat_fea):
    train = trains.copy()
    test = tests.copy()
    cat_fea = [col for col in cat_fea if col in train.columns.tolist()]

    target_enc = ce.TargetEncoder(cols=cat_fea)
    target_enc.fit(train[cat_fea], train[label])
    train = train.join(target_enc.transform(train[cat_fea]).add_suffix('_enc'))
    test = test.join(target_enc.transform(test[cat_fea]).add_suffix('_enc'))
    train = train.drop(columns=cat_fea)
    test = test.drop(columns=cat_fea)
    train.columns = [col.replace('_enc', '') for col in train.columns]
    test.columns = [col.replace('_enc', '') for col in test.columns]
    dict_target = {}
    for col in cat_fea:
        dict_value = target_enc.mapping.get(col).to_dict()
        for dic in target_enc.ordinal_encoder.mapping:
            if dic.get('col') == col:
                dict_key = dic.get('mapping').to_dict()
        dict_kv = {}
        for k, v in dict_key.items():
            dict_kv[k] = dict_value.get(v)
        dict_target[col] = dict_kv

    return train, test, dict_target


