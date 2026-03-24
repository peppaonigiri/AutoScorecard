

def get_rule_list(rules):
    rule_list = []
    for i in range(rules.shape[0]):
        variable = rules.iloc[i]['variable']
        bins = rules.iloc[i]['bin'].replace(' ', '')
        bad_rate = rules.iloc[i]['bad_rate']
        rule_dic = {'variable': variable}

        if bins.find('-inf') >= 0:
            rule_dic['type'] = '小于'
            rule_dic['value'] = float(bins.split(',')[1].replace(')', ''))
        elif bins.find(',inf') >= 0:
            rule_dic['type'] = '大于'
            rule_dic['value'] = float(bins.split(',')[0].replace('[', ''))
        elif bins.find('[') >= 0 and bins.find(')') >= 0:

            value1 = float(bins.split(',')[0].replace('[', ''))
            value2 = float(bins.split(',')[1].replace(')', ''))
            rule_dic['type'] = '区间'
            rule_dic['value'] = [value1, value2]
        else:
            rule_dic['type'] = '包含'
            rule_dic['value'] = bins.split(',')

        rule_dic['score'] = round(bad_rate * 100, 2)
        rule_list.append(rule_dic)

    return rule_list


def get_score(datas, rules_lst, flag):
    data = datas.copy()
    for index, dic in enumerate(rules_lst):
        index_str = str(index)
        key = dic['variable']
        r_type = dic['type']
        r_value = dic['value']
        if flag == 'score':
            r_score = dic['score']
        else:
            r_score = 1

        if r_type == '大于':
            data['%s_maps' % index_str] = data[key].apply(
                lambda item: r_score if (float(item)) >= float(r_value) else 0)
        elif r_type == '小于':
            data['%s_maps' % index_str] = data[key].apply(
                lambda item: r_score if (float(item)) < float(r_value) else 0)
        elif r_type == '包含':
            data['%s_maps' % index_str] = data[key].apply(
                lambda item: r_score if item in r_value else 0)
        elif r_type == '区间':
            data['%s_maps' % index_str] = data[key].apply(
                lambda item: r_score if (item >= r_value[0]) and (item < r_value[1]) else 0)

    col_bin = []
    for i in data.columns:
        if i[-5:] == '_maps':
            col_bin.append(i)

    if flag == 'score':
        data['score'] = data[col_bin].sum(axis=1)
        return data['score']
    elif flag == 'cnt':
        data['cnt'] = data[col_bin].sum(axis=1)
        return data['cnt']
    # else:
    #     data['score'] = data[col_bin].sum(axis=1)
    #     data['cnt'] = data[col_bin].sum(axis=1)
    #     return data[['score', 'cnt']]


def get_rules(df_bin_detail, threshold, flag):
    df_bin_detail['bad_rate'] = df_bin_detail['bad'] / df_bin_detail['count']

    if flag == 'bad':
        df_bin_detail = df_bin_detail.sort_values(by='bad_rate', ascending=False)
        rules = df_bin_detail
        # rules = df_bin_detail.drop_duplicates(subset=['variable'], keep='first')
        rules = rules[rules['bad_rate'] >= threshold]
    else:
        df_bin_detail = df_bin_detail.sort_values(by='bad_rate', ascending=True)
        rules = df_bin_detail
        # rules = df_bin_detail.drop_duplicates(subset=['variable'], keep='first')
        rules = rules[rules['bad_rate'] <= threshold]

    print('rules_num: ', rules.shape[0])
    return rules


def get_features(data, df_bin_detail, threshold, flag, flag_creat, flag_type):
    rules = get_rules(df_bin_detail, threshold, flag)
    rule_list = get_rule_list(rules)
    print(rule_list)

    if flag_creat:
        return get_score(data, rule_list, flag_type)


def get_test_code():
    pass
