
def sxxltcsr(data):
    if data['0.5授信额度'] > 80000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 35000
        elif data['EDULEVEL'] in ['专科']:
            return 22000
        else:
            return 18000
    elif data['0.5授信额度'] > 50000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 22000
        elif data['EDULEVEL'] in ['专科']:
            return 15000
        else:
            return 12000
    elif data['0.5授信额度'] > 35000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 14000
        elif data['EDULEVEL'] in ['专科']:
            return 10000
        else:
            return 8000
    elif data['0.5授信额度'] > 25000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 10000
        elif data['EDULEVEL'] in ['专科']:
            return 7500
        else:
            return 6000
    elif data['0.5授信额度'] > 15000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 7000
        elif data['EDULEVEL'] in ['专科']:
            return 5000
        else:
            return 4000

    elif data['0.5授信额度'] > 10000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 5000
        elif data['EDULEVEL'] in ['专科']:
            return 4000
        else:
            return 3500

    elif data['0.5授信额度'] > 5000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 4500
        elif data['EDULEVEL'] in ['专科']:
            return 3500
        else:
            return 4000

    elif data['0.5授信额度'] > 2000:
        if data['EDULEVEL'] in ['本科及以上']:
            return 3000
        elif data['EDULEVEL'] in ['专科']:
            return 2500
        else:
            return 2000

    elif data['0.5授信额度'] > 0:
        return 2000

    else:
        return 0


def gsdj(data):
    if '局' in data['EMPLOYER'] \
            or '厅' in data['EMPLOYER'] \
            or '所' in data['EMPLOYER'] \
            or '办公室'in data['EMPLOYER'] \
            or '处' in data['EMPLOYER'] \
            or '科' in data['EMPLOYER'] \
            or '政府' in data['EMPLOYER']:
        return 1.5
    elif data['公积金缴存推测收入'] > 5000:
        return 1.3
    elif data['公积金缴存推测收入'] > 3000:
        return 1.2
    elif data['公积金缴存推测收入'] > 2000:
        return 1.1
    else:
        return 1


def fdtcsr(data):
    if data['LN_WJQ_HOUSE_MIN_MOB'] > 60:
        return 2.5
    elif data['LN_WJQ_HOUSE_MIN_MOB'] > 48:
        return 2.4
    elif data['LN_WJQ_HOUSE_MIN_MOB'] > 36:
        return 2.3
    elif data['LN_WJQ_HOUSE_MIN_MOB'] > 24:
        return 2.2
    elif data['LN_WJQ_HOUSE_MIN_MOB'] > 12:
        return 2.1
    elif data['LN_WJQ_HOUSE_MIN_MOB'] > 6:
        return 2
    elif data['LN_WJQ_HOUSE_MIN_MOB'] > 0:
        return 1.5
    else:
        return 1


def rhtcsr(data):
    v1 = max(min(data['授信学历推测收入'], data['城市等级收入上限']), data['城市等级收入下限'])
    v2 = max(min(data['车房贷月推测收入'], data['城市等级收入上限']), data['城市等级收入下限'])
    v3 = max(min(data['公积金缴存推测收入'], data['城市等级收入上限']), data['城市等级收入下限'])
    l1 = v1 if v1 > 0 else -9990996.0
    l2 = v2 if v2 > 0 else -9990996.0
    l3 = v3 if v3 > 0 else -9990996.0
    lst = sorted([l1, l2, l3])
    if lst[1] > 0:
        return lst[2] * 0.7 + lst[1] * 0.3
    elif lst[2] > 0:
        return lst[2]
    else:
        return 0


def high_quality_limit(data):

    def fangdai_coef(data):
        if data['LN_WJQ_HOUSE_MIN_MOB'] >= 6:
            if data['房贷月供'] > 3000:
                return 2
            elif data['房贷月供'] > 2000:
                return 1.5
            elif data['房贷月供'] > 1000:
                return 1.2
            elif data['房贷月供'] > 500:
                return 1.1
            else:
                return 1
        else:
            return 1

    def daijika_coef(data):
        if data['DJK_BAL_GE0_MAX_HIS'] >= 6:
            if data['MAXCREDITLIMITPERORG'] > 80000:
                return 2
            elif data['MAXCREDITLIMITPERORG'] > 50000:
                return 1.5
            elif data['MAXCREDITLIMITPERORG'] > 25000:
                return 1.2
            elif data['MAXCREDITLIMITPERORG'] > 15000:
                return 1.1
            else:
                return 1
        else:
            return 1

    def gongjijin_coef(data):
        if data['GJJ_BAL_GE0_MAX_HIS'] >= 6:
            if data['公积金缴存推测收入'] > 20000:
                return 2
            elif data['公积金缴存推测收入'] > 10000:
                return 1.5
            elif data['公积金缴存推测收入'] > 5000:
                return 1.2
            elif data['公积金缴存推测收入'] > 3000:
                return 1.1
            else:
                return 1
        else:
            return 1

    coef = max(fangdai_coef(data), daijika_coef(data), gongjijin_coef(data))
    if coef == 3:
        return 30000
    elif coef == 2.8:
        return 25000
    elif coef == 2.5:
        return 20000
    elif coef == 2:
        return 10000
    elif coef == 1.5:
        return 8000
    elif coef == 1.2:
        return 5000
    elif coef == 1.1:
        return 3000
    else:
        return 0

