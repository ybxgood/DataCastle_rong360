
import pandas as pd
import numpy as np

ac_index_list = [1,3,4,5,6,7,8,10]
bill_cols = ['prior_account', 'prior_repay','credit_limit','current_balance', 'minimun_repay',
             'consume_count','account', 'adjust_account','cycle_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']

def sign(x):
    if x > 0:
        y = 1
    else:
        y = 0
    return y

def get_brow_feats(arr):
    arr1 =  list(arr[:, 1])
    feat_list = []
    tm_lat_list = [x for x in arr[:, 0] if x >= 0]
    tm_ago_list = [x for x in arr[:, 0] if x < 0]

    if len(tm_ago_list) > 0:
        tm_ago_mean = np.mean(tm_ago_list)
    else:
        tm_ago_mean = 0

    feat_list.extend([tm_ago_mean, len(tm_lat_list), len(tm_ago_list)])
    feat_list.extend(arr1.count(x) for x in ac_index_list)

    arr1_ago = [y for (x, y) in arr if x < 0]
    feat_list.extend(arr1_ago.count(x) for x in ac_index_list)

    arr1_lat = [y for (x, y) in arr if x >= 0]
    feat_list.extend(arr1_lat.count(x) for x in ac_index_list)
    return feat_list


def get_bill_feats(tab):
    feat_list = []
    feat_list.append(len(tab['bank_id'].unique()))
    feat_list.append(tab['repay_state'].sum()/(tab.shape[0] + 1))
    col2_list = [('prior_repay','prior_account'),('current_balance','account'),
                 ('minimun_repay','account'),('minimun_repay','current_balance')]

    for col1,col2 in col2_list:
        feat_list.append(tab[col1].mean() / (tab[col2].mean() + 1))
    feat_list.extend(tab[bill_cols].mean().values)

    bef_tab = tab[tab['tm_encode_3'] < 0]
    aft_tab = tab[tab['tm_encode_3'] >= 0]

    if len(bef_tab) == 0:
        feat_list.extend([0]*16)
    else:
        for col1, col2 in col2_list:
            feat_list.append(bef_tab[col1].mean() / (bef_tab[col2].mean() + 1))
        feat_list.extend(bef_tab[bill_cols].mean().values)

    if len(aft_tab) == 0:
        feat_list.extend([0]*16)
    else:
        for col1, col2 in col2_list:
            feat_list.append(aft_tab[col1].mean() / (aft_tab[col2].mean() + 1))
        feat_list.extend(aft_tab[bill_cols].mean().values)

    return feat_list

if __name__ == "__main__":
    tr_addr = '../input/pre/train_'
    te_addr = '../input/pre/test_'

    print 'getting user_info...'
    user_info_train = pd.read_csv(tr_addr + 'user_info.csv')
    user_info_test  = pd.read_csv(te_addr + 'user_info.csv')
    user_info = pd.concat([user_info_train, user_info_test])
    del user_info_train, user_info_test

    print 'getting bank_detail...'
    bank_train = pd.read_csv(tr_addr + 'bank_detail.csv')
    bank_test  = pd.read_csv(te_addr + 'bank_detail.csv')
    bank = pd.concat([bank_train, bank_test])
    bank_has = pd.DataFrame()
    bank_has['bank_has'] = bank['trade_type'].groupby(bank['id']).apply(lambda x:x.count())
    user_info = user_info.merge(bank_has, left_on = 'id', right_index = True, how = 'left')
    user_info['bank_has'] = user_info['bank_has'].apply(lambda x:sign(x))
    del bank_train, bank_test, bank, bank_has

    print 'getting browse_history...'
    browse_train = pd.read_csv(tr_addr + 'browse_history.csv')
    browse_test  = pd.read_csv(te_addr + 'browse_history.csv')
    browse = pd.concat([browse_train, browse_test])
    browse = browse.merge(user_info[['id','loan_time']], on = ['id'], how = 'left')
    browse['tm_encode_2'] = browse['tm_encode_2'] - browse['loan_time']

    brow_feats = pd.DataFrame()
    feat_cols = ['tm2_ago_avg', 'tm2_lat_len', 'tm2_ago_len']
    feat_cols.extend(['acid_all_num_' + str(x) for x in ac_index_list])
    feat_cols.extend(['acid_ago_num_' + str(x) for x in ac_index_list])
    feat_cols.extend(['acid_lat_num_' + str(x) for x in ac_index_list])
    brow_feats['feat_list'] = browse[['tm_encode_2','ac_index']].groupby(browse['id'])\
                                                                .apply(lambda x:get_brow_feats(x.values))
    for i,col in enumerate(feat_cols):
        brow_feats[col] = np.array(list(brow_feats['feat_list'].values))[:, i]
    brow_feats.drop('feat_list', axis = 1, inplace = True)
    del browse_train, browse_test, browse

    print 'getting bill_detail...'
    bill_train = pd.read_csv(tr_addr + 'bill_detail.csv')
    bill_test  = pd.read_csv(te_addr + 'bill_detail.csv')
    bill = pd.concat([bill_train, bill_test])

    mean_val = bill.loc[bill['tm_encode_3'] != 0, 'tm_encode_3'].mean()
    avg_tm_bill = bill.loc[bill['tm_encode_3'] != 0, ['id', 'tm_encode_3']].groupby(['id']).mean()
    avg_tm_bill.rename(columns = {'tm_encode_3': 'tm_fill'}, inplace=True)
    bill = bill.merge(avg_tm_bill, left_on = 'id', right_index = True, how = 'left')
    bill.loc[bill['tm_encode_3'] == 0, 'tm_encode_3'] = bill.loc[bill['tm_encode_3'] == 0, 'tm_fill']
    bill.drop('tm_fill', axis=1, inplace=True)
    bill['tm_encode_3'] = bill['tm_encode_3'].fillna(mean_val)

    bill = bill.merge(user_info[['id', 'loan_time']], on=['id'], how='left')
    bill['tm_encode_3'] = bill['tm_encode_3'] - bill['loan_time']

    bill_feats = pd.DataFrame()
    feat_cols = ['bank_num','repay_statio']
    for pre in ['now','bef','aft']:
        feat_cols.extend([pre +'_ratio' + str(i) for i in xrange(1,5)])
        feat_cols.extend([pre + col for col in bill_cols])

    bill_feats['feat_list'] = bill.groupby(bill['id']).apply(lambda x:get_bill_feats(x))
    for i, col in enumerate(feat_cols):
        bill_feats[col] = np.array(list(bill_feats['feat_list'].values))[:, i]
    bill_feats.drop('feat_list', axis = 1, inplace = True)
    del bill_train, bill_test, bill

    print 'getting train_test...'
    user_info = user_info.merge(brow_feats, left_on='id', right_index=True, how='left')
    user_info = user_info.merge(bill_feats, left_on='id', right_index=True, how='left')
    user_info.fillna(user_info.mean(),inplace = True)
    print user_info.shape
    user_info.to_csv('../input/train_test.csv',index = False)
