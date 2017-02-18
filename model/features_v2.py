
import pandas as pd
import numpy as np

bill_cols = ['tm_encode_3','prior_account', 'prior_repay','credit_limit','current_balance', 'minimun_repay',
             'consume_count','account', 'adjust_account','cycle_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']

def sign(x):
    if x > 0:
        y = 1
    else:
        y = 0
    return y

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

    for i in [1,2]:
        if i == 1:
            browse_temp = browse[browse['tm_encode_2'] >= 0]
        else:
            browse_temp = browse[browse['tm_encode_2'] < 0]

        brow_feats = browse_temp[['tm_encode_2', 'action']].groupby(browse_temp['id']).mean()
        user_info = user_info.merge(brow_feats, left_on='id', right_index=True, how='left')
        user_info.fillna(user_info.mean(), inplace=True)

        brow_feats = browse_temp['ac_index'].groupby([browse_temp['id'], browse_temp['ac_index']]).count()
        brow_feats = brow_feats.unstack()
        brow_feats.columns = [str(i) + '_ac_index_count_' + str(x) for x in xrange(1,12)]
        user_info = user_info.merge(brow_feats, left_on='id', right_index=True, how='left')

        brow_feats = browse_temp['action'].groupby([browse_temp['id'], browse_temp['ac_index']])\
                                                          .agg(['sum','mean'])
        brow_feats = brow_feats.unstack()
        feat_cols = [str(i) + '_sum_action_' + str(x) for x in xrange(1,12)]
        feat_cols.extend([str(i) + '_mean_action_' + str(x) for x in xrange(1,12)])

        brow_feats.columns = feat_cols
        user_info = user_info.merge(brow_feats, left_on='id', right_index=True, how='left')
        user_info.fillna(0, inplace=True)

    del browse_train, browse_test, browse, browse_temp, brow_feats


    print 'getting bill_detail...'
    bill_train = pd.read_csv(tr_addr + 'bill_detail.csv')
    bill_test  = pd.read_csv(te_addr + 'bill_detail.csv')
    bill = pd.concat([bill_train, bill_test])

    mean_val = bill.loc[bill['tm_encode_3'] != 0, 'tm_encode_3'].mean()
    avg_tm_bill = bill.loc[bill['tm_encode_3'] != 0, ['id', 'tm_encode_3']].groupby(['id']).mean()
    avg_tm_bill.rename(columns = {'tm_encode_3': 'tm_fill'}, inplace=True)
    bill = bill.merge(avg_tm_bill, left_on = 'id', right_index = True, how = 'left')
    bill.loc[bill['tm_encode_3'] == 0, 'tm_encode_3'] = bill.loc[bill['tm_encode_3'] == 0, 'tm_fill']
    bill.drop('tm_fill', axis = 1, inplace = True)
    bill['tm_encode_3'] = bill['tm_encode_3'].fillna(mean_val)

    bill = bill.merge(user_info[['id', 'loan_time']], on=['id'], how='left')
    bill['tm_encode_3'] = (bill['tm_encode_3'] - bill['loan_time'])/(3600*24*1000)
    split_list_1 = [2, 0.6, 0.2, 0, 0.04, 0.08, 0.15]
    split_list_2 = [0.6, 0.2, 0, 0.04, 0.08, 0.15, 0.4]
    col2_list = [('prior_repay', 'prior_account'), ('current_balance', 'account'),
                 ('minimun_repay', 'account'), ('minimun_repay', 'current_balance')]

    for  i,(low,up) in enumerate(zip(split_list_1, split_list_2)):
        bill_temp =  bill[bill['tm_encode_3'] > low]
        bill_temp =  bill_temp[bill_temp['tm_encode_3'] < up]
        group_feats = bill_temp[bill_cols].groupby(bill_temp['id']).agg(['min','max','mean'])
        feat_cols = []
        for col in [str(i + 1) + col for col in bill_cols]:
            feat_cols.extend(['min_'+col, 'max_'+col, 'mean_'+col ])
        group_feats.columns = feat_cols
        for j, (col1, col2) in enumerate(col2_list):
            col1, col2  = 'mean_'+ str(i+1) + col1, 'mean_'+ str(i+1) + col2
            group_feats[str(i) + '_bill_r_' + str(j)] = (group_feats[col1].mean() / (group_feats[col2].mean() + 1))
        user_info = user_info.merge(group_feats, left_on='id', right_index= True , how='left')


    del bill_train, bill_test, bill, bill_temp, group_feats
    print 'getting train_test...'
    user_info.fillna(0, inplace=True)
    print user_info.shape
    user_info.to_csv('../input/train_test.csv',index = False)