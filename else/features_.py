import numpy as np
import pandas as pd

def main():
    print 'getting loan_time...'
    loan_time_cols = ['id', 'loan_time']
    loan_time_train = pd.read_table('../data/train/loan_time_train.txt', sep=',', names=loan_time_cols)
    loan_time_test = pd.read_table('../data/test/loan_time_test.txt', sep=',', names=loan_time_cols)
    loan_time = pd.concat([loan_time_train, loan_time_test])
    print 'loan_time shape: ', loan_time.shape
    loan_time.index = loan_time['id']
    loan_time.drop('id',
               axis = 1,
               inplace = True)
    print 'loan_time done.'


    print 'getting user_info...'
    user_info_columns = ['id', 'sex', 'job', 'edu', 'marriage', 'loc_type']
    user_info_train = pd.read_table('../data/train/user_info_train.txt', 
                                    sep=',', names=user_info_columns)
    user_info_test = pd.read_table('../data/test/user_info_test.txt',
                                    sep=',', names=user_info_columns)
    user_info = pd.concat([user_info_train, user_info_test])
    print 'user_info shape: ', user_info.shape
    user_info.index = user_info['id']
    user_info.drop('id',
                axis = 1,
                inplace = True)
    print 'user_info done.'


    print 'getting bank_detail...'
    bank_cols = ['id', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
    bank_detail_train = pd.read_table('../data/train/bank_detail_train.txt', sep=',', names=bank_cols)
    bank_detail_test = pd.read_table('../data/test/bank_detail_test.txt', sep=',', names=bank_cols)
    bank_detail = pd.concat([bank_detail_train, bank_detail_test])
    print 'bank_detail shape: ', bank_detail.shape

    # min_tm = (bank_detail.loc[bank_detail['tm_encode'] != 0, ['id', 'tm_encode']]).groupby(['id']).min()
    # min_tm['tm_fill'] = min_tm['tm_encode']
    # min_tm['tm_fill'] -= 2e7
    # min_tm.drop('tm_encode', axis=1, inplace=True)
    # bank_detail = bank_detail.join(min_tm, on='id')
    # bank_detail.loc[bank_detail['tm_encode'] == 0, 'tm_encode'] = bank_detail.loc[bank_detail['tm_encode'] == 0, 'tm_fill']
    # bank_detail.drop('tm_fill', axis=1, inplace=True)

    bank_detail.loc[bank_detail['tm_encode'] == 0, 'tm_encode'] = bank_detail.loc[bank_detail['tm_encode'] != 0, 'tm_encode'].mean()

    bank_detail_n = (bank_detail.loc[:, ['id', 'trade_type', 'trade_amount', 'tm_encode']]).groupby(['id', 'trade_type']).mean()
    bank_detail_n = bank_detail_n.unstack()
    bank_detail_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']

    sal_type = (bank_detail.loc[:, ['id','salary_tag']]).groupby(['id']).sum()
    sal_type['sal_type_cat'] = sal_type['salary_tag']
    sal_type.loc[sal_type['sal_type_cat']>0, 'sal_type_cat'] = 1
    bank_detail_n = bank_detail_n.join(sal_type, how='outer')

    bank_detail['help_count'] = np.ones((len(bank_detail, )))
    in_out_num = (bank_detail.loc[:, ['id','trade_type', 'help_count']]).groupby(['id']).sum()
    in_out_num['out_count'] = in_out_num['trade_type']
    in_out_num['in_count'] = in_out_num['help_count'] - in_out_num['out_count']
    in_out_num.drop(['trade_type', 'help_count'], axis=1, inplace=True)
    bank_detail_n = bank_detail_n.join(in_out_num, how='outer')

    bank_detail = bank_detail.join(loan_time, on='id')
    bank_detail['over_loan_time_flag'] = bank_detail['tm_encode'] - bank_detail['loan_time']
    bank_detail.loc[bank_detail['over_loan_time_flag']>0, 'over_loan_time_flag'] = 1
    bank_detail.loc[bank_detail['over_loan_time_flag']<0, 'over_loan_time_flag'] = 0

    bank_detail_over_loan = (bank_detail.loc[:, ['id', 'over_loan_time_flag', 'trade_amount', 'trade_type']]).groupby(['id', 'trade_type', 'over_loan_time_flag']).mean()
    bank_detail_over_loan = bank_detail_over_loan.unstack(level=1).unstack()
    bank_detail_over_loan.columns = ['income_bef', 'income_aft', 'outcome_bef', 'outcome_aft']
    bank_detail_over_loan.fillna(0, inplace=True)

    in_out_num_over_loan = (bank_detail.loc[:, ['id','trade_type', 'help_count', 'over_loan_time_flag']]).groupby(['id', 'over_loan_time_flag']).sum()
    in_out_num_over_loan['out_count'] = in_out_num_over_loan['trade_type']
    in_out_num_over_loan['in_count'] = in_out_num_over_loan['help_count'] - in_out_num_over_loan['out_count']
    in_out_num_over_loan.drop(['trade_type', 'help_count'], axis=1, inplace=True)
    in_out_num_over_loan = in_out_num_over_loan.unstack()
    in_out_num_over_loan.columns = ['out_count_bef', 'out_count_aft', 'in_count_bef', 'in_count_aft']
    in_out_num_over_loan.fillna(0, inplace=True)

    sal_type_over_loan = (bank_detail.loc[:, ['id','salary_tag', 'over_loan_time_flag']]).groupby(['id', 'over_loan_time_flag']).sum()
    sal_type_over_loan['sal_type_cat'] = sal_type_over_loan['salary_tag']
    sal_type_over_loan.loc[sal_type_over_loan['sal_type_cat']>0, 'sal_type_cat'] = 1
    sal_type_over_loan = sal_type_over_loan.unstack()
    sal_type_over_loan.columns = ['salary_tag_bef', 'salary_tag_aft', 'sal_type_cat_bef', 'sal_type_cat_aft']
    sal_type_over_loan.fillna(0, inplace=True) 

    bank_detail_n = bank_detail_n.join(bank_detail_over_loan)
    bank_detail_n = bank_detail_n.join(in_out_num_over_loan)
    bank_detail_n = bank_detail_n.join(sal_type_over_loan)
    del(bank_detail, bank_detail_over_loan, in_out_num_over_loan, sal_type_over_loan)
    print 'bank_detail done.'

# ##############Ori#####################
#     print 'getting browse_history...'
#     browse_cols = ['id', 'time', 'action', 'ac_index']
#     browse_train = pd.read_table('../data/train/browse_history_train.txt', sep=',', names=browse_cols)
#     browse_test = pd.read_table('../data/test/browse_history_test.txt', sep=',', names=browse_cols)
#     browse = pd.concat([browse_train, browse_test])
#     #browse.time = browse.time.fillna(browse.time.mean())
#     print 'browse_history shape: ', browse.shape

#     browse_id_time = (browse.loc[:, ['id', 'time']]).groupby(['id']).mean()
#     browse_n = (browse.loc[:, ['id', 'action', 'ac_index']]).groupby(['id', 'ac_index']).sum()
#     browse_n = browse_n.unstack()
#     browse_n.columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']
#     browse_n.fillna(0, inplace=True)
#     browse_id_time.columns =['browse_time']
#     browse_n = browse_n.join(browse_id_time, how = 'outer')

#     b_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']
#     browse_n['average_b'] = np.sum(browse_n.loc[:, b_cols], axis=1) / 11.
#     browse_ave = (browse.loc[:, ['id', 'action', 'ac_index']]).groupby(['id', 'ac_index']).mean()
#     browse_ave = browse_ave.unstack()
#     browse_ave.columns = ['b1_ave', 'b2_ave', 'b3_ave', 'b4_ave', 'b5_ave', 'b6_ave', 'b7_ave', 
#                         'b8_ave', 'b9_ave', 'b10_ave', 'b11_ave']
#     browse_ave.fillna(0, inplace=True)
#     browse_n = browse_n.join(browse_ave, how='outer')

#     browse = browse.join(loan_time, on='id')
#     browse['over_loan_time_flag'] = browse['time'] - browse['loan_time']
#     browse.loc[browse['over_loan_time_flag']>0, 'over_loan_time_flag'] = 1
#     browse.loc[browse['over_loan_time_flag']<0, 'over_loan_time_flag'] = 0

#     browse_n_over_loan = browse.loc[:, ['id', 'action', 'ac_index', 'over_loan_time_flag']].groupby(['id', 'ac_index', 'over_loan_time_flag']).sum()
#     browse_n_over_loan = browse_n_over_loan.unstack(level=1).unstack()
#     browse_n_over_loan.columns = ['b1_bef', 'b1_aft', 'b2_bef', 'b2_aft', 'b3_bef', 'b3_aft', 'b4_bef', 'b4_aft',
#                              'b5_bef', 'b5_aft', 'b6_bef', 'b6_aft', 'b7_bef', 'b7_aft', 'b8_bef', 'b8_aft',
#                              'b9_bef', 'b9_aft', 'b10_bef', 'b10_aft', 'b11_bef', 'b11_aft']
#     browse_n_over_loan.fillna(0, inplace=True)

#     browse_n_over_loan_ave = browse.loc[:, ['id', 'action', 'ac_index', 'over_loan_time_flag']].groupby(['id', 'ac_index', 'over_loan_time_flag']).mean()
#     browse_n_over_loan_ave = browse_n_over_loan_ave.unstack(level=1).unstack()
#     browse_n_over_loan_ave.columns = ['b1_ave_bef', 'b1_ave_aft', 'b2_ave_bef', 'b2_ave_aft', 'b3_ave_bef', 'b3_ave_aft', 'b4_ave_bef', 'b4_ave_aft',
#                              'b5_ave_bef', 'b5_ave_aft', 'b6_ave_bef', 'b6_ave_aft', 'b7_ave_bef', 'b7_ave_aft', 'b8_ave_bef', 'b8_ave_aft',
#                              'b9_ave_bef', 'b9_ave_aft', 'b10_ave_bef', 'b10_ave_aft', 'b11_ave_bef', 'b11_ave_aft']
#     browse_n_over_loan_ave.fillna(0, inplace=True)

#     browse_n = browse_n.join(browse_n_over_loan, how='outer')
#     browse_n = browse_n.join(browse_n_over_loan_ave, how='outer')
#     del(browse, browse_n_over_loan, browse_n_over_loan_ave)
#     print 'browse_history done.'
# ######################################

    print 'getting browse_history...'
    browse_cols = ['id', 'time', 'action', 'ac_index']
    browse_train = pd.read_table('../data/train/browse_history_train.txt', sep=',', names=browse_cols)
    browse_test = pd.read_table('../data/test/browse_history_test.txt', sep=',', names=browse_cols)
    browse = pd.concat([browse_train, browse_test])
    print 'browse_history shape: ', browse.shape

    browse_id_time = (browse.loc[:, ['id', 'time']]).groupby(['id']).mean()
    browse_n = (browse.loc[:, ['id', 'action', 'ac_index']]).groupby(['id', 'ac_index']).sum()
    browse_n = browse_n.unstack()
    browse_n.columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']
    browse_n.fillna(0, inplace=True)

    c_cols = []
    ACTION_NUM = 216
    for i in range(ACTION_NUM):
        f = 'action_' + str(i)
        c_cols.append(f)

    browse = browse.join(loan_time, on='id')
    browse['over_loan_time_flag'] = 0
    browse['gap_time'] = browse['time'] - browse['loan_time']
    browse.loc[(browse['gap_time']>=0), 'over_loan_time_flag'] = 3
    #browse.loc[(browse['gap_time']>=-5e7) & (browse['gap_time']<0), 'over_loan_time_flag'] = 2
    #browse.loc[(browse['gap_time']>=-1e8) & (browse['gap_time']<-5e7), 'over_loan_time_flag'] = 1
    #browse.loc[(browse['gap_time']< -1e8), 'over_loan_time_flag'] = 0
    browse.loc[(browse['gap_time']<0), 'over_loan_time_flag'] = 2
    browse.drop(['gap_time'], axis=1, inplace=True)

    browse['help_c'] = 1
    browse_n_cat_over_loan = browse.loc[:, ['id', 'action', 'help_c', 'over_loan_time_flag']].groupby(['id', 'action', 'over_loan_time_flag']).sum()
    browse_n_cat_over_loan = browse_n_cat_over_loan.unstack(level=1).unstack()
    new_cols = []
    for f in c_cols:
        for t in ['_bef_2', '_aft_3']:
            new_cols.append(f+t)
    browse_n_cat_over_loan.columns = new_cols
    browse_n_cat_over_loan.fillna(0, inplace=True)
    browse.drop(['help_c'], axis=1, inplace=True)
   
    browse['help_c'] = 1
    browse_n_cat = browse.loc[:, ['id', 'action', 'help_c']].groupby(['id', 'action']).sum()
    browse_n_cat = browse_n_cat.unstack()
    c_cols = []
    ACTION_NUM = 216
    for i in range(ACTION_NUM):
        f = 'action_' + str(i)
        c_cols.append(f)
    browse_n_cat.columns = c_cols
    browse_n_cat.fillna(0, inplace=True)
    browse.drop(['help_c'], axis=1, inplace=True)

    browse['help_c'] = 1
    browse_n_index = browse.loc[:, ['id', 'ac_index', 'help_c']].groupby(['id', 'ac_index']).sum()
    browse_n_index = browse_n_index.unstack()
    c_cols = []
    for i in range(11):
        f = 'action_ind_' + str(i)
        c_cols.append(f)
    browse_n_index.columns = c_cols
    browse_n_index.fillna(0, inplace=True)

    browse_n = browse_n.join(browse_n_index)
    browse_n = browse_n.join(browse_n_cat)
    browse_n = browse_n.join(browse_n_cat_over_loan)
    print 'browse_history done.'

    print 'getting bill_details...'
    bill_details_cols = ['id', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay',
             'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
             'account', 'adjust_account', 'circulated_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']
    bill_train = pd.read_table('../data/train/bill_detail_train.txt', sep=',', names=bill_details_cols)
    bill_test = pd.read_table('../data/test/bill_detail_test.txt', sep=',', names=bill_details_cols)
    bill_details = pd.concat([bill_train, bill_test])
    print 'bill_details shape: ', bill_details.shape

    #mean_val = bill_details.loc[bill_details['tm_encode_3'] != 0, 'tm_encode_3'].mean()
    #min_tm_bill = (bill_details.loc[bill_details['tm_encode_3'] != 0, ['id', 'tm_encode_3']]).groupby(['id']).mean()
    #min_tm_bill['tm_fill'] = min_tm_bill['tm_encode_3']
    #min_tm_bill['tm_fill'] -= 4e7
    #min_tm_bill.drop('tm_encode_3', axis=1, inplace=True)
    #bill_details = bill_details.join(min_tm_bill, on='id')
    #bill_details.loc[bill_details['tm_encode_3'] == 0, 'tm_encode_3'] = bill_details.loc[bill_details['tm_encode_3'] == 0, 'tm_fill']
    #bill_details.drop('tm_fill', axis=1, inplace=True)
    #bill_details['tm_encode_3'] = bill_details['tm_encode_3'].fillna(mean_val)
    b_id = bill_details.loc[:, ['id', 'bank_id']].groupby(['id'])
    b_id = b_id['bank_id'].unique()
    b_id = b_id.map(lambda x: len(x))
    b_id = b_id.to_frame(name='num_bankid')
############ mean
    bill_details.loc[bill_details['tm_encode_3'] == 0, 'tm_encode_3'] = bill_details.loc[bill_details['tm_encode_3'] != 0, 'tm_encode_3'].mean()
    #bill_details.loc[bill_details['tm_encode_3'] == 0, 'tm_encode_3'] = bill_details['tm_encode_3'].mean()
    bill_details_n = bill_details.groupby(['id']).mean()
    bill_details_n.drop(['bank_id'], axis=1, inplace=True)
############ sum
    bill_details_n_sum = bill_details.groupby(['id']).sum()
    bill_details_n_sum.drop(['tm_encode_3', 'bank_id'], axis=1, inplace=True)
    bill_details_n_sum.columns = [f + '_sum' for f in bill_details_n_sum.columns]
############ per
    bill_details_ = bill_details.iloc[1:len(bill_details), :].copy().reset_index()
    bill_details__ = bill_details.iloc[:len(bill_details)-1, :].reset_index()
    bill_details__['id'] = 0
    bill_details_ = bill_details_.subtract(bill_details__)
    bill_details_.drop(['index', 'tm_encode_3', 'bank_id'], axis=1, inplace=True)
    bill_details_.columns = ['id'] + [f+'_per' for f in bill_details_.columns[1:]]
    bill_details_per = bill_details_.groupby(['id']).mean()


    neg_pos_cols = ['account', 'account_balance', 'adjust_account', 'avaliable_balance', 'cash_limit', 'circulated_interest', 'credit_limit',\
               'minimun_repay', 'prior_account', 'prior_repay']
    min_neg_pos = bill_details.loc[:, ['id'] + neg_pos_cols].groupby(['id']).min()
    #min_neg_pos = np.sign(min_neg_pos)
    min_neg_pos.columns = ['min_account', 'min_account_balance', 'min_adjust_account', 'min_avaliable_balance', 'min_cash_limit', \
                       'min_circulated_interest', 'min_credit_limit', 'min_minimun_repay', 'min_prior_account', 'min_prior_repay']

    max_neg_pos = bill_details.loc[:, ['id']+neg_pos_cols+['repay_state']].groupby(['id']).max()
    #max_neg_pos = np.sign(max_neg_pos)
    max_neg_pos.columns = ['max_account', 'max_account_balance', 'max_adjust_account', 'max_avaliable_balance', 'max_cash_limit', \
                       'max_circulated_interest', 'max_credit_limit', 'max_minimun_repay', 'max_prior_account', 'max_prior_repay',
                       'max_repay_state']
    #bill_details.tm_encode_3 = bill_details.tm_encode_3.fillna(bill_details.tm_encode_3.mean())

    bill_details = bill_details.join(loan_time, on='id')
    #bill_details['over_loan_time_flag'] = bill_details['tm_encode_3'] - bill_details['loan_time']
    #bill_details.loc[bill_details['over_loan_time_flag']>0, 'over_loan_time_flag'] = 1
    #bill_details.loc[bill_details['over_loan_time_flag']<0, 'over_loan_time_flag'] = 0

    bill_details['over_loan_time_flag'] = 0
    bill_details['gap_time'] = bill_details['tm_encode_3'] - bill_details['loan_time']

    bill_details.loc[(bill_details['gap_time']>6e6), 'over_loan_time_flag'] = 3
    bill_details.loc[(bill_details['gap_time']>=3e6) & (bill_details['gap_time']<6e6), 'over_loan_time_flag'] = 2
    bill_details.loc[(bill_details['gap_time']>=0) & (bill_details['gap_time']<3e6), 'over_loan_time_flag'] = 1
    bill_details.loc[(bill_details['gap_time']< 0), 'over_loan_time_flag'] = 0
    #bill_details.loc[(bill_details['gap_time']>1e7), 'over_loan_time_flag'] = 3
    #bill_details.loc[(bill_details['gap_time']>=5e6) & (bill_details['gap_time']<1e7), 'over_loan_time_flag'] = 2
    #bill_details.loc[(bill_details['gap_time']>=0) & (bill_details['gap_time']<5e6), 'over_loan_time_flag'] = 1
    #bill_details.loc[(bill_details['gap_time']< 0), 'over_loan_time_flag'] = 0

    bill_details.drop(['gap_time'], axis=1, inplace=True)

    cols = [f for f in bill_details.columns.values if f not in ['id', 'over_loan_time_flag']]
    bill_details_over_loan = bill_details.groupby(['id', 'over_loan_time_flag']).mean()
    bill_details_over_loan = bill_details_over_loan.unstack()
    new_cols = [] 
    for f in cols:
        for t in ['_before_0', '_before_1', '_after_2', '_after_3']:
            new_cols.append(f+t)
    bill_details_over_loan.columns = new_cols
    bill_details_over_loan.drop(['bank_id_before_0', 'bank_id_before_1', 'bank_id_after_2', \
                        'bank_id_after_3', 'loan_time_before_0',  'loan_time_before_1', \
                        'loan_time_after_2', 'loan_time_after_3'], axis=1, inplace=True)
    bill_details_over_loan.fillna(0, inplace=True)

    bill_details_over_loan_sum = bill_details.groupby(['id', 'over_loan_time_flag']).sum()
    bill_details_over_loan_sum = bill_details_over_loan_sum.unstack()
    new_cols = [] 
    for f in cols:
        for t in ['_sum_before_0', '_sum_before_1', '_sum_after_2', '_sum_after_3']:
            new_cols.append(f+t)
    bill_details_over_loan_sum.columns = new_cols
    bill_details_over_loan_sum.drop(['bank_id_sum_before_0', 'bank_id_sum_before_1', 'bank_id_sum_after_2', 'bank_id_sum_after_3',
                            'tm_encode_3_sum_before_0', 'tm_encode_3_sum_before_1',  'tm_encode_3_sum_after_2', 'tm_encode_3_sum_after_3',
                            'loan_time_sum_before_0', 'loan_time_sum_before_1', 'loan_time_sum_after_2', 'loan_time_sum_after_3'], 
                            axis=1, inplace=True)
    bill_details_over_loan_sum.fillna(0, inplace=True)


    bill_details_ = bill_details.iloc[1:len(bill_details), :].copy().reset_index()
    bill_details__ = bill_details.iloc[:len(bill_details)-1, :].reset_index()
    bill_details__['id'] = 0
    bill_details__['over_loan_time_flag'] = 0
    bill_details_ = bill_details_.subtract(bill_details__)
    bill_details_.drop(['index', 'tm_encode_3', 'bank_id', 'loan_time'], axis=1, inplace=True)
    bill_details_over_loan_per = bill_details_.groupby(['id', 'over_loan_time_flag']).mean()
    bill_details_over_loan_per = bill_details_over_loan_per.unstack()
    new_cols = []
    cols = [f for f in bill_details_.columns.values if f not in ['id', 'over_loan_time_flag']]
    for f in cols:
        for t in ['_per_before_0', '_per_before_1', '_per_after_2', '_per_after_3']:
            new_cols.append(f+t)
    bill_details_over_loan_per.columns = new_cols
    bill_details_over_loan_per.fillna(0, inplace=True)

    min_neg_pos_over_loan = bill_details.loc[:, ['id', 'over_loan_time_flag']+neg_pos_cols].groupby(['id', 'over_loan_time_flag']).min()
    min_neg_pos_over_loan = min_neg_pos_over_loan.unstack()
    new_cols = [] 
    for f in neg_pos_cols:
        for t in ['_min_before_0', '_min_before_1', '_min_after_2', '_min_after_3']:
            new_cols.append(f+t)
    min_neg_pos_over_loan.columns = new_cols
    min_neg_pos_over_loan.fillna(0, inplace=True)

    # max_neg_pos_over_loan = bill_details.loc[:, ['id', 'over_loan_time_flag']+neg_pos_cols+['repay_state']].groupby(['id', 'over_loan_time_flag']).max()
    # max_neg_pos_over_loan = max_neg_pos_over_loan.unstack()
    # neg_pos_cols_ = neg_pos_cols + ['repay_state']
    # new_cols = []
    # print neg_pos_cols_
    # for f in neg_pos_cols_:
    #     for t in ['_max_before_0', '_max_before_1', '_max_after_2' , '_max_after_3']:
    #         new_cols.append(f+t)

    # max_neg_pos_over_loan.columns = new_cols
    # max_neg_pos_over_loan.fillna(0, inplace=True)

    bill_details_n = bill_details_n.join(bill_details_over_loan, how='outer')
    #bill_details_n = bill_details_n.join(bill_details_over_loan_sum, how='outer')
    #bill_details_n = bill_details_n.join(bill_details_n_sum)
    bill_details_n = bill_details_n.join(min_neg_pos)
    #bill_details_n = bill_details_n.join(max_neg_pos)
    bill_details_n = bill_details_n.join(min_neg_pos_over_loan)
    #bill_details_n = bill_details_n.join(max_neg_pos_over_loan)
    bill_details_n = bill_details_n.join(b_id)
    #bill_details_n = bill_details_n.join(bill_details_per)
    #bill_details_n = bill_details_n.join(bill_details_over_loan_per)
    #bill_details_n = bill_details_n.astype('float')
    print 'bill_details done.'



    data = user_info.join(bank_detail_n, how='outer')
    data = data.join(browse_n, how='outer')
    data = data.join(bill_details_n, how='outer')
    data = data.join(loan_time, how='outer')

    train_len = len(user_info_train)
    train_data = data.iloc[:train_len, :]
    test_data = data.iloc[train_len:, :]

    assert(len(train_data) == len(user_info_train))
    assert(len(test_data) == len(user_info_test))

    train_data.to_csv('../data/train_data_v2.csv')
    test_data.to_csv('../data/test_data_v2.csv')

if __name__ == '__main__':
    main()