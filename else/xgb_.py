import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

def f1_error(preds, dtrain):
    labels = dtrain.get_label()
    pred = [int(i >= 0.5) for i in preds]
    tp = np.sum([int(i==1 and j==1) for i, j in zip(pred, labels)])
    if np.sum(pred) == 0:
        precision = 0.
    else:
        precision = float(tp) / np.sum(pred)
    if np.sum(labels) == 0:
        recall = 0.
    else:
        recall = float(tp) / np.sum(labels)
    return 'f1-score', -2 * (precision * recall / (precision + recall))

def f1_error_(labels, preds):
    #pred = 1.0 / (1.0 + np.exp(-preds))
    pred = [int(i >= 0.5) for i in preds]
    tp = np.sum([int(i==1 and j==1) for i, j in zip(pred, labels)])
    precision = float(tp) / np.sum(pred)
    recall = float(tp) / np.sum(labels)
    print 'precision: ', precision
    print 'recall: ', recall
    return -2 * (precision * recall / (precision + recall))

def ks_obj(preds, dtrain):
    labels = dtrain.get_label()
    d = {}
    #print preds
    #for p, y in zip(preds, labels):
    #    if p not in d.keys():
    #        d[p] = list([y])
    #    else:
    #        d[p].append(y)
    num = len(labels)
    for i in range(num):
        if preds[i] not in d.keys():
            d[preds[i]] = list([labels[i]])
        else:
            d[preds[i]].append(labels[i])
    pos_counts = 0
    neg_counts = 0
    pos_sum = np.sum(labels)
    neg_sum = num - pos_sum
    max_dif = 0.
    for p in sorted(d.keys()):
        t_p_c = np.sum(d[p])
        pos_counts += t_p_c
        neg_counts += len(d[p]) - t_p_c
        dif = abs(float(pos_counts)/pos_sum - float(neg_counts)/neg_sum)
        if dif > max_dif:
            max_dif = dif
    return 'ks-score', -max_dif

def ks_(preds, labels):
    d = {}
    for p, y in zip(preds, labels):
        if p not in d.keys():
            d[p] = list([y])
        else:
            d[p].append(y)
    
    pos_counts = 0
    neg_counts = 0
    pos_sum = np.sum(labels)
    neg_sum = len(labels) - pos_sum
    max_dif = 0.
    for p in sorted(d.keys()):
        t_p_c = np.sum(d[p])
        pos_counts += t_p_c
        neg_counts += len(d[p]) - t_p_c
        dif = abs(float(pos_counts)/pos_sum - float(neg_counts)/neg_sum)
        if dif > max_dif:
            max_dif = dif
    return -max_dif

if __name__ == '__main__':
    train_data = pd.read_csv('../data/train_data_n4_4billtime.csv')
    test_data = pd.read_csv('../data/test_data_n4_4billtime.csv')
    len_train = len(train_data)
    data = pd.concat([train_data, test_data])

    mean_cols = [u'income', u'outcome', u'income_tm', u'outcome_tm', u'salary_tag',
                u'out_count', u'in_count', u'browse_time', u'tm_encode_3',
                u'prior_account', u'prior_repay', u'credit_limit',
                u'account_balance', u'minimun_repay', u'consume_count', u'account',
                u'adjust_account', u'circulated_interest', u'avaliable_balance',
                u'cash_limit', u'repay_state', u'tm_encode_3_before',
                u'tm_encode_3_after', u'prior_account_before', u'prior_account_after',
                u'prior_repay_before', u'prior_repay_after', u'credit_before',
                u'credit_after', u'account_balance_before', u'account_balance_after',
                u'minimun_repay_before', u'minimun_repay_after',
                u'consume_count_before', u'consume_count_after', u'account_before',
                u'account_after', u'adjust_account_before', u'adjust_account_after',
                u'circulated_interest_before', u'circulated_interest_after',
                u'avaliable_balance_before', u'avaliable_balance_after',
                u'cash_limit_before', u'cash_limit_after', u'repay_state_before',
                u'repay_state_after']
    zero_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                'b7', 'b8', 'b9', 'b10', 'b11', 'average_b', 
                'b1_bef', 'b1_aft', 'b2_bef', 'b2_aft', 'b3_bef', 'b3_aft', 'b4_bef', 'b4_aft',
                'b5_bef', 'b5_aft', 'b6_bef', 'b6_aft', 'b7_bef', 'b7_aft', 'b8_bef', 'b8_aft',
                'b9_bef', 'b9_aft', 'b10_bef', 'b10_aft', 'b11_bef', 'b11_aft',
                'b1_ave', 'b2_ave', 'b3_ave', 'b4_ave', 'b5_ave', 'b6_ave', 'b7_ave', 
                'b8_ave', 'b9_ave', 'b10_ave', 'b11_ave',
                'b1_ave_bef', 'b1_ave_aft', 'b2_ave_bef', 'b2_ave_aft', 'b3_ave_bef', 'b3_ave_aft', 'b4_ave_bef', 'b4_ave_aft',
                'b5_ave_bef', 'b5_ave_aft', 'b6_ave_bef', 'b6_ave_aft', 'b7_ave_bef', 'b7_ave_aft', 'b8_ave_bef', 'b8_ave_aft',
                'b9_ave_bef', 'b9_ave_aft', 'b10_ave_bef', 'b10_ave_aft', 'b11_ave_bef', 'b11_ave_aft']#, 
                #'sal_type_cat', 'sal_type_cat_bef', 'sal_type_cat_aft']
    #data[mean_cols] = data[mean_cols].fillna(data[mean_cols].mean())
    na_cat_cols = ['min_account', 'min_account_balance', 'min_adjust_account', 'min_avaliable_balance', 'min_cash_limit',
                    'min_circulated_interest', 'min_credit_limit', 'min_minimun_repay', 'min_prior_account', 'min_prior_repay',
                    'max_account', 'max_account_balance', 'max_adjust_account', 'max_avaliable_balance', 'max_cash_limit',
                    'max_circulated_interest', 'max_credit_limit', 'max_minimun_repay', 'max_prior_account', 'max_prior_repay',
                    'max_repay_state',
                    'min_account_bef', 'min_account_aft', 'min_account_balance_bef', 'min_account_balance_aft', 
                    'min_adjust_account_bef', 'min_adjust_account_aft', 'min_avaliable_balance_bef', 'min_avaliable_balance_aft', 
                    'min_cash_limit_bef', 'min_cash_limit_aft', 'min_circulated_interest_bef', 'min_circulated_interest_aft',
                    'min_credit_limit_bef', 'min_credit_limit_aft', 'min_minimun_repay_bef', 'min_minimun_repay_aft', 
                    'min_prior_account_bef', 'min_prior_account_aft', 'min_prior_repay_bef', 'min_prior_repay_aft',
                    'max_account_bef', 'max_account_aft', 'max_account_balance_bef', 'max_account_balance_aft', 
                    'max_adjust_account_bef', 'max_adjust_account_aft', 'max_avaliable_balance_bef', 'max_avaliable_balance_aft', 
                    'max_cash_limit_bef', 'max_cash_limit_aft', 'max_circulated_interest_bef', 'max_circulated_interest_aft',
                    'max_credit_limit_bef', 'max_credit_limit_aft', 'max_minimun_repay_bef', 'max_minimun_repay_aft', 
                    'max_prior_account_bef', 'max_prior_account_aft', 'max_prior_repay_bef', 'max_prior_repay_aft',
                    'repay_state_bef', 'repay_state_aft']
    

    #for f in data.columns:
    #    data[f+'_nan'] = data[f].isnull()

    data[zero_cols] = data[zero_cols].fillna(0.)
    #data.loc[:, na_cat_cols] = data.loc[:, na_cat_cols].fillna(np.round(data.loc[:, na_cat_cols].mean()))
    data.fillna(data.mean(), inplace=True)

    #data['znull'] = data.isnull().sum(axis=1)

    #data['loan_bill_time'] = data['loan_time'] - data['tm_encode_3']
    #data['out_in_time'] = data['outcome_tm'] - data['income_tm']
    #data['loan_browse_time'] = data['loan_time'] - data['browse_time']
    #data['loan_in_time'] = data['loan_time'] - data['income_tm']
    #data['loan_out_time'] = data['loan_time'] - data['outcome_tm']

    #data.drop(['income_tm', 'outcome_tm', 'tm_encode_3', 'loan_time'], axis=1, inplace=True)
    f_cat = ['sex', 'job', 'edu', 'marriage', 'loc_type', 'sal_type_cat']#, 'sal_type_cat_bef', 'sal_type_cat_aft']
    #f_cat += na_cat_cols
    #f_cont = ['income', 'outcome', 'b1', 'b2', 'b3', 'b4',  
    #        'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'prior_account',   
    #        'prior_repay', 'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',   
    #        'account', 'adjust_account', 'circulated_interest', 'avaliable_balance', 'cash_limit',
    #        'repay_state', 'loan_bill_time', 'out_in_time', 'loan_browse_time']
    #f_cont = ['loan_bill_time', 'out_in_time', 'loan_browse_time', 'loan_time', 'tm_encode_3',
    #        'outcome_tm', 'income_tm', 'browse_time', 'loan_in_time', 'loan_out_time']
    for f in f_cat:
        data[f] = data[f].astype('category')
    data = pd.get_dummies(data)
    print data.columns

    def fcat_or_not(f):
        if 'sex' in f or 'job' in f or 'edu' in f\
            or 'marriage' in f or 'loc_type' in f or 'cat' in f:
            return True
        else:
            return False 


    f_cont = [f for f in data.columns if fcat_or_not(f) is False]

    #skew_feats = data[f_cont].apply(lambda x: skew(x.dropna()))
    #skew_feats = skew_feats[skew_feats > 0.25]
    #print skew_feats
    #skew_feats = skew_feats.index
    #for f in skew_feats:
    #    if data[f].min() < 0:
    #        continue
    #    data[f] += 1.
    #    data[f], lam = boxcox(data[f])

    #print data.isnull().any()

    #shift = 5e9
    #data[f_cont] -= shift
    #ss = StandardScaler()
    #data[f_cont] = ss.fit_transform(data[f_cont].values)

    dump_cols = ['loan_time', 'tm_encode_3',
            'outcome_tm', 'income_tm', 'browse_time']

    #dump_cols = [f for f in data.columns if 'aft' in f or 'bef' in f]
    data.drop(dump_cols, axis=1, inplace=True)

    #for f in data.columns:
    #    if fcat_or_not(f) is False:
    #       data[f+'_rank'] = data[f].rank(method='min')


    train_X = data.iloc[:len_train, :].values
    test_X = data.iloc[len_train:, :].values
    print 'train data shape: ', train_X.shape
    print 'test data shape: ', test_X.shape

    del (data, train_data, test_data)
    train_y = pd.read_table('../data/train/overdue_train.txt', sep=',', names=['id', 'y'])
    train_y = train_y['y'].values

    test_pred = np.zeros(test_X.shape[0])
    d_test = xgb.DMatrix(test_X)

    test_id = pd.read_table('../data/test/usersID_test.txt', sep=',', names=['id'])
    test_id = test_id['id'].values

    n_folds = 5
    folds = KFold(len(train_y), n_folds=n_folds, shuffle=True)
    oob_pred = np.zeros((train_X.shape[0],))
    for i, (train_index, test_index) in enumerate(folds):
        xgb_params = {
        'seed': 0,
        'silent': 1,
        'objective': 'binary:logistic',
        #'objective': 'rank:pairwise',
        'subsample': 0.8,
        #'colsample_bylevel': 0.8,
        'max_depth': 6,
        'min_child_weight': 10,
        'booster': 'gbtree',
        'max_delta_step': 1,
        'eta': 0.06,
        'eval_metric': 'auc',
        #'scale_pos_weight': float(len(train_y[train_y==0])) / len(train_y[train_y==1])
        }

        xtr = train_X[train_index]
        ytr = train_y[train_index]
        xte = train_X[test_index]
        yte = train_y[test_index]
        d_train = xgb.DMatrix(xtr, label=ytr)
        d_valid = xgb.DMatrix(xte, label=yte)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        clf = xgb.train(xgb_params,
                    d_train,
                    1000, 
                    watchlist,
                    early_stopping_rounds=50,
                    #feval=ks_obj
                    )
        pred = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
        oob_pred[test_index] = pred
        print 'ks score:', ks_(pred, yte)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        print 'fold ', i+1, 'accuracy:', accuracy_score(yte, pred)
        print 'fold ', i+1, 'f1: ', f1_error_(yte, pred)
        test_pred += clf.predict(d_test, ntree_limit=clf.best_ntree_limit)
        del d_train, d_valid

    print 'whole ks score: ', ks_(oob_pred, train_y)
    test_pred /= n_folds
    df = pd.DataFrame()
    df['userid'] = test_id
    df['probability'] = test_pred
    df.to_csv('../outputs/xgb_5folds_n4_.csv', index=False)