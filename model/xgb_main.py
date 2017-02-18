
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold
from scipy.stats import ks_2samp

def ks_score(preds, label):
    return ks_2samp(preds[label == 0], preds[label == 1])[0]

def ks_obj(preds, dtrain):
    labels = dtrain.get_label()
    return 'ks-score', - ks_2samp(preds[labels==0], preds[labels==1])[0]

if __name__ == "__main__":

    print 'read data...'
    data = pd.read_csv('../input/train_test.csv')
    data.drop(['id'], axis = 1, inplace = True)

    train_X = data.iloc[:55596 , :].values
    test_X  = data.iloc[55596: , :].values

    train_y = pd.read_csv('../input/raw/train_overdue.csv', names=['id', 'y'])
    train_y = train_y['y'].values
    test_id = pd.read_csv('../input/raw/test_usersID.csv', names=['id'])
    test_id = test_id['id'].values

    seed_val = 4567
    n_folds = 4
    folds = KFold(len(train_y), n_folds=n_folds, shuffle=True, random_state = seed_val)

    test_pred = np.zeros(test_X.shape[0])
    valid_pred = np.zeros(train_X.shape[0])
    test_set = xgb.DMatrix(test_X)

    xgb_params = {
        'seed': seed_val,
        'silent': 1,
        'objective': 'binary:logistic',
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'max_depth': 6,
        'min_child_weight': 10,
        'eta': 0.06,
    }

    for i, (train_index, test_index) in enumerate(folds):

        xtr, xva = train_X[train_index], train_X[test_index]
        ytr, yva = train_y[train_index], train_y[test_index]
        train_set, valid_set = xgb.DMatrix(xtr, ytr), xgb.DMatrix(xva, yva)
        watchlist = [(train_set, 'train'), (valid_set, 'eval')]
        clf = xgb.train(xgb_params, train_set, 1000, watchlist,
                        early_stopping_rounds = 50,
                        feval = ks_obj )
        pred = clf.predict(valid_set, ntree_limit = clf.best_ntree_limit)
        valid_pred[test_index] = pred
        print  i + 1,'fold ks score:', ks_score(pred, yva)
        test_pred += clf.predict(test_set, ntree_limit=clf.best_ntree_limit)
        del train_set, valid_set

    print 'whole ks score: ', ks_score(valid_pred, train_y)
    test_pred /= n_folds
    df = pd.DataFrame()
    df['userid'] = test_id
    df['probability'] = test_pred
    df.to_csv('../submit/xgb_main.csv', index=False)
