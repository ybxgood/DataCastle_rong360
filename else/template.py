# -*- coding: utf-8 -*-
# ���������python��
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ��������
# ����ֱ��user_info, bank_detail, browse_data, bill_detail, loan_data����Ԥ����

# user_info
# ��ȡ���ݼ�
user_info_train = pd.read_csv('D:/data/DataCastle/train/user_info_train.txt',
                                  header = None)
user_info_test = pd.read_csv('D:/data/DataCastle/test/user_info_test.txt',
                                 header = None)
# �����ֶΣ��У�����
col_names = ['userid', 'sex', 'occupation', 'education', 'marriage', 'household']
user_info_train.columns = col_names
user_info_test.columns = col_names
# �ϲ�train��test
user_info = pd.concat([user_info_train, user_info_test])
# ��userid���û�id������Ϊ���ݼ���index����ɾ��ԭuserid������
user_info.index = user_info['userid']
user_info.drop('userid',
                axis = 1,
                inplace = True)
# �鿴���������ݼ������ǰ5��
print user_info.head(5)

# ����Ĵ���ʽ���ƣ��ҽ�ע�Ͳ�ͬ�ĵط�
# bank_detail
bank_detail_train = pd.read_csv('D:/data/DataCastle/train/bank_detail_train.txt',
                                    header = None)
bank_detail_test = pd.read_csv('D:/data/DataCastle/test/bank_detail_test.txt',
                                    header = None)
col_names = ['userid', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
bank_detail_train.columns = col_names
bank_detail_test.columns = col_names
bank_detail = pd.concat([bank_detail_train, bank_detail_test])
# �ڸ����ݼ��У�һ���û���Ӧ������¼���������ǲ��ö�ÿ���û�ÿ�ֽ�������ȡ��ֵ���оۺ�
bank_detail_n = (bank_detail.loc[:, ['userid', 'trade_type', 'trade_amount', 'tm_encode']]).groupby(['userid', 'trade_type']).mean()
# �������ݼ����������ֶΣ��У�����
bank_detail_n = bank_detail_n.unstack()
bank_detail_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']
print bank_detail_n.head(5)

# browse_history
browse_history_train = pd.read_csv('D:/data/DataCastle/train/browse_history_train.txt',
                                       header = None)
browse_history_test = pd.read_csv('D:/data/DataCastle/test/browse_history_test.txt',
                                       header = None)
col_names = ['userid', 'tm_encode_2', 'browse_data', 'browse_tag']
browse_history_train.columns = col_names
browse_history_test.columns = col_names
browse_history = pd.concat([browse_history_train, browse_history_test])
# ������ü���ÿ���û��������Ϊ�������оۺ�
browse_history_count = browse_history.loc[:, ['userid', 'browse_data']].groupby(['userid']).sum()
print browse_history_count.head(5)

# bill_detail
bill_detail_train = pd.read_csv('D:/data/DataCastle/train/bill_detail_train.txt',
                                       header = None)
bill_detail_test = pd.read_csv('D:/data/DataCastle/test/bill_detail_test.txt',
                                       header = None)
col_names = ['userid', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay',
             'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
             'account', 'adjust_account', 'circulated_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']
bill_detail_train.columns = col_names
bill_detail_test.columns = col_names
bill_detail = pd.concat([bill_detail_train, bill_detail_test])
bill_detail_mean = bill_detail.groupby(['userid']).mean()
bill_detail_mean.drop('bank_id',
                      axis = 1,
                      inplace = True)
print bill_detail_mean.head(5)

# loan_time
loan_time_train = pd.read_csv('D:/data/DataCastle/train/loan_time_train.txt',
                              header = None)
loan_time_test = pd.read_csv('D:/data/DataCastle/test/loan_time_test.txt',
                              header = None)
loan_time = pd.concat([loan_time_train, loan_time_test])
loan_time.columns = ['userid', 'loan_time']
loan_time.index = loan_time['userid']
loan_time.drop('userid',
               axis = 1,
               inplace = True)
print loan_time.head(5)

# �ֱ������������ݼ��󣬸���userid����join����ʽѡ��outer'��û��bill����bank���ݵ�user�ڶ�Ӧ�ֶ��Ͻ�ΪNaֵ
loan_data = user_info.join(bank_detail_n, how = 'outer')
loan_data = loan_data.join(bill_detail_mean, how = 'outer')
loan_data = loan_data.join(browse_history_count, how = 'outer')
loan_data = loan_data.join(loan_time, how = 'outer')

# �ȱʧֵ
loan_data = loan_data.fillna(0.0)
print loan_data.head(5)

# ������������������ٸ�С���ӣ�
loan_data['time'] = loan_data['loan_time'] - loan_data['tm_encode_3']

# ���Ա�ְҵ�����ӱ������������Ʊ���
category_col = ['sex', 'occupation', 'education', 'marriage', 'household']
def set_dummies(data, colname):
    for col in colname:
        data[col] = data[col].astype('category')
        dummy = pd.get_dummies(data[col])
        dummy = dummy.add_prefix('{}#'.format(col))
        data.drop(col,
                  axis = 1,
                  inplace = True)
        data = data.join(dummy)
    return data
loan_data = set_dummies(loan_data, category_col)

# overdue_train����������ģ����Ҫ��ϵ�Ŀ��
target = pd.read_csv('D:/data/DataCastle/train/overdue_train.txt',
                         header = None)
target.columns = ['userid', 'label']
target.index = target['userid']
target.drop('userid',
            axis = 1,
            inplace = True)
# ����ģ��
# �ֿ�ѵ���������Լ�
train = loan_data.iloc[0: 55596, :]
test = loan_data.iloc[55596:, :]
# �������ϣ����ý�����֤����֤��ռѵ����20%���̶�������ӣ�random_state)
train_X, test_X, train_y, test_y = train_test_split(train,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 0)
train_y = train_y['label']
test_y = test_y['label']
# ������Logistic�ع�
lr_model = LogisticRegression(C = 1.0,
                              penalty = 'l2')
lr_model.fit(train_X, train_y)
# ����������֤����Ԥ����������׼ȷ�ʡ��ٻ��ʡ�F1ֵ
pred_test = lr_model.predict(test_X)
print classification_report(test_y, pred_test)
# ������Լ��û����ڻ�����ʣ�predict_proba������������ʣ�ȡ��1���ĸ���
pred = lr_model.predict_proba(test)
result = pd.DataFrame(pred)
result.index = test.index
result.columns = ['0', 'probability']
result.drop('0',
            axis = 1,
            inplace = True)
print result.head(5)
# ������
result.to_csv('result.csv')

�ǳ������յ�datacastle������μ���η�������Ҫ˵���£�����ֻ�ṩ�˲��ִ��루���Ҽ��£��������μӱ�����С�׶Խ�ģ�����и����µ��˽⣬����ʵ����Ҳ����ʲô����
�������ϴ�������ο����д���ĵط�������ָ������ϣ���ܽ���������Ļ��ᣬ��ʶ�������ݷ����������ھ�İ����ߣ�������ѧϰ������ĸɻ����ʹ�ҹ�ͬ������лл��

by ���Ź��¸�Ƿ