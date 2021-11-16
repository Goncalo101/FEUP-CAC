from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import pandas as pd


def encode_cols(df):
    le = LabelEncoder()
    for col, col_type in df.dtypes.items():
        if col_type == 'object':
            df[col] = le.fit_transform(df[col])
    return df



# clf = SVC(probability=True)
clf = LogisticRegression()


account_df = pd.read_csv(
    'data/account.csv', sep=';', index_col='account_id').rename(columns={'date': 'account_date', 'district_id': 'account_district_id'})
card_test_df = pd.read_csv(
    'data/card_test.csv', sep=';', index_col='card_id').rename(columns={'type': 'card_type'})
card_train_df = pd.read_csv('data/card_train.csv',
                            sep=';', index_col='card_id').rename(columns={'type': 'card_type'})
client_df = pd.read_csv('data/client.csv', sep=';', index_col='client_id').rename(columns={'district_id': 'client_district_id'})
disp_df = pd.read_csv(
    'data/disp.csv', sep=';', index_col='disp_id').rename(columns={'type': 'disp_type'})
district_df = pd.read_csv('data/district.csv', sep=';', index_col='code ')
loan_test_df = pd.read_csv(
    'data/loan_test.csv', sep=';', index_col='loan_id').rename(columns={'date': 'loan_date', 'amount': 'loan_amount'})
loan_train_df = pd.read_csv('data/loan_train.csv',
                            sep=';', index_col='loan_id').rename(columns={'date': 'loan_date', 'amount': 'loan_amount'})
trans_test_df = pd.read_csv('data/trans_test.csv', sep=';', index_col='trans_id').rename(
    columns={'type': 'trans_type', 'date': 'trans_date', 'amount': 'trans_amount'})
trans_train_df = pd.read_csv('data/trans_train.csv', sep=';', index_col='trans_id').rename(
    columns={'type': 'trans_type', 'date': 'trans_date', 'amount': 'trans_amount'})

account_df = encode_cols(account_df)
card_test_df = encode_cols(card_test_df)
card_train_df = encode_cols(card_train_df)
client_df = encode_cols(client_df)
disp_df = encode_cols(disp_df)
district_df = encode_cols(district_df)
loan_test_df = encode_cols(loan_test_df)
loan_train_df = encode_cols(loan_train_df)
trans_test_df = encode_cols(trans_test_df)
trans_train_df = encode_cols(trans_train_df)

pre_df = disp_df.merge(account_df, on='account_id').set_index(disp_df.index)
pre_df = pre_df.merge(client_df, on='client_id').set_index(disp_df.index)
# print(pre_df)

train_df = pre_df.merge(loan_train_df, on='account_id')# .set_index(loan_train_df.index)
# .merge(card_train_df, on='disp_id')
# .merge(trans_train_df, on='account_id')
test_df = pre_df.merge(loan_test_df, on='account_id')# .set_index(loan_test_df.index)
# .merge(card_train_df, on='disp_id').merge(
#     trans_train_df, on='account_id')

# print(train_df)

clf.fit(train_df.drop(['status'], axis=1), train_df.status)
print(train_df.drop(['status'], axis=1))
# print(clf.predict_proba(test_df.drop(['status'], axis=1)))
print(test_df.columns)

df = pd.DataFrame(data={'Id': test_df.loan_id,
                  'Predicted': clf.predict_proba(test_df.drop(['status'], axis=1))[:, -1]})
df.to_csv('data/submission.csv', index=False)