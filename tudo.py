from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import pandas as pd

clf = SVC(probability=True)


account_df = pd.read_csv(
    'data/account.csv', sep=';').rename(columns={'date': 'account_date'})
card_test_df = pd.read_csv(
    'data/card_test.csv', sep=';').rename(columns={'type': 'card_type'})
card_train_df = pd.read_csv('data/card_train.csv',
                            sep=';').rename(columns={'type': 'card_type'})
client_df = pd.read_csv('data/client.csv', sep=';')
disp_df = pd.read_csv(
    'data/disp.csv', sep=';').rename(columns={'type': 'disp_type'})
district_df = pd.read_csv('data/district.csv', sep=';')
loan_test_df = pd.read_csv(
    'data/loan_test.csv', sep=';').rename(columns={'date': 'loan_date', 'amount': 'loan_amount'})
loan_train_df = pd.read_csv('data/loan_train.csv',
                            sep=';').rename(columns={'date': 'loan_date', 'amount': 'loan_amount'})
trans_test_df = pd.read_csv('data/trans_test.csv', sep=';').rename(
    columns={'type': 'trans_type', 'date': 'trans_date', 'amount': 'trans_amount'})
trans_train_df = pd.read_csv('data/trans_train.csv', sep=';').rename(
    columns={'type': 'trans_type', 'date': 'trans_date', 'amount': 'trans_amount'})

pre_df = disp_df.merge(account_df).merge(client_df)
train_df = pre_df.merge(card_train_df).merge(
    trans_train_df).merge(loan_train_df)
test_df = pre_df.merge(card_test_df).merge(trans_test_df).merge(loan_test_df)

print(train_df)

