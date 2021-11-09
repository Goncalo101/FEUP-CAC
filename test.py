from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import pandas as pd


def is_female(number):
    return str(number)[2] == '5' or str(number)[2] == '6'


def encode_cols(df):
    le = LabelEncoder()
    df['frequency'] = le.fit_transform(df['frequency'])
    df['type'] = le.fit_transform(df['type'])
    df['gender'] = le.fit_transform(df['gender'])
    return df


clf = SVC(probability=True)
account_df = pd.read_csv('data/account.csv', sep=';')
client_df = pd.read_csv('data/client.csv', sep=';')
disp_df = pd.read_csv('data/disp.csv', sep=';')
disp_df = disp_df[disp_df['type'] != 'DISPONENT']
loan_train_df = pd.read_csv('data/loan_train.csv', sep=';')
loan_test_df = pd.read_csv('data/loan_test.csv', sep=';')


client_df['gender'] = client_df.apply(
    lambda row: 0 if is_female(row['birth_number']) else 1, axis=1)
client_df['birth_number'] = client_df.apply(
    lambda row: row['birth_number'] - 5000 if is_female(row['birth_number']) else row['birth_number'], axis=1)
inter_df = pd.merge(account_df, disp_df, on='account_id')
inter_df = pd.merge(inter_df, client_df, on='client_id')


train_df = pd.merge(inter_df, loan_train_df, on='account_id')
test_df = pd.merge(inter_df, loan_test_df, on='account_id')
test_df = test_df.rename(
    columns={'district_id_x': 'account_district_id',
             'district_id_y': 'client_district_id'}
)

result_df = train_df.rename(
    columns={'district_id_x': 'account_district_id',
             'district_id_y': 'client_district_id'}
)

result_df = encode_cols(result_df)
test_df = encode_cols(test_df)
x = result_df.drop(['status'], axis=1)
y = result_df.status
clf.set_params(kernel='linear', class_weight='balanced').fit(x, y)

# result_df.to_csv('data/result.csv', index=False)

df = pd.DataFrame(data={'Id': test_df.loan_id,
                  'Predicted': clf.predict_proba(test_df.drop(['status'], axis=1))[:, -1]})
df.to_csv('data/submission.csv', index=False)
