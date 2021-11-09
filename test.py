from sklearn.svm import SVC

import pandas as pd


def is_female(number):
    return str(number)[2] == '5' or str(number)[2] == '6'

# loan_train = pd.read_csv('data/loan_train.csv', sep=';')
# clf = SVC(probability=True)
# x = loan_train.drop(['status'], axis=1)
# y = loan_train.status
# clf.set_params(kernel='linear').fit(x, y)

# loan_test = pd.read_csv('data/loan_test.csv', sep=';')
# test = loan_test.drop(['status'], axis=1)

# df = pd.DataFrame(data={'Id': loan_test.loan_id,
#                   'Predicted': clf.predict_proba(test)[:, 1]})
# df.to_csv('data/submission.csv', index=False)

account_df = pd.read_csv('data/account.csv', sep=';')
client_df = pd.read_csv('data/client.csv', sep=';')
disp_df = pd.read_csv('data/disp.csv', sep=';')
loan_train_df = pd.read_csv('data/loan_train.csv', sep=';')

client_df['gender'] = client_df.apply(
    lambda row: 'F' if is_female(row['birth_number']) else 'M', axis=1)
client_df['birth_number'] = client_df.apply(
    lambda row: row['birth_number'] - 5000 if is_female(row['birth_number']) else row['birth_number'], axis=1)

tmp_df = pd.merge(account_df, disp_df, on='account_id')
tmp_df = pd.merge(tmp_df, loan_train_df, on='account_id')


result_df = pd.merge(tmp_df, client_df, on='client_id').rename(
    columns={'district_id_x': 'account_district_id', 'district_id_y': 'client_district_id'}
)

result_df.to_csv('data/result.csv', index=False)
print(result_df.head())
