import logging
import pandas as pd
from sklearn.impute import SimpleImputer

def open_csv(filename, format=None):
    return pd.read_csv(filename, sep=';', dtype={'k_symbol': str, 'bank': str}, na_values=['?'])


def is_female(number):
    return str(number)[2] == '5' or str(number)[2] == '6'


def date_to_str(row, date_name='date'):
    date_str = str(int(row[date_name]))
    return f'19{date_str[0:2]}-{date_str[2:4]}-{date_str[4:]}'


class Model:
    def __init__(self):
        pass

    def get_accounts(self):
        df = open_csv('./data/account.csv')
        df['date'] = df.apply(date_to_str, axis=1)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        return df

    def get_cards(self, op='test'):
        df = open_csv(f'./data/client_{op}.csv')
        df['date'] = df.apply(date_to_str, axis=1, args=['issued'])
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        return df

    def get_clients(self):
        df = open_csv('./data/client.csv')
        df['gender'] = df.apply(lambda row: 0 if is_female(
            row['birth_number']) else 1, axis=1)
        df['age'] = df.apply(lambda row: 2021 -
                             ((row['birth_number'] // 10000) + 1900), axis=1)
        df['birth_number'] = df.apply(lambda row: row['birth_number'] - 5000 if is_female(
            row['birth_number']) else row['birth_number'], axis=1)

        df['birth_number'] = df.apply(
            date_to_str, axis=1, args=['birth_number'])
        df['birth_number'] = pd.to_datetime(
            df['birth_number'], format='%Y-%m-%d')
        return df

    def get_disps(self):
        df = open_csv('./data/disp.csv')
        df = df.drop(['disp_id'], axis=1)
        return df

    def get_districts(self):
        df = open_csv('./data/district.csv')
        # TODO: Remover mais tarde o name
        df = df.rename(columns={'code ': 'district_id',
                       'name ': 'district_name'})
        
        imp_mean = SimpleImputer(strategy='mean')
        imp_mean.fit(df[[
            'unemploymant rate \'95 ',
            'no. of commited crimes \'95 ']])
        df[[
            'unemploymant rate \'95 ',
            'no. of commited crimes \'95 ']] = imp_mean.transform(df[[
                'unemploymant rate \'95 ',
                'no. of commited crimes \'95 ']])
        return df

    def get_loans(self, op='test'):
        df = open_csv(f'./data/loan_{op}.csv')
        df['date'] = df.apply(date_to_str, axis=1)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        df = df.rename(columns={'date': 'loan_date', 'amount': 'loan_amount'})
        df['status'] = df['status'].fillna('')
        return df

    def get_transactions(self, op='test'):
        df = open_csv(f'./data/trans_{op}.csv')
        df['date'] = df.apply(date_to_str, axis=1)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.loc[df['type'] == 'withdrawal in cash','type'] = 'withdrawal'
        df.loc[df['type'] == 'withdrawal', 'amount'] *= -1
        df.loc[df['operation'].isna(
        ), 'operation'] = df.loc[df['operation'].isna(), 'k_symbol']
        df = df.drop(['k_symbol', 'bank', 'account'], axis=1)
        df = df.rename(columns={'date': 'trans_date', 'amount': 'trans_amount', 'type': 'trans_type'})

        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s - %(message)s')

    model = Model()
    df = model.get_districts()
    print(df)
