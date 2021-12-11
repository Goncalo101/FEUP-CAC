import numpy as np

def count_withdrawal(trans_type):
    return sum(trans_type == 'withdrawal')

def count_credit(trans_type):
    return sum(trans_type == 'credit')

def mean_withdrawal(trans_type):
    return np.mean(trans_type == 'withdrawal')

def mean_credit(trans_type):
    return np.mean(trans_type == 'credit')

def std_withdrawal(trans_type):
    return np.std(trans_type == 'withdrawal')

def std_credit(trans_type):
    return np.std(trans_type == 'credit')

def count_credit_op(trans_op):
    return sum(trans_op == 'credit in cash')

def count_collection_op(trans_op):
    return sum(trans_op == 'collection from another bank')

def count_withdrawal_op(trans_op):
    return sum(trans_op == 'withdrawal in cash')

def count_remittance_op(trans_op):
    return sum(trans_op == 'remittance to another bank')

def count_ccw_op(trans_op):
    return sum(trans_op == 'credit card withdrawal')

def count_interest_op(trans_op):
    return sum(trans_op == 'interest credited')

def mean_credit_op(trans_op):
    return np.mean(trans_op == 'credit in cash')

def mean_collection_op(trans_op):
    return np.mean(trans_op == 'collection from another bank')

def mean_withdrawal_op(trans_op):
    return np.mean(trans_op == 'withdrawal in cash')

def mean_remittance_op(trans_op):
    return np.mean(trans_op == 'remittance to another bank')

def mean_ccw_op(trans_op):
    return np.mean(trans_op == 'credit card withdrawal')

def mean_interest_op(trans_op):
    return np.mean(trans_op == 'interest credited')

def std_credit_op(trans_op):
    return np.std(trans_op == 'credit in cash')

def std_collection_op(trans_op):
    return np.std(trans_op == 'collection from another bank')

def std_withdrawal_op(trans_op):
    return np.std(trans_op == 'withdrawal in cash')

def std_remittance_op(trans_op):
    return np.std(trans_op == 'remittance to another bank')

def std_ccw_op(trans_op):
    return np.std(trans_op == 'credit card withdrawal')

def std_interest_op(trans_op):
    return np.std(trans_op == 'interest credited')

def bal_min(balance):
    return balance.abs().min()

def bal_range(balance):
    return balance.max() - balance.min()

def days(trans_date):
    return (trans_date.max() - trans_date.min()).days