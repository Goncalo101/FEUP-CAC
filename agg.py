import numpy as np


def count_credit(x):
    return sum(x == 'credit')

def count_withdrawal(x):
    return sum(x == 'withdrawal')

def mean_withdrawal(x):
    return np.mean(x=="withdrawal")
def mean_credit(x):
    return np.mean(x=="credit")

def std_withdrawal(x):
    return np.std(x=="withdrawal")
def std_credit(x):
    return np.std(x=="credit")

def cov_withdrawal(x):
    return np.cov(x=="withdrawal")
def cov_credit(x):
    return np.cov(x=="credit")