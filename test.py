from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

import pandas as pd

loan_train = pd.read_csv('data/loan_train.csv', sep=';')
clf = SVC(probability=True)
x = loan_train.drop(['status'], axis=1)
y = loan_train.status
clf.set_params(kernel='linear').fit(x, y)

loan_test = pd.read_csv('data/loan_test.csv', sep=';')
test = loan_test.drop(['status'], axis=1)

df = pd.DataFrame(data={'Id': loan_test.loan_id, 'Predicted': clf.predict_proba(test)[:, 1]})
df.to_csv('data/submission.csv', index=False)
