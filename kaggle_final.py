import pandas as pd
import numpy as np
import xgboost as xgb
import gc

from scipy.sparse import csr_matrix

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectFromModel


# Load the data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
test_final = pd.read_csv('input/test.csv')

# remove constant columns
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# split data into 10 pairs of train and test
split=10
skf = StratifiedKFold(train.TARGET.values,n_folds=split,shuffle=False,random_state=1729)

# for each pair of train and test, generate a model
features = train.columns[1:-1]
train_preds = None
test_preds = None
visibletrain = blindtrain = train_train
index = 0
print('Change num_rounds to 350')
num_rounds = 350
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.03
params["subsample"] = 0.83
params["colsample_bytree"] = 0.79
params["silent"] = 1
params["max_depth"] = 6
params["min_child_weight"] = 3
params["eval_metric"] = "auc"
for train_index, test_index in skf:
    print('Fold:', index)
    visibletrain = train_train.iloc[train_index]
    blindtrain = train_train.iloc[test_index]
    dvisibletrain = \
        xgb.DMatrix(csr_matrix(visibletrain[features]),
                    visibletrain.TARGET.values,
                    silent=True)
    dblindtrain = \
        xgb.DMatrix(csr_matrix(blindtrain[features]),
                    blindtrain.TARGET.values,
                    silent=True)
    watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
    clf = xgb.train(params, dvisibletrain, num_rounds,
                    evals=watchlist, early_stopping_rounds=50,
                    verbose_eval=False)

    blind_preds = clf.predict(dblindtrain)
    print('Blind Log Loss:', log_loss(blindtrain.TARGET.values,
                                      blind_preds))
    print('Blind ROC:', roc_auc_score(blindtrain.TARGET.values,
                                      blind_preds))
    index = index+1
    del visibletrain
    del blindtrain
    del dvisibletrain
    del dblindtrain
    gc.collect()
    dfulltrain = \
        xgb.DMatrix(csr_matrix(train_test[features]),
                    train.TARGET.values,
                    silent=True)
    dfulltest = \
        xgb.DMatrix(csr_matrix(test[features]),
                    silent=True)
    if(train_preds is None):
        train_preds = clf.predict(dfulltrain)
        test_preds = clf.predict(dfulltest)
    else:
        train_preds *= clf.predict(dfulltrain)
        test_preds *= clf.predict(dfulltest)
    del dfulltrain
    del dfulltest
    del clf
    gc.collect()

# get the average score of the 10 models
train_preds = np.power(train_preds, 1./index)
test_preds = np.power(test_preds, 1./index)
print('Average Log Loss:', log_loss(train_test.TARGET.values, train_preds))
print('Average ROC:', roc_auc_score(train_test.TARGET.values, train_preds))

submission = pd.DataFrame({"ID": test.ID, "TARGET": test_preds})
submission.to_csv("simplexgbtest2.csv", index=False)
print('Finish.')
