#!/usr/bin/env python
# coding: utf-8

import numpy as np
import csv

from clf.load_data import train_data, test_data, train_df, ids

from clf.randomforest import RF
from clf.xgb import XGB
from clf.log_reg import LR


def output(clf):
    '''
    new_featrue = np.dot(test_data[:, idx], np.array(fscore))
    new_featrue = new_featrue / np.linalg.norm(new_featrue)
    _test_data = np.hstack((test_data, new_featrue[np.newaxis].T))
    '''
    new_featrue = xgb_clf.clf.predict_proba(test_data)
    new_feature_ = new_feature[:,0] * new_feature[:, 1]
    new_featrue2 = rf_clf.clf.predict_proba(test_data)
    new_feature2_ = new_feature2[:, 0] * new_feature2[:, 1]

    _test_data = np.hstack((new_feature, new_feature_[np.newaxis].T,
                            new_feature2, new_feature2_[np.newaxis].T))

    output = clf.clf.predict(_test_data).astype(int)

    predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'


print 'Training...'

X = train_data[0::, 1::]
y = train_data[0::, 0]


xgb_clf = XGB()
xgb_clf.fit(X, y)

rf_clf = RF()
rf_clf.fit(X, y)

'''
fscore = xgb_clf.clf.booster().get_fscore().values()
idx = [int(v[1:]) for v in xgb_clf.clf.booster().get_fscore().keys()]

new_featrue = np.dot(X[:, idx], np.array(fscore))
new_featrue = new_featrue / np.linalg.norm(new_featrue)
'''
new_feature = xgb_clf.clf.predict_proba(X)
new_feature_ = new_feature[:, 0] * new_feature[:, 1]

new_feature2 = rf_clf.clf.predict_log_proba(X)
new_feature2_ = new_feature2[:, 0] * new_feature2[:, 1]

X = np.hstack((new_feature, new_feature_[np.newaxis].T,
               new_feature2, new_feature2_[np.newaxis].T))

lr_clf = LR()
lr_clf.fit(X, y)
output(lr_clf)
