#!/usr/bin/env python
# coding: utf-8

import csv
from numpy import nan
import xgboost as xgb
from hyperopt import fmin, tpe, hp, rand
from sklearn import cross_validation
from .clf import CLF

from .load_data import train_df, test_data, ids


class XGB:

    def __init__(self):
        self.parameters = {
            'n_estimators': [100, 250, 500, 1000],
            'max_depth': [3, 5, 10, 15, 20, 25, 30, 50, 100],
            'reg_alpha': [3.0, 4.0],
            'missing': [nan, 1.0, 2.0, 3.0]
        }

        self.hp_choice = dict([(key, hp.choice(key, value))
                               for key, value in self.parameters.items()])

        self.clf = None

    def fit(self, X, y, ensemble=40, param={}):

        def estimator(args):
            print "Args:", args
            forest = xgb.XGBClassifier(**args)

            trainX, testX, trainy, testy = cross_validation.train_test_split(
                X, y, test_size=0.4, random_state=0)

            forest.fit( trainX, trainy )

            #    print 'Predicting...'
            acu = forest.score(testX, testy)

            print "Accurate:", acu
            return -acu

        if ensemble:
            best = fmin(estimator, self.hp_choice, algo=tpe.suggest, max_evals=ensemble)
            best = dict([(key, self.parameters[key][value]) for key, value in best.items()])
        else:
            best = param
        print "\nBest Model..."

        estimator(best)

        self.clf = xgb.XGBClassifier(**best)
        self.clf.fit( X, y)

    def output(self):
        output = self.clf.predict(test_data).astype(int)

        predictions_file = open("myfirstforest.csv", "wb")
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(["PassengerId","Survived"])
        open_file_object.writerows(zip(ids, output))
        predictions_file.close()
        print 'Done.'


CLF.register(XGB)
