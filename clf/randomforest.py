#!/usr/bin/env python
# coding: utf-8

import csv
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, rand
from sklearn import cross_validation
from .clf import CLF

from .load_data import train_df, test_data, ids

class RF:

    def __init__(self):
        self.parameters = {
            'n_estimators': [100, 150, 200, 250],
            'max_features': [3, 4],
            'random_state': [0],
            'n_jobs': [2],
            'min_samples_split': [3, 5, 10, 15, 20, 25],
            'max_depth': [3, 5, 10, 15, 20, 25]
        }

        self.hp_choice = dict([(key, hp.choice(key, value))
                               for key, value in self.parameters.items()])

        self.clf = None

    def fit(self, X, y, ensemble=40, param={}):

        def estimator(args):
            print "Args:", args
            forest = RandomForestClassifier(**args)

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

        self.clf = RandomForestClassifier(**best)
        self.clf.fit( X, y)

        print 'Importance:'
        print train_df.dtypes
        print self.clf.feature_importances_

    def output(self):
        output = self.clf.predict(test_data).astype(int)

        predictions_file = open("myfirstforest.csv", "wb")
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(["PassengerId","Survived"])
        open_file_object.writerows(zip(ids, output))
        predictions_file.close()
        print 'Done.'


CLF.register(RF)
