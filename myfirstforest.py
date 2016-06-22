""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
import re

from load_data import train_data, test_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from hyperopt import fmin, tpe, hp, rand
from sklearn import cross_validation



print 'Training...'

X = train_data[0::, 1::]
y = train_data[0::, 0]

parameters = {
    'n_estimators': [100, 150, 200, 250],
    'max_features': [3, 4],
    'random_state': [0],
    'n_jobs': [2],
    'min_samples_split': [3, 5, 10, 15, 20, 25],
    'max_depth': [3, 5, 10, 15, 20, 25]
}

hp_choice = dict([(key, hp.choice(key, value))
                  for key, value in parameters.items()])
#clf = GridSearchCV(RandomForestClassifier(), parameters)

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

best = fmin(estimator, hp_choice, algo=tpe.suggest, max_evals=40)
best = dict([(key, parameters[key][value]) for key, value in best.items()])
print "\nBest Model..."

estimator(best)

forest = RandomForestClassifier(**best)
forest.fit( X, y)

print 'Predicting...'
output = forest.predict(test_data).astype(int)

print train_df.dtypes
print 'Importance:'
print forest.feature_importances_


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

result = pd.read_csv('myfirstforest.csv')
test = pd.read_csv('gendermodel.csv')

c = len(filter(lambda x: result['Survived'][x] == test['Survived'][x], xrange(len(test))))

acu =  float(c) / len(test)
print "Accurate:", acu
