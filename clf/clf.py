#!/usr/bin/env python
# coding: utf-8

from abc import ABCMeta, abstractmethod
import csv

from .load_data import test_data, ids

class CLF:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parameters):
        self.hp_choice = dict([(key, hp.choice(key, value))
                               for key, value in parameters.items()])
        self.clf = None

    @abstractmethod
    def fit(self, X, y):
        pass


    def output(self):
        output = self.clf.predict(test_data).astype(int)

        predictions_file = open("myfirstforest.csv", "wb")
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(["PassengerId","Survived"])
        open_file_object.writerows(zip(ids, output))
        predictions_file.close()
        print 'Done.'
