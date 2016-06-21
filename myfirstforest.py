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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# define title
reg = re.compile(r' [A-Za-z]+\.')
title_set = set(train_df['Name'].map(lambda x: reg.search(x).group(0)).values)
title_dict = dict([x, i] for i, x in enumerate(title_set))

print title_dict

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

#title
train_df['title'] = train_df['Name'].map(lambda x: title_dict[reg.search(x).group(0)]).astype(int)

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
median_age_dict = {}

for title in title_dict.values():
    filtered = train_df[ title == train_df.title ]
    med = filtered['Age'].dropna().median()
    median_age_dict[title] = med

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    for title in title_dict.values():
            train_df[ title == train_df.title ].loc[ (train_df.Age.isnull()), 'Age'] = median_age

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Add
# https://www.kaggle.com/dataisknowledge/titanic/visualization-and-feature-engineering
train_df['family']=train_df.SibSp+train_df.Parch
train_df.family=train_df.family.apply(lambda x: 1 if x>0 else 0 )
#total no of family members onboard
train_df['family_size']=train_df.SibSp+train_df.Parch+1

train_df['Cabin'] = train_df.Cabin.astype(str).map(lambda x: x.split(' ')[0])
train_df['Cabin'] = train_df['Cabin'].map(lambda x: int(str(ord(x[0])) + x[1:]) if x != 'nan' else 0).astype(int)

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',
                          'SibSp', 'Parch', 'family', 'family_size', 'title'], axis=1)


#         int64
#

# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# title
_f = lambda x: title_dict[reg.search(x).group(0)] if reg.search(x).group(0) in title_dict else -1
test_df['title'] = test_df['Name'].map(_f).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()

for title in title_dict.values():
    filtered = test_df[ title == test_df.title ]
    med = filtered['Age'].dropna().median()
    median_age_dict[title] = med

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    for title in title_dict.values():
        test_df[ title == test_df.title ].loc[ (test_df.Age.isnull()), 'Age'] = median_age

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values


# Add
# https://www.kaggle.com/dataisknowledge/titanic/visualization-and-feature-engineering
test_df['family']=test_df.SibSp+test_df.Parch
test_df.family=test_df.family.apply(lambda x: 1 if x>0 else 0 )
test_df['family_size']=test_df.SibSp+test_df.Parch+1

test_df['Cabin'] = test_df.Cabin.astype(str).map(lambda x: x.split(' ')[0])
test_df['Cabin'] = test_df['Cabin'].map(lambda x: int(str(ord(x[0])) + x[1:]) if x != 'nan' else 0).astype(int)


# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId',
                        'SibSp', 'Parch', 'family', 'family_size', 'title'], axis=1)


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values


print 'Training...'

X = train_data[0::, 1::]
y = train_data[0::, 0]

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( X, y )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

print train_df.dtypes
#print 'Importance:'
#print forest.feature_importances_


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

result = pd.read_csv('myfirstforest.csv')
test = pd.read_csv('gendermodel.csv')

c = len(filter(lambda x: result['Survived'][x] == test['Survived'][x], xrange(len(test))))

print "Accurate:", float(c) / len(test)

