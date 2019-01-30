# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:40:59 2018

@author: The Freaky Gamer
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer

pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#print(train["Embarked"].value_counts())

train['Embarked'] = train['Embarked'].fillna('S')
#print(train["Embarked"].value_counts())

train = train.drop(['Cabin', 'Ticket'], axis = 1)
test = test.drop(['Cabin', 'Ticket'], axis = 1)


train['Title'] = train['Name'].apply(lambda x : x.split(" ")[1].split(".")[0].strip())
test['Title'] = train['Name'].apply(lambda x : x.split(" ")[1].split(".")[0].strip())

train = train.drop(['Name', 'Title', 'PassengerId'], axis = 1)
train_res = train['Survived'].values
test = test.drop(['Name', 'Title', 'PassengerId'], axis = 1)
train = train.drop(['Survived'], axis = 1)

lb = LabelEncoder()
imp = Imputer(missing_values='NaN', strategy='mean', axis = 0)
X_train = train.iloc[:, :].values
X_train[:, 2:3] = imp.fit_transform(X_train[:, 2:3])
X_test = test.iloc[:, :].values
X_test[:, 2:3] = imp.fit_transform(X_test[:, 2:3])
X_test[:, 5:6] = imp.fit_transform(X_test[:, 5:6])

df_X_test = pd.DataFrame(X_test)
#print(df_X_test[5].valuecounts())
#train = pd.DataFrame(X_train)

X_train[:, 1] = lb.fit_transform(X_train[:, 1])
X_train[:, 6] = lb.fit_transform(X_train[:, 6])

X_test[:, 1] = lb.fit_transform(X_test[:, 1])
X_test[:, 6] = lb.fit_transform(X_test[:, 6])




hot = OneHotEncoder(categorical_features=[6])
X_train = hot.fit_transform(X_train).toarray()
X_test = hot.fit_transform(X_test).toarray()
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]



df_X_train = pd.DataFrame(X_train)


#predict results

'''from sklearn.ensemble import RandomForestClassifier

ran = RandomForestClassifier(n_estimators=3000, random_state=42)
ran.fit(X_train, train_res)

y_pred = ran.predict(X_train)'''

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy')
classifier.fit(X_train, train_res)
y_pred = classifier.predict(X_test)


test2 = pd.read_csv('test.csv')
df_out = pd.DataFrame({'PassengerId':test2['PassengerId'], 'Survived':y_pred})

df_out.to_csv("out.csv")


'''X_train = hot.fit_transform(X_train).toarray()


#train = train.iloc[]


#print()
#print(train.isnull().sum())
#print(test.isnull().sum())
#train_df = train.drop(["Cabin", "Ticket"], axis = 1)'''