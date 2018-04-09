import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# load the fe data
fe = pd.read_csv('../../data/fwi_final.csv')

# define what columns are predictors and what column is the response
x_columns = [str(x) for x in range(0, 127 + 1)]
y_column = ['id']

# loop to see when test size starts to effect results
for i in range(8, 0, -1) :
    ts = i * .1

    # create train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(fe[x_columns], fe[y_column], test_size = ts, random_state = 42)

    # run random forest classification
    t1 = time.time()
    rfc = RandomForestClassifier(n_estimators = 50, n_jobs=2)
    rfc.fit(x_train, y_train)

    # make prediction on test set
    pred = rfc.predict(x_test)
    t_rf = time.time() - t1

    # evaluate prediction accuracy
    ytst = np.empty(shape = 1)
    for i in range(0, len(pred)) :
        ytst = np.append(ytst,y_test.values[i][0])
    ytst = np.delete(ytst, 0)
    print("")
    print("test_size =", ts)
    print("------------------------------------------------")
    print(accuracy_score(ytst, pred))
    print(pd.crosstab(ytst, pred, rownames=['actual'], colnames=['preds']))


# create test and training sets
x_columns = [str(x) for x in range(0, 127 + 1)]
y_column = ['id']
x_train, x_test, y_train, y_test = train_test_split(fe[x_columns], fe[y_column], test_size = 0.33, random_state = 42)

# random forest
parameters = {'n_estimators':[100, 200, 300, 400, 500], 'max_features':[1, 2, 3, 4, 5]}
rf = RandomForestClassifier(random_state=2)
rf_clf = GridSearchCV(rf, param_grid=parameters)
rf_clf.fit(x_train, y_train)
ytst = np.empty(shape = 1)
y_hat_test = rf_clf.predict(x_test)
for i in range(0, len(y_hat_test)) :
    ytst = np.append(ytst,y_test.values[i][0])
ytst = np.delete(ytst, 0)
print(accuracy_score(ytst, y_hat_test))
print(pd.crosstab(ytst, y_hat_test, rownames=['actual'], colnames=['preds']))

# bagging
parameters = {'n_estimators':[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              'max_features':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
bag = BaggingClassifier(random_state=7)
bag_clf = GridSearchCV(bag, param_grid=parameters)
bag_clf.fit(x_train, y_train)
ytst = np.empty(shape = 1)
y_hat_test = bag_clf.predict(x_test)
for i in range(0, len(y_hat_test)) :
    ytst = np.append(ytst,y_test.values[i][0])
ytst = np.delete(ytst, 0)
print(accuracy_score(ytst, y_hat_test))
print(pd.crosstab(ytst, y_hat_test, rownames=['actual'], colnames=['preds']))


# boosting
parameters = {'n_estimators':[30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
              'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10]}
boost = AdaBoostClassifier(random_state=9)
boost_clf = GridSearchCV(boost, param_grid=parameters)
boost_clf.fit(x_train, y_train)
ytst = np.empty(shape = 1)
y_hat_test = boost_clf.predict(x_test)
for i in range(0, len(y_hat_test)) :
    ytst = np.append(ytst,y_test.values[i][0])
ytst = np.delete(ytst, 0)
print(accuracy_score(ytst, y_hat_test))
print(pd.crosstab(ytst, y_hat_test, rownames=['actual'], colnames=['preds']))






from sklearn import svm
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(decision_function_shape='ovo')
svc.fit(x_train, y_train)
dec = svc.decision_function([[128]]t
dec.shape
y_hat_test = svc.predict(x_test)

clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(x_train, y_train)
print(kfold)















# decision tree
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# export_graphviz(rfc.estimators_[0], out_file=dot_data, filled=True, rounded=True, special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# from IPython.display import Image
# Image(graph[0].create_png())

