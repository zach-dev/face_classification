import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def eval_pred_accuracy(y_test, pred) :
    print("in")

    # evaluate prediction accuracy
    ytst = np.empty(shape = 1)
    for i in range(0, len(pred)) :
        ytst = np.append(ytst,y_test.values[i][0])
    ytst = np.delete(ytst, 0)
    print("in2")
    print("")
    print("------------------------------------------------")
    a_score = accuracy_score(ytst.astype(int), pred.astype(int))
    print("in3")
    print(accuracy_score(ytst.astype(int), pred.astype(int)))
    print(pd.crosstab(ytst.astype(int), pred[:,0].astype(int), rownames=['actual'], colnames=['preds']))

    return(a_score)

# load the fe data
fe = pd.read_csv('../../data/fwi_final.csv')

# factorize id column
fe['id'], _ = pd.factorize(fe['id'])

# create a histogram of the id column
#gn = np.random.randn(1000)
#plt.hist(fe['id'])
#plt.title('Face ID Histogram')
#plt.xlabel('ID')
#plt.ylabel('Value')

# create test and training sets
x_columns = [str(x) for x in range(0, 127 + 1)]
y_column = ['id']
x_train, x_test, y_train, y_test = train_test_split(fe[x_columns], fe[y_column], test_size = 0.33, random_state = 42)

# knn
# define num_neig
num_neigh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

for nn in num_neigh :
    # look at the five closest neighbors.
    knn = KNeighborsRegressor(n_neighbors = nn)

    # Fit the model on the training data.
    knn.fit(x_train, y_train)

    # Make point predictions on the test set using the fit model.
    pred = knn.predict(x_test)

    # evaluate prediction accuracy
    a_score = eval_pred_accuracy(y_test, pred)
    print('nn =', nn)
    print('accuracy score =', a_score)

# final run
t1 = time.time()
# look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors = 1)

# Fit the model on the training data.
knn.fit(x_train, y_train)

# Make point predictions on the test set using the fit model.
pred = knn.predict(x_test)
t_knn = time.time() - t1

# evaluate prediction accuracy
a_score = eval_pred_accuracy(y_test, pred)
print('t_knn =', t_knn)
print('accuracy score =', a_score)
