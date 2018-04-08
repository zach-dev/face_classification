import numpy as np
import pandas as pd
from tf_utils_fr import convert_to_one_hot
from face_classification_nn import model
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

single_run = True

np.random.seed(1)

# load the fe data
fe = pd.read_csv('../../data/fwi_final.csv')

# define what columns are predictors and what column is the response
x_columns = [str(x) for x in range(0, 127 + 1)]
y_column = ['id']

# create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(fe[x_columns], fe[y_column], test_size=0.33, random_state=42)
print("x_train original shape: " + str(x_train.shape))
print("x_test original shape: " + str(x_test.shape))

# before transposing use stadardScaler to standardize data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape the training and test variables for use by NN
x_train = x_train.T
y_train = y_train.T
x_test =  x_test.T
y_test =  y_test.T

# convert Y's from data frame to numpy array
y_train = y_train.as_matrix()
y_test  = y_test.as_matrix()

# explore dataset
num_pred, m_train = x_train.shape
m_test = x_test.shape[1]
print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Number of predictors:" + str(num_pred))
print("x_train's shape: " + str(x_train.shape))
print("x_test's shape: " + str(x_test.shape))
print("y_train shape: " + str(y_train.shape))
print("y_test shape: " + str(y_test.shape))

# Convert training and test labels to one hot matrices
y_train = convert_to_one_hot(y_train, 12)
y_test  = convert_to_one_hot(y_test,  12)

if single_run:
    parameters, test_accuracy = model(x_train, y_train, x_test, y_test, learning_rate=0.00001, num_epochs=3000)
else:
    lrs = []
    tas = []

    for i in range(1, 10):
        lrs.append(i * 1e-1)
        parameters, test_accuracy = model(x_train, y_train, x_test, y_test, learning_rate=0.00001, num_epochs=3000)

        tas.append(test_accuracy)

    for i in range(len(lrs)):
        print(lrs[i], tas[i])
