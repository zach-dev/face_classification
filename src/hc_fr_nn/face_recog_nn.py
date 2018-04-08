import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from nn import neural_network

def face_recog_nn():

    # load the fe data
    fe = pd.read_csv('../../data/fwi_final.csv')

    # define what columns are predictors and what column is the response
    x_columns = [str(x) for x in range(0, 127 + 1)]
    y_column = ['id']

    # create train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(fe[x_columns], fe[y_column], test_size = 0.33, random_state = 42)
    print("x_train original shape: " + str(x_train.shape))
    print("x_test original shape: " + str(x_test.shape))

    # before transposing use stadardScaler to standardize data
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # reshape the training and test variables for use by NN
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

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


    # initialize constants defining the model
    layer_dims = [128, 64, 32, 16, 12]  # 4-layer model
    nn = neural_network(layer_dims, x_train, y_train, num_iterations=3000, print_cost=True)
    nn.train()

    pred_train = nn.predict()
    pred_test = nn.predict(x_test, y_test)


face_recog_nn()
