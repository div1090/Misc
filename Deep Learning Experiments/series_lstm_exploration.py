import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(15)

class Series:
    Powers,AP,GP, Random, Arc = range(0,5)

class Series_Generator(object):

    def __init__(self, isSeries = True):
        self.gen_sequence = {
            Series.Powers: self.gen_powers,
            Series.AP: self.gen_AP,
            Series.GP: self.gen_GP,
            Series.Random: self.gen_Random,
            Series.Arc: self.gen_arc
        }
        self.isSeries = isSeries

    def gen_powers(self, n=0, N=100, d = 2):
        lst = []
        for t in range(n, N):
            lst.append(pow(t,d))
        if self.isSeries:
            return lst,None
        else:
            return lst,range(n,N)

    def gen_AP(self, n=0, N = 100, d=1):
        lst = []
        for t in range(0, N):
            lst.append(n + (t-1)*d)
        if self.isSeries:
            return lst,None
        else:
            return lst,range(0,N)

    def gen_GP(self, n=0, N = 100, d=2):
        lst = []
        for t in range(0, N):
            lst.append(n * pow(d,t-1))
        if self.isSeries:
            return lst,None
        else:
            return lst,range(0,N)

    def gen_Random(self, n=0,N =100, d = 7):
        lst = []
        for t in range(n, N):
            lst.append(np.random.rand())
        if self.isSeries:
            return lst,None
        else:
            return lst,range(n,N)

    def gen_arc(self, n=0,N = 45, d = 5):
        lst = []
        nst = []
        for t in range(-d*100, d*100):
            x = math.sqrt(pow(d, 2) - pow(t / 100, 2))
            lst.append(-x)
            nst.append(x)
        nst.extend(lst)
        if self.isSeries:
            return nst
        else:
            x = []
            y = []
            for t in range(-d*100, d*100):
                x.append(t/100)
                y.append(-t/100)
            x.extend(y)
            return nst, x

class Regression(object):
    def __init__(self,var):
        self.epochs = 100
        self.batch_size = 10
        self.model = None
        self.variables = var
        self.create_model()


    def create_model(self):
        if self.model == None:
            self.model = Sequential()
            self.model.add(Dense(1, input_shape=(1,)))
            self.model.add(Dense(8))
            self.model.add(Dropout(0.5))
            self.model.add(Activation('relu'))
            self.model.add(Dense(8, activation='relu'))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose = 0)

    def predict(self, X):
        return self.model.predict(X)

    def mse(self, y, z):
        return math.sqrt(mean_squared_error(y,z))


class SequentialLSTM(object):
    def __init__(self, lookback = 1):
        self.epochs = 100
        self.batch_size = 10
        self.rnn_length = lookback
        self.model = None
        self.create_model()


    def create_model(self):
        if self.model == None:
            self.model = Sequential()
            self.model.add(LSTM(4, input_shape=(1, self.rnn_length)))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose = 0)

    def predict(self, X):
        return self.model.predict(X)

    def mse(self, y, z):
        return math.sqrt(mean_squared_error(y,z))


def prepare_data(series):
    print(series)
    df = pd.DataFrame(series)
    dataset = df.astype('float32')

    return dataset


# convert an array of values into a dataset matrix
def create_lstm_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    # reshape input to be [samples, time steps, features]
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))
    return dataX, dataY

val_error = []
train_error = []

# def run(series, look_back, n, N, d, model ):
#     sg = Series_Generator(isSeries = False)
#     seq,domain = sg.gen_sequence[series](n,N,d)
#
#     dataset = prepare_data(seq)
#
#     # normalize the dataset
#     scaler = MinMaxScaler(feature_range=(0,1))
#     dataset = scaler.fit_transform(dataset)
#
#     # split into train and test sets
#     train_size = int(len(dataset) * 0.67)
#     train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#
#
#     # FOR LSTM
#     # # reshape into X=t and Y=t+1
#     # trainX, trainY = create_lstm_dataset(train, look_back)
#     # testX, testY = create_lstm_dataset(test, look_back)
#
#     trainX,testX = np.array(domain[0:train_size]), np.array(domain[train_size:len(dataset)])
#     trainX = np.reshape(trainX, (trainX.shape[0], 1))
#     testX = np.reshape(testX, (testX.shape[0], 1))
#     trainY, testY = train, test
#
#     print(trainX)
#
#     # create and fit the network
#     model.train(trainX,trainY)
#
#     # make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)
#
#     # calculate root mean squared error (On the normalized data)
#     trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
#     print('Test Score: %.2f RMSE' % (testScore))
#
#     # add to train, test error
#     train_error.append(trainScore)
#     val_error.append(testScore)
#
#     # invert predictions (reverse the normalization transform we applied)
#     trainPredict = scaler.inverse_transform(trainPredict)
#     # trainY = scaler.inverse_transform([trainY])
#     testPredict = scaler.inverse_transform(testPredict)
#     # testY = scaler.inverse_transform([testY])
#
#     # shift train predictions for plotting
#     trainPredictPlot = np.empty_like(dataset)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
#     # shift test predictions for plotting
#     testPredictPlot = np.empty_like(dataset)
#     testPredictPlot[:, :] = np.nan
#     testPredictPlot[len(trainPredict)+(look_back*2)+1:len(seq)-1, :] = testPredict
#
#     # plot baseline and predictions
#     plt.figure(2)
#     plt.plot(scaler.inverse_transform(dataset)) # Ground Truth series
#     plt.plot(trainPredictPlot) # Train preds
#     plt.plot(testPredictPlot, label= "te_" + str(d)) # Test preds
#     # plt.legend(mode="expand", borderaxespad=0., bbox_to_anchor=(1, 0))
#     # plt.show()


def run(series, look_back, n, N, d, model ):
    sg = Series_Generator(isSeries = False)
    seq,domain = sg.gen_sequence[series](n,N,d)

    dataset = prepare_data(seq)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


    # FOR LSTM
    # # reshape into X=t and Y=t+1
    # trainX, trainY = create_lstm_dataset(train, look_back)
    # testX, testY = create_lstm_dataset(test, look_back)

    trainX,testX = np.array(domain[0:train_size]), np.array(domain[train_size:len(dataset)])
    trainX = np.reshape(trainX, (trainX.shape[0], 1))
    testX = np.reshape(testX, (testX.shape[0], 1))
    trainY, testY = train, test

    # create and fit the network
    model.train(trainX,trainY)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # calculate root mean squared error (On the normalized data)
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # add to train, test error
    train_error.append(trainScore)
    val_error.append(testScore)

    # invert predictions (reverse the normalization transform we applied)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    # shift train predictions for plotting

    # plot baseline and predictions
    plt.figure(2)
    plt.plot(trainX,trainPredict) # Train preds
    plt.plot(trainX, trainY)
    plt.plot(testX, testPredict, label= "te_" + str(d)) # Test preds
    plt.plot(testX, testY, label = "gt_" + str(d))


if __name__ == '__main__':
    model = None
    for i in (1,3,5):
        if model is None:
            model = Regression(1)
        run(series = Series.Arc, look_back= 1, n = 1, N = 100, d = i, model = model)
        model = None


    plt.legend(mode="expand", borderaxespad=0., bbox_to_anchor=(0.8, 0.2))
    plt.show()

    plt.figure(1)
    plt.plot(train_error, label = "train")
    plt.plot(val_error, label = "eval")
    plt.legend(mode="expand", borderaxespad=0., bbox_to_anchor=(0.8, 0.2))
    plt.show()
