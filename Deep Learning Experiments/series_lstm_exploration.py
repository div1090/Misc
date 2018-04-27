import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class Series:
    Squares,Cubes,AP,GP = range(0,4)

class Series_Generator(object):

    def __init__(self):
        self.gen_sequence = {
            Series.Squares: self.gen_squares,
            Series.Cubes: self.gen_cubes,
            Series.AP: self.gen_AP,
            Series.GP: self.gen_GP
        }

    def gen_squares(self, n=0, N=100):
        lst = []
        for t in range(n, N):
            lst.append(t*t)
        return lst
    def gen_cubes(self, n=0, N=100):
        lst = []
        for t in range(n, N):
            lst.append(t*t*t)
        return lst
    def gen_AP(self, n=0, N = 100, d=1):
        lst = []
        for t in range(0, N):
            lst.append(n + (t-1)*d)
        return lst
    def gen_GP(self, n=0, N = 100, d=2):
        lst = []
        for t in range(0, N):
            lst.append(n * pow(d,t-1))
        return lst

class SequentialLSTM(object):
    def __init__(self, lookback = 1):
        self.epochs = 100
        self.batch_size = 1
        self.rnn_length = lookback
        self.create_model()


    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, self.rnn_length)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose = 4)

    def predict(self, X):
        return self.model.predict(X)

    def mse(self, y, z):
        return math.sqrt(mean_squared_error(y,z))


def prepare_data(series):
    df = pd.DataFrame(series)
    dataset = df.astype('float32')

    return dataset


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def run(series, look_back):
    sg = Series_Generator()
    seq = sg.gen_sequence[series](n=1,N=1000)

    # split into train and test sets
    dataset = prepare_data(seq)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = SequentialLSTM(look_back)
    model.train(trainX,trainY)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    print(trainPredictPlot.shape, trainPredict.shape, dataset.shape)
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(seq)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset)) # Ground Truth series
    plt.plot(trainPredictPlot) # Train preds
    plt.plot(testPredictPlot, label= "te_" + str(look_back)) # Test preds
    # plt.show()

if __name__ == '__main__':
    for i in range(5, 12, 2):
        run(Series.Squares, i)

    plt.legend(mode="expand", borderaxespad=0.)
    plt.show()