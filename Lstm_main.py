from lstm import LSTM, plot, errors
import numpy as np
import pandas as pd


def main():
    # read into dataframe
    df = pd.read_csv("sbi.csv")
    # extract only the adjusted close prices of the selected stock
    df = df['Adj Close']
    normalization_value=1000
    #1st two days
    train_x1 = [[df[i-6], df[i-5]] for i in range(len(df[:550])) if i >= 6]
    #3rd and 4th day
    train_x2 = [[df[i-4], df[i-3]] for i in range(len(df[:550])) if i >= 6]
    #5th and 6th day
    train_x3 = [[df[i-2], df[i-1]] for i in range(len(df[:550])) if i >= 6]
    #7th day or targeted train_pred
    train_y = [[i] for i in df[6:550]]

    #convert into arrays
    train_x1 = np.array(train_x1, dtype=float)
    train_x2 = np.array(train_x2, dtype=float)
    train_x3 = np.array(train_x3, dtype=float)
    train_y = np.array(train_y, dtype=float)

    # Normalize
    train_x1= train_x1/normalization_value
    train_x2 = train_x2/normalization_value
    train_x3 = train_x3/normalization_value
    train_y = train_y/normalization_value

    # create neural network
    NN = LSTM()

    # number of training cycles
    training_cycles = 100
    # train the neural network
    for cycle in range(training_cycles):
        print("training cycle:"+str(cycle))
        for n in train_x1:
            train_pred = NN.train(train_x1, train_x2, train_x3, train_y)

    # print various accuracies
    errors("train",train_y,train_pred)
    # de-Normalize
    train_pred = np.array(train_pred, dtype=float)
    train_pred *=normalization_value
    train_y *=normalization_value

    # transpose
    train_pred = train_pred.T




    test_x1 = [[df[i - 6], df[i - 5]] for i in range(550, 650)]
    test_x2 = [[df[i - 4], df[i - 3]] for i in range(550, 650)]
    test_x3 = [[df[i - 2], df[i - 1]] for i in range(550, 650)]
    test_y = [[i] for i in df[550:650]]

    test_x1 = np.array(test_x1, dtype=float)
    test_x2 = np.array(test_x2, dtype=float)
    test_x3 = np.array(test_x3, dtype=float)
    test_y = np.array(test_y, dtype=float)


    # Normalize

    test_x1 = test_x1/normalization_value
    test_x2 = test_x2/normalization_value
    test_x3 = test_x3/normalization_value
    test_y = test_y/normalization_value

    # test_pred the network with unseen data
    test_pred = NN.test(test_x1, test_x2, test_x3)
    test_pred = np.array(test_pred, dtype=float)

    # print various accuracies
    errors("test",test_y, test_pred)

    # de-Normalize data
    test_pred *=normalization_value

    test_y *=normalization_value


    # transplose test_pred results
    test_pred = test_pred.T



    # plotting training and test_pred results on same graph

    plot(df[0:650], train_pred, test_pred)


if __name__ == '__main__':
    main()