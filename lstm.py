import numpy as np
import matplotlib.pyplot as plt


# def error functions
def errors(type,actual,prediction):
    #mse
    mse=np.mean((actual - prediction) ** 2)
    print("\n"+str(type)+" MSE error: {:.4f}".format( mse))
    #rmse
    rmse=np.sqrt(((actual - prediction) ** 2).mean())
    print(str(type)+" RMSE error: {:.4f}".format(rmse))




# plotting of graphs
def plot(actual, train, test):
    plt.plot(actual, label="Actual")
    plt.plot(train, label="Train prediction")
    test = [i for i in test]
    # connect train and test lines
    test.insert(0, train[-1])
    # x values for test prediction plot
    plt.plot([x for x in range(len(train) - 1, len(train) + len(test) - 1)], test, label="Test prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title("Stock Prediction")
    plt.legend()
    plt.grid()
    plt.show()




class LSTM:

    def __init__(self, input=2, lstm_cell_weights=2, output=1, learning_rate=0.5):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = input
        self.lstm_cell_weights = lstm_cell_weights
        self.output_nodes = output

        # fw = weights for forget gate
        self.fw = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        # iw = weights for input gate
        self.iw = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        # ow = weights for ouput gate
        self.ow = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        # cw = weights for candidate
        self.cw = np.random.randn(self.input_nodes, self.lstm_cell_weights).T
        # low = weights from LSTM cells to output
        self.low = np.random.randn(2, 1).T

        # set default LSTM cell state
        self.cell_state = [[1, 1] for i in range(100)]
        self.cell_state = np.array(self.cell_state, dtype=float)
        self.cell_state = np.array(self.cell_state, ndmin=2).T

        # learning rate
        self.learn = learning_rate

    # sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # tanh activation function
    def tanh(self, x):
        return 1 - np.square(np.tanh(x))

    def forget_gate(self, gate_input, out=1):
        gate_input = np.dot(self.fw, gate_input)
        gate_input = out * gate_input
        gate_output = self.sigmoid(gate_input)
        self.cell_state = self.cell_state * gate_output

    def input_gate(self, gate_input, out=1):
        gate_input_1 = np.dot(self.iw, gate_input)
        gate_input_1 = out * gate_input_1
        gate_input_2 = np.dot(self.cw, gate_input)
        gate_input_2 = out * gate_input_2
        gate_output = self.sigmoid(gate_input_1) * self.tanh(gate_input_2)
        self.cell_state = self.cell_state + gate_output

    def output_gate(self, gate_input, out=1):
        gate_input = np.dot(self.ow, gate_input)
        gate_input = out * gate_input
        gate_output = self.sigmoid(gate_input)
        out = self.tanh(self.cell_state) * gate_output

        return out

    def forward(self, input_1, input_2, input_3):
        self.cell_state = [[1, 1] for i in range(len(input_1[0]))]
        self.cell_state = np.array(self.cell_state, dtype=float)
        self.cell_state = np.array(self.cell_state, ndmin=2).T
        # Pass input though first lstm cell
        self.forget_gate(input_1)
        self.input_gate(input_1)
        fin_out = self.output_gate(input_1)
        # Pass input though second lstm cell
        self.forget_gate(input_2, fin_out)
        self.input_gate(input_2, fin_out)
        fin_out = self.output_gate(input_2, fin_out)
        # Pass input though third lstm cell
        self.forget_gate(input_3, fin_out)
        self.input_gate(input_3, fin_out)
        fin_out = self.output_gate(input_3, fin_out)
        # dot product of final cell output and output weights
        final_input = np.dot(self.low, fin_out)
        # compute the neural networks output
        final_output = self.sigmoid(final_input)
        return final_output, fin_out

    def error(self, target, final_output):
        output_error = target - final_output
        hidden_error = np.dot(self.low.T, output_error)

        return output_error, hidden_error

    def backpropagation(self, train_x1, train_x2, train_x3, fin_out, final_output, output_error,
                        cell_error):
        self.low += self.learn * np.dot((output_error * final_output * (1.0 - final_output)), fin_out.T)
        self.fw += self.learn * np.dot((cell_error * fin_out * (1.0 - fin_out)), train_x1.T)
        self.iw += self.learn * np.dot((cell_error * fin_out * (1.0 - fin_out)), train_x2.T)
        self.cw += self.learn * np.dot((cell_error * fin_out * (1.0 - fin_out)), train_x2.T)
        self.ow += self.learn * np.dot((cell_error * fin_out * (1.0 - fin_out)), train_x3.T)

    def train(self, train_x1, train_x2, train_x3, target):
        # convert lists to 2d arrays
        train_x1 = np.array(train_x1, ndmin=2).T
        train_x2 = np.array(train_x2, ndmin=2).T
        train_x3 = np.array(train_x3, ndmin=2).T
        target = np.array(target, ndmin=2).T

        # forward propagation 
        final_output, fin_out = self.forward(train_x1, train_x2, train_x3)

        # calculate output and cell output error
        output_error, cell_error = self.error(target, final_output)

        # back propagation
        self.backpropagation(train_x1, train_x2, train_x3, fin_out, final_output, output_error,
                             cell_error)

        return final_output

    def test(self, test_x1, test_x2, test_x3):
        # transpose input
        test_x1 = test_x1.T
        test_x2 = test_x2.T
        test_x3 = test_x3.T
        # forward propagation
        final_output, fin_out = self.forward(test_x1, test_x2, test_x3)
        # return final input
        return final_output
