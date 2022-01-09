from Layer import *
from images import test_normal
import numpy as np

class cnn():
    def __init__(self, l, lr):
        self.layers=l
        self.learning_rate=lr
    
    def forward(self, input, target):
        output=input
        for layer in self.layers:
            output=layer.forward(output)
        #calculate loss
        prediction = list(output).index(max(output))
        correct = 1 if prediction == target else 0
        loss = -np.log(output[target])
        return output, correct, loss

    def backpropogate(self, input, target):
        output, _, loss = self.forward(input, target)
        grad = np.zeros(self.layers[-1].shape[1])
        grad[target] = -1/output[target]
        self.layers[-1].backpropogate(grad, self.learning_rate)
        """for layer in reversed(self.layers):
            grad = layer.backpropogate(grad, self.learning_rate)"""

    def train(self, train_data):
        for train_index in range(len(train_data)):
            self.backpropogate(train_data[0][train_index], train_data[1][train_index])

    def test(self, test_data):
        accuracy, avg_loss = 0, 0
        for test_index in range(len(test_data[0])):
            #print("{}/{}".format(test_index+1, len(test_data[0])))
            o, c, l = self.forward(test_data[0][test_index], test_data[1][test_index])
            accuracy+=c
            avg_loss+=l

        avg_loss/=len(test_data[0])
        accuracy*=100/len(test_data[0])

        print("Loss: {}, accuracy: {}%".format(avg_loss, accuracy))
        return avg_loss, accuracy
        

from tensorflow.keras.datasets.mnist import load_data
mnist = load_data()
training, testing = [np.divide(mnist[0][0], 254), mnist[0][1]], [mnist[1][0], mnist[1][1]]
test_net = cnn([convolutionLayer(2, [3, 3]), poolingLayer("MAX", [2, 2]), Softmax([13*13*2, 10])], 0.05)

test_size = 100

test_net.test([training[0][:test_size], training[1][:test_size]])
test_net.train([training[0][:1000], training[1][:test_size]])
test_net.test([training[0][:test_size], training[1][:test_size]])