from Layer import *
from images import test_normal
import numpy as np

e=0.0000001

class cnn():
    def __init__(self, l, lr):
        self.layers=l
        self.learning_rate=lr
    
    def forward(self, input, target, debug=False):
        output=input
        for layer in self.layers:
            output=layer.forward(output)
            if debug:
                try:
                    draw_image(output)
                except:
                    if output.shape==(10,):
                        print(output)
                    else:
                        for m in output:
                            draw_image(m)

        #calculate loss
        assert sum(output)<1+e and sum(output)>1-e
        prediction = list(output).index(max(output))
        correct = 1 if prediction == target else 0
        loss = -np.log(output[target])
        return output, correct, loss

    def backpropogate(self, input, target, debug=False):
        output, _, loss = self.forward(input, target, debug)
        grad = np.zeros(self.layers[-1].shape[1])
        grad[target] = -1/output[target]
        self.layers[-1].backpropogate(grad, self.learning_rate)
        """for layer in reversed(self.layers):
            grad = layer.backpropogate(grad, self.learning_rate)"""

    def train(self, train_data, show_progress=False):
        for train_index in range(len(train_data)):
            self.backpropogate(train_data[train_index][0], train_data[train_index][1])
            if show_progress:
                print("Training: {}/{}".format(train_index, len(train_data)))

    def test(self, test_data, debug=False):
        accuracy, avg_loss = 0, 0
        for test_index in range(len(test_data)):
            #print("{}/{}".format(test_index+1, len(test_data[0])))
            o, c, l = self.forward(test_data[test_index][0], test_data[test_index][1], debug)
            accuracy+=c
            avg_loss+=l

        avg_loss/=len(test_data)
        accuracy*=100/len(test_data)

        print("Loss: {}, accuracy: {}%".format(avg_loss, accuracy))
        return avg_loss, accuracy

from tensorflow.keras.datasets.mnist import load_data
mnist = load_data()
training, testing = [(i/254, l) for i, l in zip(mnist[0][0], mnist[0][1])], [(i/254, l) for i, l in zip(mnist[1][0], mnist[1][1])]
test_net = cnn([convolutionLayer(3, [3, 3]), poolingLayer("MAX", [2, 2]), Softmax([13*13*3, 10])], 0.5)

test_size = 100
#test_net.backpropogate(training[0][0], training[0][1], debug=True)
test_net.test(testing[:test_size])
test_net.train(training[:1000], show_progress=True)
test_net.test(testing[:test_size])
