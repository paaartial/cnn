from Layer import *
from images import test_normal
import numpy as np
import json
import time

e=0.0000001

class cnn():
    def __init__(self, n, l, lr):
        self.layers=l
        self.learning_rate=lr
        self.name=n
    
    def forward(self, input, target, debug=False):
        output=input
        for layer in self.layers:
            if debug:
                try:
                    draw_image(output)
                except:
                    if output.shape==(10,):
                        print(output)
                    else:
                        for m in output:
                            draw_image(m)
            output=layer.forward(output)

        #calculate loss
        assert sum(output)<1+e and sum(output)>1-e
        prediction = list(output).index(max(output))
        correct = 1 if prediction == target else 0
        loss = -np.log(output[target])
        return output, correct, loss, prediction

    def backpropagate(self, input, target, debug=False):
        output, _, loss, p = self.forward(input, target, debug)
        grad = np.zeros(self.layers[-2].shape[1])
        grad[target] = -1/output[target]
        for layer in reversed(self.layers):
            #print(grad)
            grad = layer.backpropagate(grad, self.learning_rate)

    def train(self, train_data, show_progress=False):
        time_start = time.time()
        for train_index in range(len(train_data)):
            self.backpropagate(train_data[train_index][0], train_data[train_index][1])
            if show_progress:
                print("Training: {}/{}".format(train_index, len(train_data)))
        time_taken = time.time()-time_start
        print("{} minutes, {} seconds".format(time_taken//60, time_taken%60))

    def test(self, test_data, debug=False):
        accuracy, avg_loss = 0, 0
        for test_index in range(len(test_data)):
            #print("{}/{}".format(test_index+1, len(test_data[0])))
            o, c, l, p = self.forward(test_data[test_index][0], test_data[test_index][1], debug)
            accuracy+=c
            avg_loss+=l

        avg_loss/=len(test_data)
        accuracy*=100/len(test_data)

        print("Loss: {}, accuracy: {}%".format(avg_loss, accuracy))
        return avg_loss, accuracy

    def serialize(self):
        with open(self.name + ".json", "w") as outfile:
            to_save={}
            to_save["layer data"] = [l.get() for l in self.layers]
            to_save["learning rate"] = self.learning_rate
            json.dump(to_save, outfile)
            outfile.close()

    def visual_test(self, to_test):
        for test_image in to_test:
            _, _, _, p = self.forward(test_image[0], test_image[1])
            draw_image(test_image[0], p)

    def graph_loss(self, to_test):
        xs=[]
        ys=[]

def load_network(name):
    with open(name + ".json", 'r') as openfile:
        reader = json.load(openfile)
        layers=[]
        layer_data = reader["layer data"]
        for ld in layer_data:
            if ld[0] == "fully connected":
                l=FullyConnected(w=np.array(ld[1]), b=np.array(ld[2]))
            if ld[0] == "convolution":
                l=convolutionLayer(k=np.array(ld[1]), s=ld[2])
            if ld[0] == "softmax":
                l=Softmax()
            if ld[0] == "pooling":
                l=poolingLayer(ld[1], ld[2])
            if ld[0] == "ReLu":
                l=ReLu()
            layers.append(l)
        loaded = cnn(name, layers, reader["learning rate"])
        return loaded

from tensorflow.keras.datasets.mnist import load_data
mnist = load_data()
test, train = [(i/254, l) for i, l in zip(mnist[0][0], mnist[0][1])], [(i/254, l) for i, l in zip(mnist[1][0], mnist[1][1])]
test_size = 1000
train_size=10000
training, testing = split_train_test(train, test, train_size, test_size)
test_net=[]
acc=1

if __name__ == "__main__":
    test_size = 1000
    num_filters=5
    #cnn("test", [convolutionLayer(num_filters, [3, 3]), poolingLayer("MAX", [2, 2]), ReLu(), FullyConnected(s=[13*13*num_filters, 10]), Softmax()], 0.06)
    #cnn("mlp", [FullyConnected([28*28, 100]), ReLu(), FullyConnected(s=[100, 10]), Softmax()], 0.05)
    test_net = cnn("test", [convolutionLayer(num_filters, [5, 5]), poolingLayer("MAX", [2, 2]), ReLu(), FullyConnected(s=[12*12*num_filters, 10]), Softmax()], 0.05)
    test_net.train(training[:10000], show_progress=True)
    _, acc = test_net.test(training[:test_size])
    if acc>90:
        test_net.serialize()

    #conv94p=load_network("5x5conv94p")
#test_net.backpropagate(training[0][0], training[0][1], debug=True)