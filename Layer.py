
import numpy as np

from images import test_rand, test_num
from helper import *

class Softmax():
    def __init__(self) -> None:
        pass

    def forward(self, input):
        self.last_input=input
        e = np.exp(input)
        self.out_activations = e/np.sum(e)
        return self.out_activations

    def backpropagate(self, gradient, lr):
        index_c = np.where(gradient!=0)[0][0]
        a = np.exp(self.last_input)
        s = np.sum(a)
        out_in = -a[index_c] * a / (s ** 2)
        out_in[index_c] = a[index_c] * (s - a[index_c]) / (s ** 2)
        return out_in * gradient[index_c]

class FullyConnected():
    def __init__(self, shape) -> None:
        self.shape=shape
        self.activations=[]
        self.weights=np.random.randn(shape[1], shape[0])
        self.biases=np.zeros(shape[1])

    def forward(self, input):
        f=input.flatten()
        self.last_input=f
        self.last_input_shape=input.shape
        self.activations=np.dot(self.weights, f)+self.biases
        return self.activations

    def backpropagate(self, gradient, lr):
        delta_w = self.last_input[np.newaxis].T @ gradient[np.newaxis] # d_L_d_w
        delta_b = gradient * 1

        new_grad= self.weights.T @ gradient

        self.weights -= lr * delta_w.T
        self.biases -= lr * delta_b

        return new_grad.reshape(self.last_input_shape)


class convolutionLayer():
    #Only 3x3 filters for now
    def __init__(self, num_kernels=3, kernel_shape=[3, 3], p=0, s=1) -> None:
        self.kernels = np.random.randn(num_kernels, kernel_shape[0], kernel_shape[1]) / 9
        self.kernel_shape=kernel_shape
        self.stride=s
        self.padding=p
        self.num_kernels=num_kernels

    def forward(self, input):
        #[(W−K+2P)/S]+1
        self.last_input=input
        output=np.zeros(shape=(self.num_kernels, 1+len(input[0])+2*self.padding-self.kernel_shape[0], 1+len(input)+2*self.padding-self.kernel_shape[1]))
        for kernel in enumerate(self.kernels):
            for row in range(0, len(input)-kernel[1].shape[1], self.stride):
                for col in range(0, len(input[row])-kernel[1].shape[0], self.stride): 
                    s=np.multiply([input[r][col:col+self.kernel_shape[0]] for r in range(row, row+self.kernel_shape[1])], kernel[1])
                    output[kernel[0]][row][col]=sum(map(sum, s))
        out_output=ReLu(output)
        return output

    def backpropagate(self, gradient, lr):
        deltas=np.zeros(shape=self.kernels.shape)
        for index, m in enumerate(gradient):
            for row in range(0, len(self.last_input)-1-m.shape[0], self.stride):
                for col in range(0, len(self.last_input[row])-1-m.shape[1], self.stride): 
                    sect = [self.last_input[r][col:col+m.shape[0]] for r in range(row, row+m.shape[1])]
                    s=np.multiply(sect, m)
                    deltas[index][row][col]=sum(map(sum, s))
        self.kernels -= lr * deltas

class poolingLayer():
    #no support for backpropogation of average pooling yet nor ever
    def __init__(self, type, receptiveField, k=None, p=0, s=2) -> None:
        self.kernel=k
        self.stride=s
        self.padding=p
        self.type=type
        self.rf=receptiveField
        #[l[i][r1:r2] for i in range(b1, b2)]
    def forward(self, input):
        try:
            type(input[0][0][0])
        except:
            raise dimensionError
        self.last_input=input
        output=np.zeros(shape=(len(input), int(len(input[0])/self.stride), int(len(input[0][0])/self.stride)))
        for i in enumerate(input):
            for row in range(0, len(i[1])-1, self.stride):
                for col in range(0, len(i[1][row])-1, self.stride):
                    if self.type=="MAX":   
                        output[i[0]][int(row/self.stride)][int(col/self.stride)]=np.amax([i[1][r][col:col+self.rf[0]] for r in range(row, row+self.rf[1])])
                    elif self.type=="AVG":
                        output[i[0]][int(row/self.stride)][int(col/self.stride)]=np.sum([i[1][r][col:col+self.rf[0]] for r in range(row, row+self.rf[1])])/self.stride**2
                    else:
                        print("Pooling type not specified. Use either MAX or AVG")
                        return
            #print(ret)
        return output

    def backpropagate(self, gradient, lr):
        d_L_d_in = np.zeros(shape=self.last_input.shape)
        for map_index in range(len(self.last_input)):
            for row in range(0, len(self.last_input[map_index])-1, self.stride):
                for col in range(0, len(self.last_input[map_index][row])-1, self.stride):
                    if self.last_input[map_index][row][col]==gradient[map_index][int(row/self.stride)][int(col/self.stride)]:
                        d_L_d_in[map_index][row][col]=gradient[map_index][int(row/self.stride)][int(col/self.stride)]
        return d_L_d_in

if __name__=="__main__":
    cl = convolutionLayer(3, [15, 15])
    f1 = cl.forward(test_num)
    """
    pl = poolingLayer("MAX", [2, 2])
    f2 = pl.forward(f1)
    
    fc = FullyConnected([13*13*3, 10])
    f3 = fc.forward(f2)

    s = Softmax()
    f4 = s.forward(f3)

    poowork = [cl, pl, fc, s]"""