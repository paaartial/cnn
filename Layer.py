import numpy as np

from images import test_rand, test_num
from helper import *

def convolute(input, filter, stride=1, padding=0):
    out=np.zeros(shape=(1+len(input[0])+2*padding-filter.shape[0], 1+len(input)+2*padding-filter.shape[1]))
    input = np.pad(input, padding)
    for row in range(0, len(input)-filter.shape[1], stride):
        for col in range(0, len(input[row])-filter.shape[0], stride): 
            s=np.multiply([input[r][col:col+filter.shape[0]] for r in range(row, row+filter.shape[1])], filter)
            out[row][col]=sum(map(sum, s))
    return out

class ReLu():
    def __init__(self) -> None:
        pass

    def forward(self, input):
        self.last_input = input
        return np.maximum(input, 0)

    def backpropagate(self, gradient, lr):
        return np.greater(self.last_input, 0) * gradient 

    def get(self):
        return ("ReLu", None)


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
    
    def get(self):
        return ("softmax", None)

class FullyConnected():
    def __init__(self, s=None, w=[], b=[]) -> None:
        if s==None:
            self.shape=w.shape
        else:
            self.shape=s
        if w==[]:
            #limit = np.sqrt(3.0 * (1/max(1., sum(self.shape)/2.)))
            #limit = np.sqrt(6/sum(self.shape))
            limit = 2/sum(self.shape)
            self.weights = np.random.uniform(low=-limit, high=limit, size=(self.shape[1], self.shape[0]))
        else:
            self.weights=w
        if b==[]:
            self.biases=np.zeros(self.shape[1])
        else:
            self.biases=b
        self.activations=[]

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

    def get(self):
        return ("fully connected", self.weights.tolist(), self.biases.tolist())


class convolutionLayer():
    #Only 3x3 filters for now
    def __init__(self, num_kernels=3, kernel_shape=[3, 3], k=[], p=0, s=1) -> None:
        self.stride=s
        self.padding=p
        if k==[]:
            self.kernels = np.random.randn(num_kernels, kernel_shape[0], kernel_shape[1]) / 9
            self.kernel_shape=kernel_shape
            self.num_kernels=num_kernels
        else:
            self.kernels=k
            self.kernel_shape=k[0].shape
            self.num_kernels=len(k)

    def forward(self, input):
        #[(W−K+2P)/S]+1
        self.last_input=input
        return np.array([convolute(input, kernel, self.stride, self.padding) for kernel in self.kernels])

    def backpropagate(self, gradient, lr):
        deltas=np.zeros(shape=self.kernels.shape)
        next_error = []
        #FULL PADDING = PAD(LEN(KERNEL)-1)
        for index, m in enumerate(gradient):
            deltas[index] = convolute(self.last_input, m, self.stride, self.padding)
            """rotated_kernel=np.rot90(self.kernels[index], 2)
            error_at_layer = convolute(m, rotated_kernel, padding=len(rotated_kernel)-1)
            #print(m)
            if next_error==[]:
                next_error=error_at_layer
            else:
                next_error+=error_at_layer"""
            
        self.kernels -= lr * deltas

        return next_error

    def get(self):
        return ("convolution", self.kernels.tolist(), self.stride)

class poolingLayer():
    #no support for backpropogation of average pooling yet nor ever
    def __init__(self, type, receptiveField, k=None, p=0, s=2) -> None:
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
                    sect = [self.last_input[map_index][r][col:col+self.rf[0]] for r in range(row, row+self.rf[1])]
                    if self.last_input[map_index][row][col]==np.amax(sect):
                        d_L_d_in[map_index][row][col]=gradient[map_index][int(row/self.stride)][int(col/self.stride)]
        return d_L_d_in

    def get(self):
        return ("pooling", self.type, self.rf)

if __name__=="__main__":
    cl = convolutionLayer(3, [15, 15])
    f1 = cl.forward(test_num)
