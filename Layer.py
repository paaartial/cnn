
import numpy as np

from images import test_rand, test_num
from helper import *

class dimensionError(Exception):
    #raised when an input has the wrong number of dimensions
    pass

class Softmax():
    def __init__(self, shape) -> None:
        self.shape=shape
        self.in_activations=[]
        self.out_activations=[]
        self.weights=np.random.randn(shape[1], shape[0])
        self.biases=np.zeros(shape[1])

    def forward(self, input):
        f=input.flatten()
        self.last_input=f
        self.last_input_shape=input.shape
        self.in_activations=np.dot(self.weights, f)+self.biases
        e = np.exp(self.in_activations)
        self.out_activations = e/np.sum(e)
        return self.out_activations

    def backpropogate(self, gradient, lr):
            index = np.where(gradient==np.amax(gradient))[0][0]
            a = np.exp(self.in_activations)
            s = np.sum(a)
            out_in = -a[index] * a / (s ** 2)
            out_in[index] = a[index] * (s - a[index]) / (s ** 2)

            d_in_d_w = self.last_input
            d_in_d_b = 1
            d_in_d_i = self.weights

            d_L_d_in = gradient[index] * out_in

            nabla_weights = d_in_d_w[np.newaxis].T @ d_L_d_in[np.newaxis]
            nabla_biases = d_L_d_in * d_in_d_b

            d_L_d_i= d_in_d_i.T @ d_L_d_in

            self.weights -= lr * nabla_weights.T
            self.biases -= lr * nabla_biases

            return d_L_d_i.reshape(self.last_input_shape)




class convolutionLayer():
    #Only 3x3 filters for now
    def __init__(self, num_kernels=0, kernel_shape=[3, 3], k=[], p=0, s=1) -> None:
        if k==[]:
            self.kernels = np.random.randn(num_kernels, kernel_shape[0], kernel_shape[1]) / 9
        else:
            self.kernels=k
        self.kernel_shape=kernel_shape
        self.stride=s
        self.padding=p
        self.num_kernels=num_kernels

    def forward(self, input):
        #[(W−K+2P)/S]+1
        output=np.zeros(shape=(self.num_kernels, 1+len(input[0])+2*self.padding-self.kernel_shape[0], 1+len(input)+2*self.padding-self.kernel_shape[1]))
        for kernel in enumerate(self.kernels):
            for row in range(0, len(input)-1-self.stride, self.stride):
                for col in range(0, len(input[row])-1-self.stride, self.stride): 
                    s=np.multiply([input[r][col:col+self.kernel_shape[0]] for r in range(row, row+self.kernel_shape[1])], kernel[1])
                    output[kernel[0]][row][col]=sum(map(sum, s))
        return output

    def backpropogate(self, gradient):
        pass

class poolingLayer():
    #no support for backpropogation of average pooling yet
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

    def backpropogate(self, gradient):
        pass



l=[[[5, 4, 6, 4, 5, 2],
   [0, 1, 3, 7, 1, 0]], [[5, 4, 6, 4, 5, 2],
   [0, 1, 3, 7, 1, 0]], [[5, 4, 6, 4, 5, 2],
   [0, 1, 3, 7, 1, 0]]]
#input: 6x4
#kernel 2x2, stride 2, output: 3x2
#input0/2, input1/2
#kernel 2x2, stride 1, output: 5x3
#input - kernel / stride
#mnist_train, mnist_test = (mnist[0][0], mnist[0][1]), (mnist[1][0], mnist[1][1])
#kernel=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
tkl=convolutionLayer(num_kernels=5, s=1)
tpl=poolingLayer("MAX", [2, 2], s=2)
tsl=Softmax([845, 10])
im = [[i/255 for i in j] for j in test_num]

im = tkl.forward(im)
im = tpl.forward(im)
im = tsl.forward(im)

#print(im)

