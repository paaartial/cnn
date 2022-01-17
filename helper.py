import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class dimensionError(Exception):
    #raised when an input has the wrong number of dimensions
    pass

def draw_image(img_to_draw):
    if len(img_to_draw.shape)==2:
        plt.imshow(img_to_draw, cmap = mpl.cm.binary, interpolation="nearest")
        plt.axis("off")
        plt.show()
    elif len(img_to_draw.shape)==3:
        for m in img_to_draw:
            plt.imshow(m, cmap = mpl.cm.binary, interpolation="nearest")
            plt.axis("off")
            plt.show()
    else: 
        raise dimensionError

def draw_arr(arr_to_draw):
    prnt=""
    for r in arr_to_draw:
        for c in r:
            if c == 0:
                prnt +="  "
            else:
                prnt +=str(c)
            prnt+=" "
        prnt+="\n"
    print(prnt)

def sigmoid(xl, deriv=False):
    if not deriv:
        return [1/(1 + np.exp(-x)) for x in xl]
    return [sx * (1-sx) for sx in sigmoid(xl)]

def split_train_test(train, test, train_size, test_size):
    shuffled_train_indices = np.random.permutation(len(train[0]))[:train_size]
    shuffled_test_indices = np.random.permutation(len(test[0]))[:test_size]
    to_train_pairs = [(train[0][shuffled_index], train[1][shuffled_index]) for shuffled_index in range(len(shuffled_train_indices))]
    to_test_pairs = [(test[0][shuffled_index], test[1][shuffled_index]) for shuffled_index in range(len(shuffled_test_indices))]
    return (to_train_pairs, to_test_pairs)

def ReLu(xl, deriv=False):
    if not deriv:
        return np.maximum(xl, 0)
    return np.max(xl, 0)/xl