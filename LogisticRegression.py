import numpy as np
from numpy import ndarray
from base_model import DataUtils

"""
    未调试
"""
class LogisticRegression:
    def __init__(self, n_input, n_output, i_iter, learn_rate):
        self.iter, self.learn_rate = i_iter, learn_rate
        self.in_dim, self.out_dim = n_input, n_output
        self.weights, self.bias = None, None
        self.weights_grad, self.bias_grad = None, None
        self.in_data, self.target = None, None
        self.init_model()

    def init_model(self):
        self.weights = np.random.random([self.in_dim, self.out_dim])
        self.bias = np.random.random([1, self.out_dim])

    def fit(self, in_data: ndarray, target: ndarray):
        self.in_data, self.target = in_data, target

    def cost_func(self, data: ndarray, target: ndarray):
        hypothesis_out = self.hypothesis(data)
        cost_out = -target*np.log(hypothesis_out)-(1-target)*np.log(1-hypothesis_out)
        self.weights_grad = np.matmul(self.in_data.T, hypothesis_out-target.reshape(hypothesis_out.shape))/data.shape[0]
        self.bias_grad = np.mean((hypothesis_out - target.reshape(hypothesis_out.shape)), 0)
        return np.mean(cost_out, 0)

    def hypothesis(self, data: ndarray)->ndarray:
        return 1/(1+np.exp(-data))

    def train(self):
        in_data = self.in_data
        cost = None
        for i in range(self.iter):
            linear_out = self.linear(in_data)
            cost = self.cost_func(linear_out, self.target)
            self.weights = self.weights - self.learn_rate*self.weights_grad
            self.bias = self.bias - self.learn_rate*self.bias_grad
        print("------ loss: "+str(np.mean(cost)))
        return

    def linear(self, in_data: ndarray):
        return np.matmul(in_data, self.weights)+self.bias

    def predict(self, in_data: ndarray):
        return self.linear(in_data)
        # return self.hypothesis(self.linear(in_data))

if __name__ == "__main__":
    orig = [[  1.79900000e+01,  1.03800000e+01,  1.22800000e+02,  1.00100000e+03,
    1.18400000e-01,   2.77600000e-01,   3.00100000e-01,   1.47100000e-01,
    2.41900000e-01,   7.87100000e-02,   1.09500000e+00,   9.05300000e-01,
    8.58900000e+00,   1.53400000e+02,   6.39900000e-03,   4.90400000e-02,
    5.37300000e-02,   1.58700000e-02,   3.00300000e-02 ,  6.19300000e-03,
    2.53800000e+01,   1.73300000e+01,   1.84600000e+02,   2.01900000e+03,
    1.62200000e-01,   6.65600000e-01,   7.11900000e-01,   2.65400000e-01,
    4.60100000e-01,   1.18900000e-01]]

    weight = [[ 0.40571405],
 [ 0.419992],
 [ 0.75075186],
 [ 0.97869078],
 [ 0.26055824],
 [ 0.33457899],
 [ 0.59087213],
 [ 0.3232564 ],
 [ 0.44236315],
 [ 0.98810268],
 [ 0.9542948 ],
 [ 0.11981115],
 [ 0.64138997],
 [ 0.6394366 ],
 [ 0.11550734],
 [ 0.15455986],
 [ 0.47423629],
 [ 0.27764346],
 [ 0.07820953],
 [ 0.50557285],
 [ 0.47270045],
 [ 0.9879579 ],
 [ 0.71624103],
 [ 0.42996838],
 [ 0.82318582],
 [ 0.86666844],
 [ 0.58590146],
 [ 0.13089389],
 [ 0.46720169],
 [ 0.87864388]]

    orig = np.array(orig)
    # weight = np.array(weight)
    #
    # print(np.matmul(DataUtils.normalize(orig), weight))
    d = orig.T
    print(orig.T)
    print(orig)