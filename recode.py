import numpy as np
import matplotlib.pyplot as plt
import sklearn

np.random.seed(666)
N = 200
nn_input_dim = 2
nn_output_dim = 2
lr = 0.1
reg_lambda = 0.01

X, y = sklearn.datasets.make_moons(n_samples=N, noise=0.1)


# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

def calculate_loss(model):
    W1, b1, W2, b2 = model.values()
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W1) + b2
    exp_scores = np.exp(z2)
    probs = np.exp(z2) / np.sum(exp_scores, axis=1, keepdims=True)
    log_probs = -np.log(probs[:, y])
    sum_loss = np.sum(log_probs)
    return sum_loss / N


def build_model(nn_hidden_dim, num_passes=10000, print_loss=True):
    W1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)  # 调整方差
    b1 = np.zeros((1, nn_hidden_dim))
    W2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)  # 调整方差
    b2 = np.zeros((1, nn_output_dim))

    model = {}
    for i in range(num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W1) + b2
        exp_scores = np.exp(z2)
        probs = np.exp(z2) / np.sum(exp_scores, axis=1, keepdims=True)
        delta3 = probs

        # 反向传播
        delta3[range(N), y] = delta3[range(N), y]

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return model
