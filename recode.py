import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

np.random.seed(666)
N = 200
nn_input_dim = 2
nn_output_dim = 2
lr = 0.005
reg_lambda = 0.01

X, y = sklearn.datasets.make_moons(n_samples=N, noise=0.1)


# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

def calculate_loss(model):
    W1, b1, W2, b2 = model.values()
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = np.exp(z2) / np.sum(exp_scores, axis=1, keepdims=True)  # softmax
    log_probs = -np.log(probs[range(N), y])  # cross entropy
    # probs[range(N), y]的取法要注意, 取第range个值第y个变量
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
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = np.exp(z2) / np.sum(exp_scores, axis=1, keepdims=True)
        delta3 = probs

        # 反向传播
        delta3[range(N), y] -= 1

        # 公式没明白
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)  # 不知道为什么要sum
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        W1 -= lr * dW1
        W2 -= lr * dW2
        b1 -= lr * db1
        b2 -= lr * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print(f"Loss after iteration{i}:{calculate_loss(model)}")

    return model


build_model(nn_hidden_dim=20, print_loss=True)
