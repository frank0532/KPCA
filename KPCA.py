from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class KPCA():
    def __init__(self, kernel='rbf'):
        self.kernel = kernel
    def fit_transform_plot(self, X, y):
        if self.kernel == 'None':
            C = np.cov(X.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            arg_max = eigvals.argsort()[-2:]
            eigvecs_max = eigvecs[:, arg_max]
            K = X
        else:
            if self.kernel == 'linear':
                K = np.dot(X, X.T)
            elif self.kernel == 'log':
                dists = pdist(X) ** 0.2
                mat = squareform(dists)
                K = -np.log(1 + mat)
            elif self.kernel == 'rbf':
                dists = pdist(X) ** 2
                mat = squareform(dists)
                beta = 10
                K = np.exp(-beta * mat)
            else:
                print('kernel error!')
                return None
            N = K.shape[0]
            one_n = np.ones([N, N]) / N
            K_hat = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
            eigvals, eigvecs = np.linalg.eigh(K_hat)
            arg_max = eigvals.argsort()[-2:]
            eigvecs_max = eigvecs[:,arg_max]
        X_new = np.dot(K, eigvecs_max)
        for i in range(2):
            tmp = y == i
            Xi = X_new[tmp]
            plt.scatter(Xi[:,0], Xi[:,1])
        plt.show()

if __name__ == '__main__':
    X, y = make_circles(n_samples=400, factor=.3, noise=.05, random_state=0)
    kpca = KPCA('rbf')
    kpca.fit_transform_plot(X, y)
