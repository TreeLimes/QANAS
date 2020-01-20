import numpy as np
import torch

def permute_data(x, y):
    rand_permute = np.random.permutation(np.arange(len(x)))

    return x[rand_permute], y[rand_permute]


def unpickle(file, n):
    import pickle

    X, Y = [], []

    D = []

    for i in range(n):
        with open(file + "/data_batch_" + str(i+1), 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
            X.append(d[b'data'])
            Y.append(d[b'labels'])

    X = np.concatenate(tuple(X), axis=0)
    Y = np.concatenate(tuple(Y), axis=0)


    return X, Y  # shape is (N, 3072)

def get_whitened_images(X):

    mean = X.mean(axis=0)


    X = X - X.mean(axis=0)

    X = X / np.sqrt((X ** 2).sum(axis=1))[:,None]

    # compute the covariance of the image data
    cov = np.cov(X, rowvar=True)   # cov is (N, N)
    # singular value decomposition
    U, S, V = np.linalg.svd(cov)     # U is (N, N), S is (N,)
    # build the ZCA matrix
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))

    print(zca_matrix.shape)
    # transform the image data       zca_matrix is (N,N)
    zca = np.dot(zca_matrix, X)    # zca is (N, 3072)

    return torch.Tensor(zca)

def preprocess_data(file, n, num_classes):
    X, Y = unpickle(file, n)

    X, Y = permute_data(X, Y)

    X_, Y_ = [], []

    for x, y in zip(X, Y):
        if y < num_classes:
            X_.append(x)
            Y_.append(y)

    X_ = np.asarray(X_)
    Y_ = torch.LongTensor(Y_)

    X_ = get_whitened_images(X_[:num_classes*5000])

    #put X_ back in (N, 3, 32, 32)
    X_ = torch.Tensor(X_.reshape((-1, 3, 32, 32)))


    X_train = X_[:(num_classes-1)*5000]

    Y_train = Y_[:(num_classes-1)*5000]

    X_val = X_[(num_classes-1)*5000:(num_classes)*5000]

    Y_val = Y_[(num_classes-1)*5000:(num_classes)*5000]

    return X_train, Y_train, X_val, Y_val

def save_and_preprocess():
    x_train,y_train,x_val,y_val = preprocess_data("cifar-10-batches", 5, 4)

    torch.save(x_train, "data/x_train")
    torch.save(y_train, "data/y_train")
    torch.save(x_val, "data/x_val")
    torch.save(y_val, "data/y_val")

