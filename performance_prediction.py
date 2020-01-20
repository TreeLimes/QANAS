import csv
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np


def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def permute_data(x, y):
    rand_permute = np.random.permutation(np.arange(len(x)))

    return x[rand_permute], y[rand_permute]


def get_data(reader, n, T):
    #first dimension arc, second is epoch
    y = []
    y_prime = []
    y_double_prime = []
    y_T = []

    header = reader.__next__()
    acc_index = header.index('acc_1')

    for row in reader:

        y_T.append(int_or_float(row[acc_index + T - 1]))

        row_accs = [int_or_float(x) for x in row[acc_index:acc_index+n+2]]
        y.append(row_accs[0:n])

        y_prime_ele = []

        for i in range(len(row_accs)-1):
            y_prime_ele.append(row_accs[i+1] - row_accs[i])

        y_prime.append(y_prime_ele[0:n])

        y_double_prime_ele = []
        for i in range(len(y_prime_ele)-1):
            y_double_prime_ele.append(y_prime_ele[i+1] - y_prime_ele[i])
        y_double_prime.append(y_double_prime_ele)

    features = []
    for i in range(len(y)):
        features.append(y[i] + y_prime[i] + y_double_prime[i])

    return features, y_T


def get_pp_model(data_name, t_arcs, v_arcs, epochs, T):

    features, y_T = [], []

    with open(data_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        features, y_T = get_data(reader, epochs, T)

    features = np.asarray(features)
    y_T = np.asarray(y_T)

    features, y_T = permute_data(features, y_T)

    train_features = features[0:t_arcs]
    train_y_T = y_T[0:t_arcs]
    val_features = features[t_arcs+1:t_arcs+1+v_arcs]
    val_y_T = y_T[t_arcs+1:t_arcs+1+v_arcs]

    clf = SVR(gamma='scale')

    grid_param = {
        'degree': [0, 1, 2, 3, 4, 5],
        'C': [.2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'epsilon': [.1, .01, .001]
    }

    gd_sr = GridSearchCV(estimator=clf,
                         param_grid=grid_param,
                         scoring="r2",
                         cv=3,
                         n_jobs=-1)
    gd_sr.fit(train_features, train_y_T)

    score = gd_sr.best_estimator_.score(val_features, val_y_T)

    return gd_sr.best_estimator_, score










