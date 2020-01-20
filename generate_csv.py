import cnn_ as c
import torch
import csv
import numpy as np 

def write_csv(name, arcs, accs):

    with open(name + ".csv", mode='w') as arc_writer:
        arc_writer = csv.writer(arc_writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        acc_names = ["architecture"]
        for i in range(len(accs[0])):
            acc_names.append("acc_"+str(i))

        arc_writer.writerow(acc_names)

        for arc, acc in zip(arcs,accs):
            arc_writer.writerow(arc + acc)

def draw_randomly_from_each(lists):
    draws = []
    for l in lists:
        draws.append(l[np.random.randint(0, len(l))])
    return draws

def train_random_nets(n_arcs=20, epochs=20, n_classes=4, batch_size=32,
                      k_hs=[1,3,5,7], k_ws=[1,3,5,7], 
                      s_hs=[1], s_ws=[1], 
                      n_fs=[24,36,48,64], n_ls=[2,4,6,8]):

    x_t = torch.load("data/x_train")[:1000]
    y_t = torch.load("data/y_train")[:1000]
    x_v = torch.load("data/x_val")[:500]
    y_v = torch.load("data/y_val")[:500]

    x_t = x_t[:int(int(len(x_t) / batch_size)*batch_size)]
    y_t = y_t[:int(int(len(x_t) / batch_size)*batch_size)]
    x_v = x_v[:int(int(len(x_t) / batch_size)*batch_size)]
    y_v = y_v[:int(int(len(x_t) / batch_size)*batch_size)]

    arcs = []
    accs = []

    for i in range(n_arcs):

        enc = []

        for j in range(n_ls[np.random.randint(0, len(n_ls))]):
            enc.append(draw_randomly_from_each([k_hs, k_ws, s_hs, s_ws, n_fs]) + [[None]])

        net = c.gen_cnn_from(enc, n_classes)

        accs.append(c.train_(net, x_t, x_v, y_t, y_v, epochs, 32))

    return arcs, accs