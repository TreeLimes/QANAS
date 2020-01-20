import rpi
import cnn_
import torch
import numpy as np 

def train_and_save(path, n, epochs, k_hs=[3, 5, 7], k_ws=[3, 5, 7], s_hs=[1], s_ws=[1], fs=[12, 24, 36], ls=[4, 6, 8]):
    

    cnns = []

    x_train, x_val, y_train, y_val = torch.load("data/x_train"), torch.load("data/x_val"), torch.load("data/y_train"), torch.load("data/y_val")

    for i in range(n):
        enc = []
        num_layers = ls[np.random.randint(0, len(ls))]

        for j in range(num_layers):
            params = []
            params.append(k_hs[np.random.randint(0, len(k_hs))])
            params.append(k_ws[np.random.randint(0, len(k_ws))])
            params.append(s_hs[np.random.randint(0, len(s_hs))])
            params.append(s_ws[np.random.randint(0, len(s_ws))])
            params.append(fs[np.random.randint(0, len(fs))])
            params.append([None])
            enc.append(params)

        cnns.append(cnn_.gen_cnn_from(enc, n_classes=4))

    trained_cnns = []
    val_accs = []
    for cnn in cnns:
        t_c, v_a = cnn_.train_(cnn, x_train, x_val, y_train, y_val, epochs=epochs, batch_size=32)
        trained_cnns.append(t_c)
        val_accs.append(v_a)

    torch.save({"models": trained_cnns, "accs": val_accs} ,"data/saved_models.pt")




