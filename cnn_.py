import torch
import torch.nn as nn
import numpy as np

def permute_data(x, y):
    rand_permute = np.random.permutation(np.arange(len(x)))

    return torch.Tensor(np.asarray(x)[rand_permute]), torch.LongTensor(np.asarray(y)[rand_permute])


def conv_block(in_c, num_f, k_h, k_w, s_h, s_w):
    return nn.Sequential(
        nn.Conv2d(in_c, num_f, (k_h, k_w), (s_h, s_w), padding=1),
        nn.BatchNorm2d(num_f),
        nn.ReLU()
    )


class CNN_(nn.Module):
    def __init__(self, layers, classifier_inp_size, skip_connections):
        super().__init__()

        self.classifier_inp_size = classifier_inp_size
        self.skip_connections = skip_connections
        conv_layers = layers[:len(layers)-2]

        self.conv_list = nn.ModuleList(conv_layers)

        self.classifier = nn.Sequential(*layers[len(layers)-2:])

    def forward(self, x):
        for layer in self.conv_list:
            x = layer(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.classifier(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def gen_cnn_from(enc, n_classes):

    layers = []

    prev_num_f = 3

    skip_connections = []

    W, H = 32, 32
    for l in enc:
        #print("W: ", W, " H: ", H, " prev_num_f: ", prev_num_f)
        k_h = l[0]
        k_w = l[1]
        s_h = l[2]
        s_w = l[3]
        num_f = l[4]
        skip_connections.append(l[5])

        W = int((W-k_w + 2)/s_w) + 1
        H = int((H-k_h + 2)/s_h) + 1

        layers.append(nn.Conv2d(prev_num_f, num_f, (k_h, k_w), (s_h, s_w), padding=1))
        layers.append(nn.BatchNorm2d(num_f))
        layers.append(nn.ReLU())

        prev_num_f = num_f

    c_inp_size = W*H*prev_num_f
    #print("W: ", W, " H: ", H, " prev_num_f: ", prev_num_f)

    layers.append(nn.Linear(c_inp_size, n_classes))
    layers.append(nn.Softmax())

    net = CNN_(layers, c_inp_size, skip_connections)

    return net


def train_(net, x_train, x_val, y_train, y_val, epochs, batch_size):

    train_its = int(float(len(x_train)) / float(batch_size))

    val_accs = []

    optimizer = torch.optim.SGD(net.parameters(), lr=.001, momentum=0.9, weight_decay=1e-4, nesterov=True )
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()

        x_train, y_train = permute_data(x_train, y_train)

        train_loss = 0

        for i in range(train_its):

            x_batch, y_batch = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: ", epoch, "Avg Train loss: ", train_loss / train_its)

        val_its = int(float(len(x_val)) / float(batch_size))

        with torch.no_grad():
            net.eval()
            num_correct = 0

            for i in range(val_its):
                optimizer.zero_grad()

                x_batch, y_batch = x_val[i*batch_size:(i+1)*batch_size], y_val[i*batch_size:(i+1)*batch_size]

                out = net(x_batch)

                _, pred = out.max(1)

                for p, t in zip(pred, y_batch):
                    if p == t: num_correct += 1


            val_accs.append(float(num_correct) / float(len(x_val)))

    return net, val_accs




