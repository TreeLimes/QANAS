
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import torch 
import itertools

#BY ASSUMPTION AT BOTTOM, WE CAN REDUCE EACH FILTER TO A 
#SMALLER FILTER

def layer_pca_to(layer, n_components):
    w = layer.state_dict()["weight"]
    X = []
    print(w.size())

    for i in range(w.size(0)):
        for j in range(w.size(1)):
            x = (w[i, j].view(-1).numpy())
            print(x.shape)
            X.append(x)

    X = np.array(X)
    pca = PCA(n_components=n_components*n_components)
    pca.fit(X)

    reduced_X = pca.transform(X)

    tensors_reduced_X = []

    for x in reduced_X:
        tensors_reduced_X.append(torch.Tensor(x))

    s = torch.stack(tensors_reduced_X).view(w.size(0), w.size(1), n_components, n_components )
    
    #UNSURE IF THIS IS ALL GOOD
    return s


#ASSUMING CNN, WITH STRIDE 1, PADDING SET SO INPUT SIZE IS SAME, AND
#SAME NUMBER OF FILTERS (will relax this idea later)

class rolling_arc(nn.Module):
    def __init__(self, base_arc_list):
        super().__init__()
        self.base_arc_list = nn.ModuleList(base_arc_list)
        self.linear_ind = len(base_arc_list) - 2

        self.optimizer = torch.optim.SGD(self.parameters(), lr=.001, momentum=0.9, weight_decay=1e-4, nesterov=True )

    def replace_conv_n_with(self, n, layer, PCA_intial=False):
        with torch.no_grad():
            if PCA_intial:
                reduced_w = layer_pca_to(self.base_arc_list[n],
                                         layer.kernel_size[0])
                layer.weight = nn.Parameter(reduced_w)
                #Only works this simply since same number of filters
                layer.bias = self.base_arc_list[n].bias

            #if out channels is not same then next layer needs more parameters
            if layer.out_channels is not self.base_arc_list[n].out_channels and n + 1 is not self.linear_ind:
                next_layer = self.base_arc_list[n+1]
                new_next_layer = nn.Conv2d(layer.out_channels, next_layer.out_channels, next_layer.kernel_size, next_layer.stride)
                #set params of new_next_layer based on old next_layer

                self.base_arc_list[n+1] = new_next_layer



            self.base_arc_list[n] = layer



            #fix linear layer
            n_classes = self.base_arc_list[self.linear_ind].out_features
            self.base_arc_list[self.linear_ind] = nn.Linear(last_output_size(self.base_arc_list[:self.linear_ind]), n_classes)
            #how to find good intialization of parameters using old linear layer

    def focus_optim_on_layers(self, ns):
        f_params = []

        #for i in range(len(self.base_arc_list)):
            #if i not in ns:
                #for param in self.base_arc_list[i].parameters():
                    #param.requires_grad = False
        for n in ns:
            f_params.append(self.base_arc_list[n].parameters())

        #forgets state of optimizer but probably no better alternative

        self.optimizer = torch.optim.SGD(itertools.chain(*f_params), lr=.001, momentum=0.9, weight_decay=1e-4, nesterov=True )


    def unfreeze_all(self):

        #for layer in self.base_arc_list:
            #for param in layer.parameters():
                #param.requires_grad = True


        #again forgets state but probably no better alternative
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.001, momentum=0.9, weight_decay=1e-4, nesterov=True )



    def forward(self, x):
        for l in self.base_arc_list[:self.linear_ind]:
                x = l(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.base_arc_list[self.linear_ind]
        x = self.base_arc_list[self.linear_ind+1]


        return x 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def last_output_size(mod_list, input_size=(32, 32)):

    W, H = input_size[0], input_size[1]
    num_out_channels = mod_list[len(mod_list)-1].out_channels
    for l in mod_list:
        if isinstance(l, nn.Conv2d):
            W = int((W-l.kernel_size[0] + 2)/l.stride[0]) + 1
            H = int((H-l.kernel_size[1] + 2)/l.stride[1]) + 1


    c_inp_size = W*H*num_out_channels

    return c_inp_size










