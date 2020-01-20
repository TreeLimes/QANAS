import performance_prediction as pp
import cnn_
import controller_
import torch.nn as nn
import torch 
import rpi


PP_EPOCHS = 5
ITERS = 5
NUM_LAYERS = 2
NUM_ARC_PARAMS_PER_LAYER = 5
APLHA = .01
BETA = .01


class value_estimator(nn.Module):
    def __init__(self, controller_hidden_size=35):
        super().__init__()
        #36 or 39 depending on wether action is represented as single number or vector of 1 hot
        self.lin1 = nn.Linear(controller_hidden_size+1, 16)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(16, 1)
        self.tanh = nn.Tanh()

        self.optim = torch.optim.SGD(self.parameters(), lr=BETA, momentum=0.9)



    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.tanh(x)

        return x


def main():

    pp_model, score = pp.get_pp_model("data/arc_accs.csv", 
                                      t_arcs=15, v_arcs=5,
                                      epochs=PP_EPOCHS, T=20)

    rolling_arc = rpi.rolling_arc()
    controller = controller_.controller(num_classes=4, hidden_size=35,
                                        num_layers=NUM_LAYERS,
                                        num_arc_params_per_layer=NUM_ARC_PARAMS_PER_LAYER)

    ve = value_estimator()

    
    num_steps = ITERS*NUM_LAYERS*NUM_ARC_PARAMS_PER_LAYER

    action, log_prob, h, _ = controller.step()

    last_v = ve(torch.cat((action, h), 0))

    for i in range(num_steps):
        controller.optim.zero_grad()
        rolling_arc.in_sequence_transform(action)
        pp_features = rolling_arc.train()
        #SHOULd not just return prediction. SHould see 
        r = pp_model.predict(pp_features)

        o_action, o_log_prob, o_h = action, log_prob, h
        action, log_prob, h, _ = controller.step()
        new_v = ve(torch.cat((action, h), 0))

        #delta should not have its gradient taken, ALSO SHOULD CHECK IF AT END STATE
        with torch.no_grad():
            delta = r + new_v - last_v
            last_v = new_v

        ve_loss = delta*ve(torch.cat((o_action, o_h), 0))
        ve_loss.backward()
        ve.optim.step()

        loss = delta*o_log_prob
        loss.backward()
        controller.optim.step()


