# QANAS
We note that NAS, as it stands in https://arxiv.org/pdf/1611.01578.pdf has an extremely high training time. To combat this, we use a prediction
performance mechanism as described here (https://openreview.net/pdf?id=BJypUGZ0Z). More so, policy gradient and action evaluation can benefit
from updating multiple times throughout an episode, in contrast to the full monte carlo esque method from the original paper. 

# Two Things
1. The original paper's implementation has no way to get the reward after every cycle in the controller without training a new architecture. Thus
we make use of the performance prediction here along with a "rolling architecture".
2. There is no action evaluation in the original paper. Thus we implement a simple feedforward network that takes the controller's hidden output
and the current action as input, and is trained every step along side the controller.

