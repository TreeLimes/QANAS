# QANAS
We seek to increase NAS's controller's generality by feeding it input from varying but similar tasks, so that the controller is "pre-trained" to quickly adapt to generating good architectures for tasks related but unseen during training. However, this would most certainly increase the already extremely high training time. https://arxiv.org/pdf/1611.01578.pdf

To combat this, we use a prediction
performance mechanism as described here (https://openreview.net/pdf?id=BJypUGZ0Z). More so, policy gradient and action evaluation can benefit
from updating multiple times throughout an episode, in contrast to the full monte carlo esque method from the original paper. So, in hopes 
to decrease training time we do the following:

1. The original paper's implementation has no way to get the reward after every cycle in the controller without training a new architecture. Thus
we make use of the performance prediction here along with a "rolling architecture".
2. There is no action evaluation in the original paper. Thus we implement a simple feedforward network that takes the controller's hidden output
and the current action as input, and is trained every step along side the controller.


