import numpy as py 


class Optimzer(object):

    def __init__(self,model):
        self.model = model
        pass

    def log_loss(self, output, truth):
        # Get which of these outputs
        boolean_mask = (output == truth)
        log_probs = np.sum()
        








