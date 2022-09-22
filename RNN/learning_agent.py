import numpy as py 
import functions as funcs


class Optimzer(object):

    def __init__(self,model):
        self.model = model
        pass

    def log_loss(self, output, truth):
        # Get which of these outputs
        boolean_mask = (output == truth)
        log_probs = np.sum()

    def BPTT_t(self,ground_truths, predictions, big_sum,hidden_states):
        # We get a single run of the whole thing here
        dLdV = []; dLdU = []; dLdW= []

        # The Favorite sun
        dLdZ = LogLossSoftmax_b(predictions,ground_truths)
        dZdV = [np.vstack(hidden_states,self.model.oputput_dim) for i in range(hidden_states.shape[1])]

        dLdV = dLdZ @ d

        dLdH = np.sum(dLdZ,axis=1)
        dLdV = np.sum(dLdZ,axis=1)
        dLdU = np.sum(dLdZ,axis=1)

        return dLdV. dLdU, dldW






        








