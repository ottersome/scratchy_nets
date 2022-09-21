import numpy as np

def tanh_f(object):
    return np.tanh(alpha)
def tanh_b(self,x):
    return 1-self.f(x)**2

class LogLossSoftmax_f(object):
    def f(self,true_vals, logbits):
        predictions = np.softmax(logbits)
        final_sum = -np.sum(
                np.multiply(true_vals,np.log(predictions))
                )
        return final_sum

def LogLossSoftmax_b(self,predictions,ground_truth,inputf='softmax'):
    return predictions-ground_truth
    

# class LogBits_f():
    # return self.U + self.V@hidden
# def LogBits_b(self):
    #dfdu = np.ones(self.U.shape)#TODO i think this is okay

