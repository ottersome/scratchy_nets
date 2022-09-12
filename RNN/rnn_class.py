import numpy as np

class Model(object):
    def softmax(self):
        return np.softmax()

class RNN(model):
    def __init__(self,hidden_dim,input_dim,output_dim,sequence_len):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.output_dim = output_dim

        self.input_weights = np.random.uniform(-1,-1,size=[hidden_dim, hidden_dim])
        self.hidden_weights = np.random.uniform(-1,1,size=[hidden_dim, input_dim])

        self.bias = np.random.uniform(-1,1,size = [hidden_dim,1])

        self.output_bias  = np.random.uniform(-1,-1,size=[output_dim, 1])
        self.output_weights=np.random.uniform(-1,-1,size=[output_dim, hidden_dim])


    def forward(self,x): # X is a vector of input elemetns

        init_hidden = self.init_hidden()
        hiddens,outputs = self.rnn_multipass(init_hidden,x)

        return hidden,ouputs

    def rnn_multipass(self,init_hidden,x):
        # TODO we still need to somehow store stuff for backprop
        # This is sequential so we take one element at a time
        H = np.array([self.hidden_dim, self.sequence_len])
        O = np.array([self.hidden_dim, self.sequence_len])
        for i, xi in enumerate(x):
            H[:,i] = np.tanh(self.bias + self.hidden_weights@h_i+self.input_weights@xi)
            O[:,i] = self.output_bias + self.output_weights@H[:,i]
            # This output is a butt load of probabilities from which we need to pick the most likely word
            # TODO: I feel like this very likely wil take a loong time to train to randomly hit a good word. 
            
            # 
            
        return H,O

    def init_hidden(self):
        pass

class Trainer(object):
    def __init__(self,model):
        self.model = model

    def train_batch(self,x,y):
        pass

