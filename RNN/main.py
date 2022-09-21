import os
import glob
import numpy
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from rnn_class import RNN
import nltk.data

embeddings_dict = {}
training_samples = 100
batch_size = 100
num_batchers = 100


if __name__ == '__main__':
    
    # Load Embeddings
    with open('./GloveEmbeddings/glove.6B.50d.txt','r',encoding='utf-8') as f:
        for line in f.readline():
            # Split Line
            values = line.split()
            if len(values) == 0 : continue;
            word = values[0]
            vector = np.asarray(values[1:],'float32')
            embeddings_dict[word] = vector

    ## Get The Stories
    # Generate choice of stories at uniform
    sample_dir_name = os.path.dirname('./Corpus/dailymail/Generated/')
    list_of_files = glob.glob(sample_dir_name+"/*.cstory")
    print("Sample Dir Name",sample_dir_name)
    random_files = np.random.choice(list_of_files,size=training_samples,replace=False)
    
    # Load some Sentences
    network_input = []
    for file in random_files:
        sentence_splitter = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        fp = open(file)
        sentences = sentence_splitter.tokenize(fp.read())
        network_input.extend(sentences)# We still need to shuffle them
        fp.close()

    print("Showing 5 of the obtained sentences.")
    for choice in np.random.choice(network_input,size=5,replace=False):
        print(choice)


    vocabulary_size = len(embeddings_dict)
    # Create Model
    model = RNN(hidden_dim = vocabulary_size,# I dont know if this is necessarily the way but lets see.
            input_dim =  50,
            output_dim  =vocabulary_size,
            )

    for batchno in range(num_batchers):

        print("Working on batch number : ",batchno)
        sentences = np.random.choice(network_input,100,replace=False)

        for sentence in sentences:
            x = np.asarray([embeddings_dict[word] for word in sentence.split()])
            model.forward(x)
        # Do Gradient Descent

