from keras.layers import Embedding
import numpy as np


def generate_embedding_layer(glove_path):
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 1000

    # read embedding file
    embeddings_index = {}
    with open(glove_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embeddings_index[word] = coefs

    f.close()

    # generate embedding matrix
    embeddings_matrix = np.zeros((len(embeddings_index) + 1, EMBEDDING_DIM))

    values = embeddings_index.values()
    for index,value in enumerate(values):

    # for key,value in embeddings_index.items():
    #     print(key)
        embeddings_vector = value
        if embeddings_vector is not None:
            embeddings_matrix[index] = embeddings_vector


    embeddings_layer = Embedding(len(embeddings_index) + 1, EMBEDDING_DIM,
                                 weights=[embeddings_matrix],
                                 input_length = MAX_SEQUENCE_LENGTH,
                                 trainable=False)

    return embeddings_layer


glove_path = '../../glove/glove.42B.300d.txt'
generate_embedding_layer(glove_path)