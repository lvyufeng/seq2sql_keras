import numpy as np
import os
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_embedding(glove_dir):
    # glove_dir = '/Users/lvyufeng/PycharmProjects/keras_sequences/glove.6B'
    embedding_index = {}

    f = open(glove_dir)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print(len(embedding_index))
    return embedding_index

def load_data_agg(data_dir):
    nl_question = []
    agg = []
    max_col_num = 0

    with open(data_dir) as inf:
        for index, line in enumerate(inf):
            sql = json.loads(line.strip())
            # sql_data.append(sql)
            nl_question.append(sql['question'])
            agg.append(str(sql['sql']['agg']))

    return nl_question,agg

def generate_embedding_matrix(max_words,word_index,embedding_index,embedding_dim):

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)
    return embedding_matrix

def make_one_hot(agg,num_words):
    agg_tokenizer = Tokenizer(num_words=num_words)
    agg_tokenizer.fit_on_texts(agg)
    one_hot = agg_tokenizer.texts_to_matrix(agg, mode='binary')
    print(agg_tokenizer.word_index)
    print(one_hot.shape)
    return one_hot

def tokenizer_data(nl_question,max_words,maxlen):

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(nl_question)
    sequences = tokenizer.texts_to_sequences(nl_question)
    word_index = tokenizer.word_index
    print('find token: ', len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen)
    print(data.shape)

    return word_index,data