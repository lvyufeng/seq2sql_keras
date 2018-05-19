from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense,LSTM,Bidirectional
from keras.activations import tanh,linear

def generate_model(max_words,embedding_dim,maxlen,embedding_matrix):
    model = Sequential()
    # load pretrained word embedding into embedding layer
    embedding_layer = Embedding(max_words,embedding_dim,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False)
    # embedding_layer.set_weights([embedding_matrix])
    # embedding_layer.trainable = False

    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(100, dropout=0.2)))
    # model.add(Bidirectional(LSTM(100, dropout=0.2)))
    model.add(Dense(100,activation='tanh'))
    model.add(Dense(7,activation='softmax'))
    # model.add(Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model