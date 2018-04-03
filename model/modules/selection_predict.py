from keras import layers
from keras import models
from model.util.parse_glove import generate_embedding_layer
from keras.layers import LSTM,Dense,Activation


# sequential model , adding layer one by one
model = models.Sequential()
# using pretrained glove embedding layer
model.add(generate_embedding_layer('../../glove/glove.42B.300d.txt'))
model.add(LSTM(128))

model.add(Dense(20))
model.add(Activation('softmax'))
# model

model.compile(optimizer='sgd',loss='crossentropy')