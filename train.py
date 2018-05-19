import json
from utils import load_embedding,generate_embedding_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models.agg_model import generate_model
from utils import load_data_agg,make_one_hot,tokenizer_data


# load training data
train_path = 'data/data_resplit/train.jsonl'
nl_question,agg = load_data_agg(train_path)
# load val data
val_path = 'data/data_resplit/dev.jsonl'
dev_question,dev_agg = load_data_agg(val_path)

# tabel_path = '/Users/lvyufeng/PycharmProjects/keras_sequences/nl2sql_keras/data/data_resplit/tables.jsonl'

# sql_data = []
# table_data = {}
# with open(tabel_path) as inf:
#     for line in inf:
#         table = json.loads(line.strip())
#         table_data[table['id']] = table
# print(len(nl_question),len(agg))
# print(nl_question[4],agg[4])

# make label to one-hot
train_agg = make_one_hot(agg,7)
val_agg = make_one_hot(dev_agg,7)
# print(one_hot[4])

# tokenizer data
maxlen = 100
max_words = 50000
word_index,train_data = tokenizer_data(nl_question,max_words,maxlen)
word_index2,val_data = tokenizer_data(dev_question,max_words,maxlen)
# load pre-trained word embedding

glove_dir = 'pre_trained_embedding/glove.42B.300d.txt'
embedding_index = load_embedding(glove_dir)

# print(len(embedding_index))

# use pre-trained embedding matrix vector
embedding_dim = 300
embedding_matrix = generate_embedding_matrix(max_words,word_index,embedding_index,embedding_dim)

# define model

model = generate_model(max_words,embedding_dim,maxlen,embedding_matrix)

history = model.fit(train_data,train_agg,
                    epochs=100,
                    batch_size=32,
                    validation_data=(val_data,val_agg),
                    verbose=2,
                    validation_steps=32)

model.save('trained_agg_model.h5')

