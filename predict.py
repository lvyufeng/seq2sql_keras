from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nl2sql_keras.models.agg_model import generate_model
from nl2sql_keras.utils import load_embedding,generate_embedding_matrix
import json

sql_path = '/Users/lvyufeng/PycharmProjects/keras_sequences/nl2sql_keras/data/data_resplit/test.jsonl'

nl_question = []
agg = []
max_col_num = 0

with open(sql_path) as inf:
    for index,line in enumerate(inf):
        sql = json.loads(line.strip())
        # sql_data.append(sql)
        nl_question.append(sql['question'])
        agg.append(str(sql['sql']['agg']))
# with open(tabel_path) as inf:
#     for line in inf:
#         table = json.loads(line.strip())
#         table_data[table['id']] = table
print(len(nl_question),len(agg))
print(nl_question[4],agg[4])

# make label to one-hot
agg_tokenizer = Tokenizer(num_words=7)
agg_tokenizer.fit_on_texts(agg)
one_hot = agg_tokenizer.texts_to_matrix(agg,mode='binary')
print(agg_tokenizer.word_index)
print(one_hot.shape)
print(one_hot[4])


glove_dir = '/Users/lvyufeng/PycharmProjects/keras_sequences/nl2sql_keras/pre_trained_embedding/glove.42B.300d.txt'

embedding_index = load_embedding(glove_dir)

# print(len(embedding_index))

# tokenizer data
# nl_question = ['What is the total of win percentages when the year is 2008 and the crew is varsity 8+?','Name the number of average for steals per game']
maxlen = 100
max_words = 50000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(nl_question)
sequences = tokenizer.texts_to_sequences(nl_question)

word_index = tokenizer.word_index

print('find token: ', len(word_index))

data = pad_sequences(sequences,maxlen=maxlen)

embedding_dim = 300
embedding_matrix = generate_embedding_matrix(max_words,word_index,embedding_index,embedding_dim)

model = generate_model(max_words,embedding_dim,maxlen,embedding_matrix)
model.load_weights('/Users/lvyufeng/PycharmProjects/keras_sequences/nl2sql_keras/trained_agg_model.h5')
# cost = model.evaluate(x_test,y_test)
# print('test cost: ',cost)

agg = {'0': 1, '3': 2, '1': 3, '2': 4, '5': 5, '4': 6}

cost = model.evaluate(data,one_hot)
print('cost:',cost)

Y_pred = model.predict(data)
y_pred = Y_pred.tolist()
print(type(Y_pred))
print(y_pred)
print(y_pred[0].index(max(Y_pred[0])))