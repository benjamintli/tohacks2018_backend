import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.utils as util
import keras.preprocessing.text as kt
import h5py
import json


#index the sentiment and the tweet
training = np.genfromtxt('training_data/processed.csv',
                         delimiter=',',
                         skip_header=1,
                         usecols=(0, 1),
                         dtype=None)

train_x = [str(x[1]) for x in training]
train_y = np.asarray([x[0] for x in training])

#preprocess the data by creating a 'dictionary' that has indexed words

tokenizer = kt.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_x)

dictionary = tokenizer.word_index
with open('training_data/dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    return[dictionary[word] for word in kt.text_to_word_sequence(text)]


allWordIndices = []
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)


allWordIndices = np.asarray(allWordIndices)

train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')

train_y = util.to_categorical(train_y, 2)


#now we can make the model

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()


model.fit(train_x, train_y,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

model_json = model.to_json()
with open('training_data/model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('training_data/model.h5')