'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.layers.core import *
import keras
from keras_attn_fun import Attention
from keras.callbacks import EarlyStopping

def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)




X_dev = X_test[(len(X_test) * 0.8):]
Y_dev = y_test[(len(X_test) * 0.8):]

y_test = y_test[:(len(X_test) * 0.8)]
X_test = X_test[:(len(X_test) * 0.8)]

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_dev = sequence.pad_sequences(X_dev, maxlen=maxlen)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')


class BestModelCB(keras.callbacks.Callback):

    def __init__(self, dev_x, dev_y):
        self.best_model = None
        self.best_acc = -1
        self.dev_y = dev_y
        self.dev_x = dev_x

        super(BestModelCB, self).__init__()

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        score, dev_acc = self.model.evaluate(self.dev_x, self.dev_y)

        if dev_acc > self.best_acc:
            print('Updated!!')
            self.best_acc = dev_acc
            self.best_model = self.model.get_weights()
        print(dev_acc)

#Inputs

main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
emb = Embedding(max_features, 128, dropout=0.2)(main_input)
lstm = LSTM(128, dropout_W=0.2, dropout_U=0.2)(emb)
#h_n = Lambda(get_H_n, output_shape=(128,), name="h_n")(lstm)
outd = Dense(1,activation='sigmoid')(lstm)

# try using different optimizers and different optimizer configs
model = Model(input=[main_input],output=outd)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
best_model_callback = BestModelCB(X_dev, Y_dev)
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=7, validation_data=[X_test, y_test], callbacks=[best_model_callback ])

print
model.set_weights(best_model_callback.best_model)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

