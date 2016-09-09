

import numpy
import numpy as np
import random

from keras import backend as K
from keras.engine.topology import Layer

import theano.tensor as T
import theano


class SuperSimpleAttnWithEncoding(Layer):


    def __init__(self, output_dim, return_sequence=True, **kwargs):

        self.output_dim = output_dim
        self.return_sequence = return_sequence
        super(SuperSimpleAttnWithEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.x_input_shape = input_shape
        self.W_h = K.variable(np.random.random((input_dim, input_dim)))
        self.W_y = K.variable(np.random.random((input_dim, input_dim)))
        self.w_a = K.variable(np.random.random((input_dim,)))

        self.trainable_weights = [self.W_y, self.w_a, self.W_h]

    def call(self, x, mask=None):

        full_input = x[0]
        encoding = x[1]

        whhn = T.dot(encoding, self.W_h)
        whhn_r = T.extra_ops.repeat(whhn, full_input.shape[1], axis=0).reshape(full_input.shape)

        M = T.tanh(T.dot(full_input, self.W_y) + whhn_r)
        a =  T.nnet.softmax(T.dot(M, self.w_a))
        ar = T.extra_ops.repeat(a, full_input.shape[-1], axis=1).reshape(full_input.shape)

        if self.return_sequence:
            return full_input * ar
        else:
            return T.sum(full_input * ar, axis=1)

    def call_softmax(self, x, mask=None):

        full_input = x[0]
        encoding = x[1]

        whhn = T.dot(encoding, self.W_h)
        whhn_r = T.extra_ops.repeat(whhn, full_input.shape[1], axis=0).reshape(full_input.shape)

        M = T.tanh(T.dot(full_input, self.W_y) + whhn_r)
        a =  T.nnet.softmax(T.dot(M, self.w_a))

        return a

    def get_output_shape_for(self, input_shape):

        if self.return_sequence:
            return self.x_input_shape[0]
        else:
            return self.x_input_shape[1]#(self.x_input_shape[0], self.x_input_shape[-1])



class SuperSimpleAttn(Layer):


    def __init__(self, output_dim, return_sequence=True, **kwargs):

        self.output_dim = output_dim
        self.return_sequence = return_sequence
        super(SuperSimpleAttn, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.x_input_shape = input_shape
        self.W_y = K.variable(np.random.random((input_dim, input_dim)))
        self.w_a = K.variable(np.random.random((input_dim,)))
        self.trainable_weights = [self.W_y, self.w_a]

    def call(self, x, mask=None):

        full_input = x
        M = T.tanh(T.dot(full_input, self.W_y))
        a =  T.nnet.softmax(T.dot(M, self.w_a))
        ar = T.extra_ops.repeat(a, full_input.shape[-1], axis=1).reshape(full_input.shape)

        if self.return_sequence:
            return full_input * ar
        else:
            return T.sum(full_input * ar, axis=1)

    def call_softmax(self, x, mask=None):

        full_input = x
        M = T.tanh(T.dot(full_input, self.W_y))
        a =  T.nnet.softmax(T.dot(M, self.w_a))

        return a

    def get_output_shape_for(self, input_shape):

        if self.return_sequence:
            return self.x_input_shape
        else:
            return (self.x_input_shape[0], self.x_input_shape[-1])


def main():

    window = 5
    vec_size = 20
    vec_len = 20

    #Make the fucking data
    x = []
    y = []
    for ax in range(90000):
        #
        cx = numpy.ones(window)
        #label
        label = random.randint(0,1)
        if label > 0:
            cx[random.randint(0,window - 1)] = 2
        else:
            cx[random.randint(0,window - 1)] = 3          
        x.append(cx)
        y.append(label)

    x = numpy.array(x, dtype=np.int32)
    y = numpy.array(y)

    dev_x = x[8000:]
    dev_y = y[9000:]

    x = x[:8000]
    y = y[:8000]

    #And now for the little goddamn network

    from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
    from keras.layers.embeddings import Embedding
    from keras.models import Model
    from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, TimeDistributedDense, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Convolution1D
    from keras.layers.recurrent import SimpleRNN, LSTM, GRU
    from keras.layers.pooling import MaxPooling1D

    x_input = Input(shape=(window, ), name='x_input', dtype='int32')

    char_emb = Embedding(4, vec_size, input_length=window, mask_zero=False)
    emb_out = char_emb(x_input)

    simple_attn = SuperSimpleAttn(emb_out.shape)

    emb_attn = simple_attn(emb_out)

    cl = Convolution1D(64, 3, border_mode='same', input_shape=(window, vec_len))
    mp_1 = MaxPooling1D(pool_length=window)

    cl_out = cl(emb_attn)
    mp_1_out = mp_1(cl_out)

    flattener = Flatten()
    f_out = flattener(mp_1_out)

    l_dense = Dense(1, activation='sigmoid')
    d_out = l_dense(f_out)

    model = Model(input=[x_input], output=d_out)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(x,y,nb_epoch=1)

    #Softmax output functions
    t_input = T.imatrix()
    testf = theano.function([t_input], simple_attn.call_softmax(char_emb.call(t_input)))

    np.set_printoptions(precision=3)
    for example in dev_x[:100]:
        print 'Example', example
        print 'Attention'
        print testf(np.array([example],dtype=np.int32))[0]
        print 
        print

    import pdb;pdb.set_trace()

####Test with encoding vec

    x_input = Input(shape=(window, ), name='x_input', dtype='int32')

    char_emb = Embedding(4, vec_size, input_length=window, mask_zero=False)
    emb_out = char_emb(x_input)

    simple_attn = SuperSimpleAttnWithEncoding(emb_out.shape)

    rnn = GRU(vec_size)
    encoding = rnn(emb_out)

    emb_attn = simple_attn([emb_out, encoding])

    cl = Convolution1D(64, 3, border_mode='same', input_shape=(window, vec_len))
    mp_1 = MaxPooling1D(pool_length=window)

    cl_out = cl(emb_attn)
    mp_1_out = mp_1(cl_out)

    flattener = Flatten()
    f_out = flattener(mp_1_out)

    l_dense = Dense(1, activation='sigmoid')
    d_out = l_dense(f_out)

    model = Model(input=[x_input], output=d_out)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(x,y,nb_epoch=1)

    #Softmax output functions
    t_input = T.imatrix()
    testf = theano.function([t_input], simple_attn.call_softmax([char_emb.call(t_input), rnn.call(char_emb.call(t_input))]))

    np.set_printoptions(precision=3)
    for example in dev_x[:100]:
        print 'Example', example
        print 'Attention'
        print testf(np.array([example],dtype=np.int32))[0]
        print 
        print



main()
