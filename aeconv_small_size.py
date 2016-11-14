from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.layers.core import Reshape
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import glob
from scipy.misc import imread
import random

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

def example_sizes():
     return len(glob.glob('./small_imgs_data/*jpg'))

def data_generator(batch_size):

    counter = 0
    the_data = []
    x = []
    files = glob.glob('./small_imgs_data/*jpg')
    example_size = len(files)

    #Numpy matrices
    data_matrix = np.zeros((batch_size, 64, 128, 3))
    x_matrix = np.zeros((batch_size, 2))

    while True:
        random.shuffle(files)
        
        for f in files:            
            data_matrix[counter] = imread(f)
            xx, yy = f.split('/')[-1][:-4].split(',')
            x_matrix[counter][0] = float(xx)
            x_matrix[counter][1] = float(yy)

            counter += 1

            if counter >= batch_size:
                counter = 0
                yield data_matrix, data_matrix


def load_images(normalize=True):

    counter = 0
    the_data = []
    x = []
    files = glob.glob('./small_imgs_data/*jpg')
    for f in files:
        counter += 1
        if counter > 100: break

        if normalize:
            the_data.append(imread(f)/float(256))
        else:
            the_data.append(imread(f))
        xx, yy = f.split('/')[-1][:-4].split(',')
        x.append((float(xx), float(yy)))
    return np.array(the_data), np.array(x)

def main():

    print 'Loading...'
    the_data, x = load_images(normalize=False)
    norm_data = np.array(the_data,dtype=np.float32) / 255

    input_img = Input(shape=(64,128,3))

    #Conv_1
    e_convolution_1 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')
    e_downsampling_1 = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')

    e_convoluted_1 = e_convolution_1(input_img)    
    e_downsampled_1 = e_downsampling_1(e_convoluted_1)

    #Conv_2
    e_convolution_2 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')
    e_downsampling_2 = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')

    e_convoluted_2 = e_convolution_2(e_downsampled_1)    
    e_downsampled_2 = e_downsampling_2(e_convoluted_2)

    #Conv_3
    e_convolution_3 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')
    e_downsampling_3 = MaxPooling2D(pool_size=(2, 2), dim_ordering='tf')

    e_convoluted_3 = e_convolution_3(e_downsampled_2)    
    e_downsampled_3 = e_downsampling_3(e_convoluted_3)

    #Conv_xx
    e_convolution_xx = Convolution2D(1, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')
    e_convoluted_xx = e_convolution_xx(e_downsampled_3)    

    #
    e_reshape_0 = Reshape((128,))
    encoding = e_reshape_0(e_convoluted_xx)

    #Here be VAE

    batch_size = 100
    original_dim = 128
    latent_dim = 100
    intermediate_dim = 50
    nb_epoch = 50
    epsilon_std = 1.0

    def sampling(args):
        z_mean, z_log_var = args
        #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
        #                          std=epsilon_std)
        return (z_mean + K.exp(z_log_var / 2))# * epsilon

    def add_epsilon(z):
        epsilon = K.random_normal(shape=z.shape, mean=0.,
                                  std=epsilon_std)
        return z * epsilon


    z_mean = Dense(latent_dim)(encoding)
    z_log_var = Dense(latent_dim)(encoding)

    #Works until this!

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    z_s = Lambda(add_epsilon, output_shape=(latent_dim,))(z)


    decoding = Dense(128)(encoding)#(z_s)

    #model = Model(input=input_img, output=z_s)
    #model.compile(optimizer='adam',loss='mse')

    #import pdb;pdb.set_trace()



    #This is the decoder
    reshape_0 = Reshape((8,16,1))
    out_d_reshape = reshape_0(decoding)

    #uconv_1
    #(x,16,8,256) - > (x,32,16,256)
    upsampling_1 = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_1 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    upsampled_1 = upsampling_1(out_d_reshape)
    convoluted_1 = convolution_1(upsampled_1)

    #uconv_2
    #(x,32,16,256) - > (x,64,32,92)

    upsampling_2 = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_2 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    upsampled_2 = upsampling_2(convoluted_1)
    convoluted_2 = convolution_2(upsampled_2)

    #uconv_3
    #(x,64,32,92) - > (x,128,64,92)

    upsampling_3 = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_3 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    upsampled_3 = upsampling_3(convoluted_2)
    convoluted_3 = convolution_3(upsampled_3)

    #uconv_xx
    #(x,256,128,92) - > (x,516,256,3)
    #upsampling_xx = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_xx = Convolution2D(3, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    convoluted_xx = convolution_xx(convoluted_3)

    #Fuck, this does not work :(

    def vae_loss(x, x_decoded):
        #Used to have original_dim * ...
        xent_loss = objectives.mean_squared_error(x, x_decoded)
        #kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss# + kl_loss

    model = Model(input=input_img, output=convoluted_xx)
    model.compile(optimizer='adam',
              loss='mse')#vae_loss)

    tt = 0
    model.load_weights('./model_ae_' + str(tt))
    save(model, the_data, 'ae_small_' + str(tt) + '.jpg')




    lenx = example_sizes()

    for tt in range(10000):
        model.fit_generator(data_generator(50), lenx, nb_epoch=10)
        model.save_weights('./model_ae_' + str(tt))
        save(model, the_data, 'ae_small_' + str(tt) + '.jpg')

    import pdb;pdb.set_trace()

def xxfdc():

    dense_0 = Dense(1024)
    dense_1 = Dense(1024)
    dense = Dense(16*8*256)

    out_0 = dense_0(input_1)
    out_1 = dense_1(out_0)
    out_d = dense(out_1)

    #reshape
    reshape_0 = Reshape((8,16,256))
    out_d_reshape = reshape_0(out_d)

    #uconv_1
    #(x,16,8,256) - > (x,32,16,256)
    upsampling_1 = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_1 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    upsampled_1 = upsampling_1(out_d_reshape)
    convoluted_1 = convolution_1(upsampled_1)

    #uconv_2
    #(x,32,16,256) - > (x,64,32,92)

    upsampling_2 = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_2 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    upsampled_2 = upsampling_2(convoluted_1)
    convoluted_2 = convolution_2(upsampled_2)

    #uconv_3
    #(x,64,32,92) - > (x,128,64,92)

    upsampling_3 = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_3 = Convolution2D(92, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    upsampled_3 = upsampling_3(convoluted_2)
    convoluted_3 = convolution_3(upsampled_3)

    #uconv_xx
    #(x,256,128,92) - > (x,516,256,3)
    #upsampling_xx = UpSampling2D(size=(2, 2), dim_ordering='tf')
    convolution_xx = Convolution2D(3, 5, 5, border_mode='same', activation='relu', dim_ordering='tf')

    convoluted_xx = convolution_xx(convoluted_3)

    model = Model(input=input_1, output=convoluted_xx)
    model.compile(optimizer='adam',
              loss='mse')

    print model.predict(np.array([[0.1,0.5],[1.0,0.6]])).shape
    lenx = example_sizes()

    for tt in range(10000):
        model.fit_generator(data_generator(50), lenx, nb_epoch=10)
        model.save_weights('./model_small_' + str(tt))
        save(model, x, 'small_' + str(tt) + '.jpg')
    import pdb;pdb.set_trace()

def save(model, x, fname):
    from scipy.misc import imsave
    imsave(fname, model.predict(x[:1])[0] )

main()
