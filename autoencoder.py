from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keras modules

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, LeakyReLU, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

class AAE():
    def __init__(self, image_shape, image_hepler):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_hepler
        self.img_shape = image_shape
        self.channels = image_shape[2]
        self.latent_dimension = 11
        
        print("Build models...")
        self._build_encoder_model()
        self._build_decoder_generator_model()
        self._build_discriminator_model()
        self._build_and_compile_aae(optimizer)

    def train(self, epochs, train_data, batch_size):
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            
            latent_vector_fake = self.encoder_model.predict(batch)
            latent_vector_real = np.random.normal(size=(batch_size, self.latent_dimension))
            
            loss_real = self.discriminator_model.train_on_batch(latent_vector_real, real)
            loss_fake = self.discriminator_model.train_on_batch(latent_vector_fake, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            #  Train Generator
            generator_loss = self.aae.train_on_batch(batch, [batch, real])

            # Plot the progress
            print ("---------------------------------------------------------")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(discriminator_loss[0]))
            print ("Generator loss: {}".format(generator_loss))
            print ("---------------------------------------------------------")
            
            history.append({"D":discriminator_loss[0],"G":generator_loss})
            
            # Save images from every hundereth epoch generated images
            if epoch % 100 == 0:
                self._save_images(epoch)
                
        self._plot_loss(history)
        self._image_helper.makegif("generated-aae/")        
    
    def _build_encoder_model(self):
        print("Building Encoder...")
        
        encoder_input = Input(shape=self.img_shape)

        encoder_sequence = Flatten()(encoder_input)
        encoder_sequence = Dense(512)(encoder_sequence)
        encoder_sequence = LeakyReLU(alpha=0.2)(encoder_sequence)
        encoder_sequence = Dense(512)(encoder_sequence)
        encoder_sequence = LeakyReLU(alpha=0.2)(encoder_sequence)
        mean = Dense(self.latent_dimension)(encoder_sequence)
        deviation = Dense(self.latent_dimension)(encoder_sequence)
        
        """
                mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                output_shape=lambda p: p[0]
        """
        latent_vector = concatenate([mean, deviation])
        latent_vector = Dense(self.latent_dimension)(latent_vector)
        latent_vector = LeakyReLU(alpha=0.2)(latent_vector)
        #latent_vector = Dense(512)(latent_vector)
        #latent_vector = LeakyReLU(alpha=0.2)(latent_vector)
        
        
        self.encoder_model = Model(encoder_input, latent_vector, name = 'encoder')
        self.encoder_model.summary()
    
    def _build_decoder_generator_model(self):
        print("Building Decoder Generator...")
        
        decoder_generator_input = Input(shape=(self.latent_dimension,))
        decoder_generator_sequence = Dense(512, input_dim=self.latent_dimension)(decoder_generator_input)
        decoder_generator_sequence = LeakyReLU(alpha=0.2)(decoder_generator_sequence)
        decoder_generator_sequence = Dense(512)(decoder_generator_sequence)
        decoder_generator_sequence = LeakyReLU(alpha=0.2)(decoder_generator_sequence)
        decoder_generator_sequence = Dense(np.prod(self.img_shape), activation='tanh')(decoder_generator_sequence)
        decoder_generator_sequence = Reshape(self.img_shape)(decoder_generator_sequence)

        self.decoder_generator_model = Model(decoder_generator_input, decoder_generator_sequence, name = 'decoder')
        self.decoder_generator_model.summary()
        
    def _build_discriminator_model(self):
        print("Building Discriminator...")
        discriminator_input = Input(shape=(self.latent_dimension,))
        discriminator_sequence = Dense(512, input_dim=self.latent_dimension)(discriminator_input)
        discriminator_sequence = LeakyReLU(alpha=0.2)(discriminator_sequence)
        discriminator_sequence = Dense(256)(discriminator_sequence)
        discriminator_sequence = LeakyReLU(alpha=0.2)(discriminator_sequence)
        discriminator_sequence = Dense(1, activation="sigmoid")(discriminator_sequence)
        
        self.discriminator_model = Model(discriminator_input, discriminator_sequence, name = 'discriminator')
        self.decoder_generator_model.summary()
    
    def _build_and_compile_aae(self, optimizer):
               
        print("Compile Discriminator...")
        self.discriminator_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator_model.trainable = False
        
        print("Conecting models...")
        real_input = Input(shape=self.img_shape)
        encoder_output = self.encoder_model(real_input)
        decoder_output = self.decoder_generator_model(encoder_output)
        discriminator_output = self.discriminator_model(encoder_output)        
        
        self.aae = Model(real_input, [decoder_output, discriminator_output], name = 'AAE')
        self.aae.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)
    
    def _save_images(self, epoch):
        noise = np.random.normal(size=(26, self.latent_dimension))
        generated = self.decoder_generator_model.predict(noise)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, "generated-aae/")
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(20,5))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()
