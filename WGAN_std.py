from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import pylab as pl


### Data Generation
nb_samples = 10000
radius = 1 
nz =.1

# generate the data
X_train = np.zeros((nb_samples,2))
r = np.random.normal(radius,nz,nb_samples)
theta=np.random.rand(nb_samples)*2*np.pi
X_train[:,0]=r*np.cos(theta)
X_train[:,1]=r*np.sin(theta)

pl.figure(figsize=(6,6))
pl.scatter(X_train[:,0], X_train[:,1],s = 20, alpha=0.8, \
    edgecolor = 'k', marker = 'o',label='original samples') 
pl.xticks([], [])
pl.yticks([], [])
pl.legend(loc='best')
pl.tight_layout()
pl.show()



### Building the model
def make_generator(noise_dim=100):
    model = Sequential()
    model.add(Dense(128,  kernel_initializer='he_normal', input_dim=noise_dim))
    model.add(Activation('relu'))      
    model.add(Dense(64,  kernel_initializer='he_normal'))
    model.add(Activation('relu'))      
    model.add(Dense(units=2, activation='linear'))
    return model


def make_discriminator():
    model = Sequential()
    model.add(Dense(128, kernel_initializer='he_normal', input_dim=2))
    model.add(Activation('relu'))      
    model.add(Dense(64, kernel_initializer='he_normal', input_dim=2))
    model.add(Activation('relu'))      
    model.add(Dense(units=1, activation='linear'))
    return model


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
    
    
noise_dim=2
generator = make_generator(noise_dim)
discriminator = make_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

# we connect the generator and the discrimnator together to get 
# the full generator model that will be optimized
generator_input = Input(shape=(noise_dim,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], \
    outputs=[discriminator_layers_for_generator])

# we compile the generator
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), \
    loss=wasserstein_loss)

# we can express the total discriminator model by freezing the generator layers
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(noise_dim,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)


discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator])

# finally, we compile it with the two expectation losses
discriminator_model.compile(optimizer=Adam(0.001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,wasserstein_loss])


### Running the Full Model
def generate_images(generator_model,noise_dim, num_samples=1000):
    predicted_samples = generator_model.predict(np.random.rand(num_samples, noise_dim))
    pl.figure(figsize=(6,6))
    pl.scatter(X_train[:,0], X_train[:,1],s = 40, alpha=0.2, edgecolor = 'k', \
        marker = '+',label='original samples')
    pl.scatter(predicted_samples[:,0], predicted_samples[:,1],s = 10, \
        alpha=0.9,c='r', edgecolor = 'k', marker = 'o',label='predicted') 
    pl.xticks([], [])
    pl.yticks([], [])
    pl.legend(loc='best')
    pl.tight_layout()    
    pl.show()

    
def discriminator_clip(f,c):
    for l in f.layers:
        weights = l.get_weights()
        weights = [np.clip(w, -c, c) for w in weights]
        l.set_weights(weights)
        

BATCH_SIZE = 128
TRAINING_RATIO = 5
n_epochs = 1000

positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y


for epoch in range(n_epochs):
    np.random.shuffle(X_train)

    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    
    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        for j in range(TRAINING_RATIO):
            discriminator_clip(discriminator_model,0.3) # 0.3 is a good value for our toy example
            sample_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            
            # we sample the noise
            noise = np.random.rand(BATCH_SIZE, noise_dim).astype(np.float32)
            
            # and we train on batches the discrimnator
            discriminator_model.train_on_batch([sample_batch, noise],[positive_y, negative_y])
        
        # then we train the generator after TRAINING_RATIO updates
        generator_model.train_on_batch(np.random.rand(BATCH_SIZE, noise_dim), positive_y)
        
    #Visualization of intermediate results
    if epoch%50 == 0:
        print("Epoch: ", epoch, "/", n_epochs)
        generate_images(generator, noise_dim)