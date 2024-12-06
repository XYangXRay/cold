import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv1DTranspose, Dropout, BatchNormalization, LeakyReLU, Flatten, Reshape
import numpy as np

def nor_data(data):
    return (data - data.min()) / (data.max() - data.min())

def tfnor_data(data):
    return (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data))

# def tfnor_data(data):
#     return data - tf.reduce_min(data)
 
def tfnor_recon(data):
    data = data / tf.reduce_max(data)
    data = data - tf.reduce_min(data)
    return data
    
def nn_model(data, kernel):
    model = tf.keras.Sequential([
        Dense(1024, activation=tf.nn.relu, input_shape=(data.shape[1],)), 
  # Dropout(0.25),
  # BatchNormalization(),
        Dense(512),
        LeakyReLU(),
  # Dropout(0.25),
  # BatchNormalization(),
        Dense(512),
        LeakyReLU(),
  # Dropout(0.25),
  # BatchNormalization(),
  # Dense(512, activation=LeakyReLU, use_bias=True),
  
  
        Dense(kernel.shape[1])
        ])
    # model = tf.keras.Sequential([
    #     Dense(1024, activation=tf.nn.relu, input_shape=(data.shape[1],)), 
    #     # Dropout(0.25),
    #     # BatchNormalization(),
        
    #     Dense(512, activation= tf.nn.relu, use_bias=True),
        
    #     # Dropout(0.25),
    #     # BatchNormalization(),
    #     Dense(512, activation=tf.nn.relu, use_bias=True),
       
    #     # Dropout(0.25),
    #     # BatchNormalization(),
    #     Dense(512, activation=tf.nn.relu, use_bias=True),
    #     Dense(256, activation=tf.nn.relu, use_bias=True),
    #     # BatchNormalization(),
    #     # Dense(kernel.shape[1])
    #     # Dense(kernel.shape[1], activation=tf.nn.leaky_relu),
    #     # Dense(kernel.shape[1], activation=tf.nn.softplus)
    #     Dense(kernel.shape[1])
    
    # ])
    return model

# def nn_model(data, kernel):
#     model = tf.keras.Sequential()
#     model.add(Input(shape=(None, data.shape[1])))
#     model.add(Dense(1024, activation=tf.nn.relu))
#     # model.add(Dropout(0.25))
#     # model.add(BatchNormalization())
#     # model.add(Conv1D(32, 3, activation='relu', padding='same'))
#     # model.add(Conv1D(32, 3, activation='relu', padding='same'))
    
    
#     # model.add(Dense(512, activation=tf.nn.relu, use_bias=True))
#     # model.add(Dropout(0.25))
#     # model.add(BatchNormalization())
#     # model.add(Dense(512, activation=tf.nn.relu, use_bias=True))
#     # model.add(Dropout(0.25))
#     # model.add(BatchNormalization())
#     # model.add(Dense(256, activation=tf.nn.relu, use_bias=True))
#     # model.add(Dropout(0.25))
#     # model.add(BatchNormalization())
#     # model.add(Dense(512, activation=tf.nn.relu, use_bias=True))
#     # model.add(Dropout(0.25))
#     model.add(Dense(kernel.shape[1]))
#     return model
 
def conv1d_model(data, kernel):
    model = Sequential()
    model.add(Conv1DTranspose(16, 7, activation='relu', padding='same', input_shape=(data.shape[1], 1)))
    model.add(Conv1DTranspose(32, 11, activation='relu', padding='same'))
    # model.add(Conv1D(32, 5, activation='relu', padding='same'))
    # model.add(Conv1D(32, 3, activation='relu', padding='same'))
    # model.add(Conv1D(1, 3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(kernel.shape[1]))
    model.add(Reshape((kernel.shape[1], 1)))
    model.add(Conv1D(32, 11, activation='relu', padding='same'))
    model.add(Conv1D(16, 7, activation='relu', padding='same'))
    model.add(Conv1D(1, 3, padding='same'))
    return model
    
    
    # model = Sequential()
    # model.add(Conv1DTranspose(128, 7, activation='relu', padding='same', input_shape=(data.shape[1], 1)))
    # model.add(Conv1DTranspose(256, 11, activation='relu', padding='same'))
    # # model.add(Conv1D(32, 5, activation='relu', padding='same'))
    # # model.add(Conv1D(32, 3, activation='relu', padding='same'))
    # # model.add(Conv1D(1, 3, activation='relu', padding='same'))
    # model.add(Flatten())
    # model.add(Dense(kernel.shape[1]))
    # model.add(Reshape((kernel.shape[1], 1)))
    # model.add(Conv1D(256, 11, activation='relu', padding='same'))
    # model.add(Conv1D(128, 7, activation='relu', padding='same'))
    # model.add(Conv1D(1, 3, padding='same'))
    # return model
        

# def build_conv1D_model():

# #   n_timesteps = train_data_reshaped.shape[1] #13
# #   n_features  = train_data_reshaped.shape[2] #1 
#   model = Sequential(name="model_conv1D")
#   model.add(Input(shape=(n_timesteps,n_features)))
#   model.add(Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"))
#   model.add(Dropout(0.5))
#   model.add(Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
  
#   model.add(Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
  
#   model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
#   model.add(Flatten())
#   model.add(Dense(32, activation='relu', name="Dense_1"))
#   model.add(Dense(n_features, name="Dense_2"))


# #   optimizer = tf.keras.optimizers.RMSprop(0.001)

# #   model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
#   return model


class DNNrec1d:
    def __init__(self):
        self.learning_rate
        
        pass
    
    def loss(self):
        loss_object = tf.keras.losses.MeanSquaredError() 
        recon = model(data, kernel)
        recon = tf.reshape(recon, kernel.shape[1])
        data_model = tf.tensordot(kernel, recon, axes=1)
        
    def grad(model, data, kernel):
        with tf.GradientTape() as tape:
            loss_value = loss(model, data, kernel)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)
        
    def recon(data, kernel, num_epochs = 2001):
        data = tf.cast(data, tf.float32)
        kernel = tf.cast(kernel, tf.float32)
        train_loss_results = []
        model = nn_model(data, kernel)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        for epoch in range(num_epochs):
            loss_value, grads = grad(model, data, kernel)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_results.append(loss_value)
        if epoch % 200 == 0:
            print("Epoch {:04d}: Loss: {:.5f}".format(epoch, loss_value))
    

        

def loss(model, data, kernel):
    # loss_object = tf.keras.losses.MeanSquaredError() 
    loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
    recon = model(data, kernel)
    recon = tfnor_recon(recon)
    # recon = tf.math.abs(recon)
    # print(recon)
    # recon = recon/tf.math.reduce_sum(recon)
    recon = tf.reshape(recon, kernel.shape[1])
    
    data_model = tf.tensordot(kernel, recon, axes=1)
    data_model = tfnor_data(data_model)   
    data_model = tf.reshape(data_model, [1, data.shape[1]]) 
    # print(data_model.shape)
    return loss_object(data, data_model)

def grad(model, data, kernel):
  with tf.GradientTape() as tape:
    loss_value = loss(model, data, kernel)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def ca_fit(data, kernel, num_epochs = 2001):
    data = tf.cast(data, tf.float32)
    kernel = tf.cast(kernel, tf.float32)
    train_loss_results = []
    # model = nn_model(data, kernel)
    model = conv1d_model(data, kernel)
    tf.keras.utils.plot_model(model, show_shapes=True)
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    for epoch in range(num_epochs):
        loss_value, grads = grad(model, data, kernel)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_results.append(loss_value)
        # if epoch % 200 == 0:
            # print("Epoch {:04d}: Loss: {:.5f}".format(epoch, loss_value))
    sig = model(data)
    # sig = tfnor_data(sig)
    sig = tfnor_recon(sig)
    # sig = tf.math.abs(sig)
    sig = np.reshape(sig, (kernel.shape[1]))
    return sig

def signal_compute(model, data, sig_l):
    sig_dnn = model(data)
    sig_dnn = tfnor_recon(sig_dnn)
    sig_dnn = np.reshape(sig_dnn,(sig_l))
    return sig_dnn



