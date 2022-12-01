import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
import numpy as np

def nor_data(data):
    return (data - data.min()) / (data.max() - data.min())

def tfnor_data(data):
    return (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data))
 
def tfnor_recon(data):
    data = data / tf.reduce_max(data)
    data = data - tf.reduce_min(data)
    return data
    
def nn_model(data, kernel):
    model = tf.keras.Sequential([
        Dense(1024, activation=tf.nn.relu, input_shape=(data.shape[1],)), 
  # Dropout(0.25),
  # BatchNormalization(),
        Dense(512, activation= tf.nn.relu, use_bias=True),
        
  # Dropout(0.25),
  # BatchNormalization(),
        Dense(512, activation=tf.nn.relu, use_bias=True),
       
  # Dropout(0.25),
  # BatchNormalization(),
        Dense(512, activation=tf.nn.relu, use_bias=True),
        Dense(256, activation=tf.nn.relu, use_bias=True),
        Dense(kernel.shape[1])
        ])
    return model
    


def loss(model, data, kernel):
    loss_object = tf.keras.losses.MeanSquaredError() 
    recon = model(data, kernel)
    recon = tfnor_recon(recon)
    # print(recon.shape)
    recon = tf.reshape(recon, kernel.shape[1])
    # print(recon.shape)
    data_model = tf.tensordot(kernel, recon, axes=1)
    data_model = tfnor_data(data_model)   
     
    # print(data_model.shape)
    return loss_object(data, data_model)

def grad(model, data, kernel):
  with tf.GradientTape() as tape:
    loss_value = loss(model, data, kernel)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def ca_fit(data, kernel, num_epochs = 2001):
    train_loss_results = []
    model = nn_model(data, kernel)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(num_epochs):
        loss_value, grads = grad(model, data, kernel)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_results.append(loss_value)
        # if epoch % 200 == 0:
            # print("Epoch {:04d}: Loss: {:.5f}".format(epoch, loss_value))
    return model

def signal_compute(model, data):
    sig_dnn = model(data)
    sig_dnn = tfnor_recon(sig_dnn)
    sig_dnn = np.reshape(sig_dnn,(256))
    return sig_dnn



