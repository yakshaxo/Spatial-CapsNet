import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ----------------------------------------
#           IDX FILE LOADER
# ----------------------------------------
def load_images(filepath):
    """Loads images from the Fashion-MNIST IDX format file."""
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
        imgs = img_data.reshape(num, rows, cols, 1)
        return imgs.astype('float32') / 255.0

def load_labels(filepath):  
    """Loads labels from the Fashion-MNIST IDX format file."""
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# ----------------------------------------
#               LOAD DATA
# ----------------------------------------
folder = r"C:\fashion-mnist\\" 
MODEL_SAVE_PATH = folder + "capsnet_fashion_model.keras" 

# Load all training and test data
x_train = load_images(folder + "train-images-idx3-ubyte")
y_train = load_labels(folder + "train-labels-idx1-ubyte")
x_test  = load_images(folder + "t10k-images-idx3-ubyte")
y_test  = load_labels(folder + "t10k-labels-idx1-ubyte")

print("train:", x_train.shape, y_train.shape)
print("test :", x_test.shape, y_test.shape)

# one-hot encode labels
y_train_ohe = tf.one_hot(y_train, depth=10)
y_test_ohe  = tf.one_hot(y_test, depth=10)

# ----------------------------------------
#             CAPSULE NETWORK COMPONENTS
# ----------------------------------------


@tf.keras.utils.register_keras_serializable()
def squash(vectors, axis=-1):
    """Squashing function for capsule vector output."""
    s2 = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s2 / (1 + s2) / tf.sqrt(s2 + 1e-7)
    return scale * vectors

@tf.keras.utils.register_keras_serializable()
def get_capsule_length(z):
    """Calculates the Euclidean length (magnitude) of the capsule vector."""
    return tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1))

class CapsuleLayer(layers.Layer):
    """The core Capsule Layer performing dynamic routing."""
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]
        self.W = self.add_weight(
            shape=(self.input_num_caps, self.num_capsules, 
                   self.input_dim_caps, self.dim_capsule),
            initializer='glorot_uniform',
            trainable=True)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        u_expanded = tf.expand_dims(inputs, 2)
        u_expanded = tf.expand_dims(u_expanded, 4)
        u_tiled = tf.tile(u_expanded, [1, 1, self.num_capsules, 1, 1])
        W_tiled = tf.expand_dims(self.W, 0)
        W_tiled_transposed = tf.transpose(W_tiled, perm=[0, 1, 2, 4, 3])
        u_hat = tf.squeeze(tf.matmul(W_tiled_transposed, u_tiled), axis=-1)
        b = tf.zeros([tf.shape(inputs)[0], self.input_num_caps, self.num_capsules])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.einsum('bij,bijk->bjk', c, u_hat)
            v = squash(s)
            if i < self.routings - 1:
                b += tf.einsum('bijk,bjk->bij', u_hat, v)
        return v    

def build_capsnet():
    x = layers.Input(shape=(28, 28, 1))
    
    conv1 = layers.Conv2D(256, (9, 9), activation='relu', name='conv1')(x)
    primary = layers.Conv2D(8 * 32, (9, 9), strides=2, activation='relu', name='primary_conv')(conv1)
    primary = layers.Reshape(target_shape=[-1, 8], name='primary_reshape')(primary)
    primary = layers.Lambda(squash, name='primary_squash')(primary)

    digit_caps = CapsuleLayer(10, 16, routings=3, name='digit_caps')(primary)

    out_caps = layers.Lambda(
        get_capsule_length,
        output_shape=(10,), 
        name='output_length'
    )(digit_caps)

    return models.Model(inputs=x, outputs=out_caps)

# ----------------------------------------
#             LOSS FUNCTION
# ----------------------------------------
@tf.keras.utils.register_keras_serializable()
def margin_loss(y_true, y_pred):
    """Standard margin loss for Capsule Networks."""
    m_plus, m_minus, lamb = 0.9, 0.1, 0.5
    loss_present = y_true * tf.square(tf.maximum(0., m_plus - y_pred))
    loss_absent = lamb * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
    L = loss_present + loss_absent
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# ----------------------------------------
#             TRAIN AND SAVE MODEL
# ----------------------------------------
model = build_capsnet()

model.compile(optimizer='adam',
              loss=margin_loss,
              metrics=['accuracy'])

print("\nStarting Training...")

model.fit(x_train, y_train_ohe,
          batch_size=128,
          epochs=10, 
          validation_data=(x_test, y_test_ohe))

# --- SAVE THE TRAINED MODEL ---
print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH, save_format='keras') 
# ---------------------------------------

loss, acc = model.evaluate(x_test, y_test_ohe)
print(f"Test accuracy: {acc * 100:.2f}%")