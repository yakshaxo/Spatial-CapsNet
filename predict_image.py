import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ----------------------------------------
# --- FILE PATHS (IMPORTANT: CHECK THESE) ---
folder = r"C:\fashion-mnist\\" 
MODEL_SAVE_PATH = folder + "capsnet_fashion_model.keras" 
NEW_IMAGE_PATH = folder + "shirt1.png"
# ------------------------------------

# ----------------------------------------
# 1. CAPSULE NETWORK COMPONENTS
# ----------------------------------------

# FIX: Use 'tf.keras.utils' which works for older TensorFlow versions
@tf.keras.utils.register_keras_serializable()
def squash(vectors, axis=-1):
    s2 = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s2 / (1 + s2) / tf.sqrt(s2 + 1e-7)
    return scale * vectors

@tf.keras.utils.register_keras_serializable()
def get_capsule_length(z):
    return tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1))

@tf.keras.utils.register_keras_serializable()
def margin_loss(y_true, y_pred):
    m_plus, m_minus, lamb = 0.9, 0.1, 0.5
    loss_present = y_true * tf.square(tf.maximum(0., m_plus - y_pred))
    loss_absent = lamb * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
    L = loss_present + loss_absent
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]
        self.W = self.add_weight(
            shape=(self.input_num_caps, self.num_capsules, self.input_dim_caps, self.dim_capsule),
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
        

# ----------------------------------------
# 2. PREDICTION LOGIC
# ----------------------------------------

def run_prediction():
    """Loads the model and predicts the class of the specified image."""

    if not os.path.exists(NEW_IMAGE_PATH):
        print(f"\n--- ❌ ERROR: Image file not found ---")
        print(f"Please check and update the path: {NEW_IMAGE_PATH}")
        return
        
    print(f"\n--- Running Standalone Prediction ---")
    
    # Custom objects dictionary
    CUSTOM_OBJECTS = {
        "CapsuleLayer": CapsuleLayer,
        "margin_loss": margin_loss,
        "squash": squash,
        "get_capsule_length": get_capsule_length
    }
    
    # A. Load the Model
    # Note: safe_mode=False is only explicitly required in newer TF versions, 
    # but we include it for safety if your version supports it.
    try:
        try:
            # Try loading with safe_mode argument (Newer TF)
            loaded_model = load_model(MODEL_SAVE_PATH, 
                                      custom_objects=CUSTOM_OBJECTS,
                                      safe_mode=False)
        except TypeError:
            # Fallback for Older TF that doesn't accept safe_mode
            loaded_model = load_model(MODEL_SAVE_PATH, 
                                      custom_objects=CUSTOM_OBJECTS)

    except Exception as e:
        print(f"Error loading model from {MODEL_SAVE_PATH}. Error detail: {e}")
        print("ACTION: Ensure you ran 'fashion.py' with epochs=1 to save the corrected model.")
        return

    # B. Prepare the Image Input
    print(f"Loading and preprocessing image: {NEW_IMAGE_PATH}")
    
    try:
        img = Image.open(NEW_IMAGE_PATH).convert('L').resize((28, 28))
    except Exception as e:
        print(f"Error opening image file: {e}")
        return
        
    img_array = np.array(img, dtype='float32') / 255.0
    input_data = np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0)

    # C. Prediction
    prediction_results = loaded_model.predict(input_data)
    
    # D. Interpretation
    predicted_index = np.argmax(prediction_results[0])
    confidence = prediction_results[0][predicted_index]
    
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", 
        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    print("\n[PREDICTION RESULT] 👚")
    print(f"Image File: {os.path.basename(NEW_IMAGE_PATH)}")
    print(f"Predicted Class: **{CLASS_NAMES[predicted_index]}**")
    print(f"Confidence (Capsule Length): {confidence:.4f}")
    print("-" * 35)

# ----------------------------------------
# EXECUTE
# ----------------------------------------
if __name__ == '__main__':
    run_prediction()