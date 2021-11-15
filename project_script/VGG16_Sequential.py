import tensorflow as tf
from tensorflow.keras.applications import VGG16
MODEL_NAME = 'VGG16_Sequential'
base_model = VGG16(input_shape=[128, 128, 3], include_top=False)
cat_layers = []
for layer in base_model.layers:
    if "input" in layer.name:
        continue
        
    if "_pool" in layer.name:
        print(layer.name, layer.input_shape)
        cat_layers.append(layer.output)
        
new_cat_layers = []
for cat_layer in cat_layers:
    extra_dense_layer = tf.keras.layers.Dense(3072)(cat_layer)
    reshaped_layer = tf.keras.layers.Reshape((128, 128, -1))(extra_dense_layer)
    
    #new_cat_layers.append(extra_dense_layer)
    new_cat_layers.append(reshaped_layer)

penultimate_layer = base_model.layers[-1]
# Note penultimate_layer is one of the above layers
concat_layer = tf.keras.layers.Concatenate()(new_cat_layers)
intermediate_layer = tf.keras.layers.Dense(768)(concat_layer)
top_layer = tf.keras.layers.Dense(3)(intermediate_layer)
model = tf.keras.models.Model(base_model.input, top_layer)
model.summary()