import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
MODEL_NAME = 'MobileNetV2_Sequential'
base_model = MobileNetV2(input_shape=[128, 128, 3], include_top=False)
cat_layers = []
cat_names = [
    'block_11_expand_BN', 
    'block_12_expand_BN', 
    'block_13_expand_BN']

for layer in base_model.layers:
    if "input" in layer.name:
        continue
        
    if "_expand_BN" in layer.name and layer.name in cat_names:
        print(layer.name, layer.input_shape)
        cat_layers.append(layer.output)

penultimate_layer = base_model.layers[-1]
concat_layer = tf.keras.layers.Concatenate()(cat_layers)
intermediate_layer = tf.keras.layers.Dense(1728)(concat_layer) 
top_layer = tf.keras.layers.Dense(768)(intermediate_layer)
new_top_layer = tf.keras.layers.Reshape((128, 128, 3))(top_layer)
model = tf.keras.models.Model(base_model.input, new_top_layer)
model.summary()