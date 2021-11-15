import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
MODEL_NAME = 'MobileNetV2_Sequential_Depth'
#base_model = MobileNetV2(input_shape=[128, 128, 3], include_top=False)

def sequential_model(base_model, output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])


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
    model = tf.keras.Model(base_model.input, new_top_layer)

    return model

def copyModel(base_model):
    base_model_copy = tf.keras.models.clone_model(base_model)
    base_model_copy.set_weights(base_model.get_weights())
    return base_model_copy

def MobileNetV2_sequential(output_channels:int):
    base_model = copyModel(MobileNetV2(input_shape=[128, 128, 3], include_top=False))

    return sequential_model(base_model, output_channels=output_channels)

# Create Model A
modelA = MobileNetV2_sequential(output_channels=3)
# Rename Model A layers to be unique
for i, layer in enumerate(modelA.layers):
    layer._name += '_A'

# Create Model B
modelB = MobileNetV2_sequential(output_channels=3)
# Rename Model B layers to be unique
for i, layer in enumerate(modelB.layers):
    layer._name += '_B'


combinedOutput = tf.keras.layers.Concatenate()([modelA.output, modelB.output])
penaultimateLayer = tf.keras.layers.Dense(8, activation='relu')(combinedOutput)
outputLayer = tf.keras.layers.Dense(3)(penaultimateLayer)

model = tf.keras.Model(inputs=[modelB.input, modelA.input], outputs=outputLayer)

model.summary()