import tensorflow as tf
from tensorflow.keras.applications import VGG16
MODEL_NAME = 'VGG16_Sequential_Depth'
#base_model = VGG16(input_shape=[128, 128, 3], include_top=False)

def sequential_model(base_model, output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

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

    return model

def copyModel(base_model):
    base_model_copy = tf.keras.models.clone_model(base_model)
    base_model_copy.set_weights(base_model.get_weights())
    return base_model_copy

def VGG16_sequential(output_channels:int):
    base_model = copyModel(VGG16(input_shape=[128, 128, 3], include_top=False))

    return sequential_model(base_model, output_channels=output_channels)

# Create Model A
modelA = VGG16_sequential(output_channels=3)
# Rename Model A layers to be unique
for i, layer in enumerate(modelA.layers):
    layer._name += '_A'

# Create Model B
modelB = VGG16_sequential(output_channels=3)
# Rename Model B layers to be unique
for i, layer in enumerate(modelB.layers):
    layer._name += '_B'


combinedOutput = tf.keras.layers.Concatenate()([modelA.output, modelB.output])
penaultimateLayer = tf.keras.layers.Dense(8, activation='relu')(combinedOutput)
outputLayer = tf.keras.layers.Dense(3)(penaultimateLayer)

model = tf.keras.Model(inputs=[modelB.input, modelA.input], outputs=outputLayer)

model.summary()