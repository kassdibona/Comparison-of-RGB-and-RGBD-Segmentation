import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow_examples.models.pix2pix import pix2pix
MODEL_NAME = 'VGG16_UNet_Depth'

def unet_model(down_stack, up_stack, output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def copyModel(base_model):
    base_model_copy = tf.keras.models.clone_model(base_model)
    base_model_copy.set_weights(base_model.get_weights())
    return base_model_copy

def VGG16_UNet(output_channels:int):
    init_model = copyModel(VGG16(input_shape=[128, 128, 3], include_top=False))

    # Use the activations of these layers
    layer_names = [
        'block2_conv2',   # 64x64
        'block3_conv2',   # 32x32
        'block4_conv2',   # 16x16
        'block5_conv2',  # 8x8
    ]
    init_model_outputs = [
        init_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=init_model.input, outputs=init_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    return unet_model(down_stack, up_stack, output_channels=output_channels)

# Create Model A
# modelA = tf.keras.Sequential()
# modelA.add(tf.keras.layers.InputLayer(input_shape=[128, 128, 1]))
# modelA.add(tf.keras.layers.Dense(128, activation='relu'))
# modelA.add(tf.keras.layers.Dense(64, activation='relu'))
# modelA.add(tf.keras.layers.Dense(3))
modelA = VGG16_UNet(output_channels=3)

# Create Model B
modelB = VGG16_UNet(output_channels=3)

combinedOutput = tf.keras.layers.Concatenate()([modelA.output, modelB.output])
penaultimateLayer = tf.keras.layers.Dense(8, activation='relu')(combinedOutput)
outputLayer = tf.keras.layers.Dense(3)(penaultimateLayer)

model = tf.keras.Model(inputs=[modelB.input, modelA.input], outputs=outputLayer)