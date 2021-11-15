import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow_examples.models.pix2pix import pix2pix
MODEL_NAME = 'VGG16_UNet'
base_model = VGG16(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block2_conv2',   # 64x64
    'block3_conv2',   # 32x32
    'block4_conv2',   # 16x16
    'block5_conv2',  # 8x8
]
base_model_outputs = [
    base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
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

OUTPUT_CLASSES = 3
model = unet_model(output_channels=OUTPUT_CLASSES)