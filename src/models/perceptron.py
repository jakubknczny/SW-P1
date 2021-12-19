import keras.layers
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D

img_width = 448
img_height = 448
model_name = 'perceptron-40e'

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = keras.layers.Rescaling(1./255)(inputs)
x = Dense(128, activation='relu')(x)
x = GlobalAveragePooling2D()(x)
outputs = Dense(4, activation='softmax')(x)


model = tf.keras.Model(inputs=inputs, outputs=outputs)

tf.keras.utils.plot_model(model, 'temp.png')
'''
first run was with globalAveragePooling and no rescaling
'''