import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, DepthwiseConv2D, Input, Conv2D
from tensorflow.keras.models import Sequential

model_name = 'cnn-357357-4+4e'
img_width = 448
img_height = 448

model = Sequential()
model.add(Input(shape=(img_height, img_width, 3)))
model.add(tf.keras.layers.Rescaling(1./255))
model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Conv2D(filters=32, kernel_size=(5, 5)))
model.add(DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), depth_multiplier=1))
model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Conv2D(filters=32, kernel_size=(5, 5)))
model.add(DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), depth_multiplier=1))
model.add(GlobalMaxPooling2D())
model.add(Dense(4, activation='softmax'))


model.summary()
