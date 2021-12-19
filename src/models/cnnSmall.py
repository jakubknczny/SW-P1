import tensorflow as tf
from tensorflow.keras import layers

model_name = 'cnnSmall-32e'
img_width = 448
img_height = 448

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
rescale = tf.keras.layers.Rescaling(1. / 255)(inputs)
conv3 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))(rescale)
conv3_relu = layers.ReLU(max_value=6)(conv3)
conv3_relu = layers.BatchNormalization()(conv3_relu)
conv3_relu = layers.Dropout(0.4)(conv3_relu)

brb_11 = layers.Conv2D(filters=32 * 6, kernel_size=(1, 1))(conv3_relu)
brb_1_relu1 = layers.ReLU(max_value=6)(brb_11)
brb_1_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_1_relu1)
brb_1_relu2 = layers.ReLU(max_value=6)(brb_1_dc)
brb_1_outputs = layers.Conv2D(filters=48, kernel_size=(1, 1), activation='linear')(brb_1_relu2)
brb_1_outputs = layers.BatchNormalization()(brb_1_outputs)
brb_1_outputs = layers.Dropout(0.4)(brb_1_outputs)

brb_21 = layers.Conv2D(filters=40 * 6, kernel_size=(1, 1))(brb_1_outputs)
brb_2_relu1 = layers.ReLU(max_value=6)(brb_21)
brb_2_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_2_relu1)
brb_2_relu2 = layers.ReLU(max_value=6)(brb_2_dc)
brb_2_outputs = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='linear')(brb_2_relu2)
brb_2_outputs = layers.BatchNormalization()(brb_2_outputs)
brb_2_outputs = layers.Dropout(0.4)(brb_2_outputs)

brb_31 = layers.Conv2D(filters=90 * 6, kernel_size=(1, 1))(brb_2_outputs)
brb_3_relu1 = layers.ReLU(max_value=6)(brb_31)
brb_3_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_3_relu1)
brb_3_relu2 = layers.ReLU(max_value=6)(brb_3_dc)
brb_3_outputs = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='linear')(brb_3_relu2)
brb_3_outputs = layers.BatchNormalization()(brb_3_outputs)
brb_3_outputs = layers.Dropout(0.4)(brb_3_outputs)

brb_41 = layers.Conv2D(filters=160 * 6, kernel_size=(1, 1))(brb_3_relu2)
brb_4_relu1 = layers.ReLU(max_value=6)(brb_41)
brb_4_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_4_relu1)
brb_4_relu2 = layers.ReLU(max_value=6)(brb_4_dc)
brb_4_outputs = layers.Conv2D(filters=196, kernel_size=(1, 1), activation='linear')(brb_4_relu2)
brb_4_outputs = layers.BatchNormalization()(brb_4_outputs)
brb_4_outputs = layers.Dropout(0.4)(brb_4_outputs)

brb_51 = layers.Conv2D(filters=240 * 6, kernel_size=(1, 1))(brb_4_outputs)
brb_5_relu1 = layers.ReLU(max_value=6)(brb_51)
brb_5_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_5_relu1)
brb_5_relu2 = layers.ReLU(max_value=6)(brb_5_dc)
brb_5_outputs = layers.Conv2D(filters=360, kernel_size=(1, 1), activation='linear')(brb_5_relu2)
# stem
stem_avgpool = layers.GlobalMaxPooling2D()(brb_5_outputs)
stem_dense = layers.Dense(units=4, activation=tf.keras.activations.softmax)(stem_avgpool)

model = tf.keras.Model(inputs=inputs, outputs=stem_dense)
