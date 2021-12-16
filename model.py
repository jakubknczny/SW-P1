import tensorflow as tf
from tensorflow.keras import layers


img_width = 448
img_height = 448

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
conv3 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2))(inputs)
conv3_relu = layers.ReLU(max_value=6)(conv3)
# bottleneck residual block 1 - in 224x224x32 - out 112x112x48
brb_11 = layers.Conv2D(filters=32 * 6, kernel_size=(1, 1))(conv3_relu)
brb_1_relu1 = layers.ReLU(max_value=6)(brb_11)
brb_1_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_1_relu1)
brb_1_relu2 = layers.ReLU(max_value=6)(brb_1_dc)
brb_1_outputs = layers.Conv2D(filters=48, kernel_size=(1, 1), activation='linear')(brb_1_relu2)
# inputs for bottleneck residual block 2 - out 112x112x(32 + 48 = 80) => 40
conv3_for_brb_2 = layers.AvgPool2D(pool_size=(2, 2))(conv3_relu)
brb_2_concat = layers.concatenate([brb_1_outputs, conv3_for_brb_2])
pre_2_bn = layers.BatchNormalization()(brb_2_concat)
pre_2_relu = layers.ReLU(max_value=6)(pre_2_bn)
brb_2_inputs = layers.Conv2D(filters=40, kernel_size=(1, 1))(pre_2_relu)
# bottleneck residual block 2 - in 56x56x40 - out 56x56x64
brb_21 = layers.Conv2D(filters=40 * 6, kernel_size=(1, 1))(brb_2_inputs)
brb_2_relu1 = layers.ReLU(max_value=6)(brb_21)
brb_2_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_2_relu1)
brb_2_relu2 = layers.ReLU(max_value=6)(brb_2_dc)
brb_2_outputs = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='linear')(brb_2_relu2)
# inputs for bottleneck residual block 3 - out 56x56x(64 + 40 + 32 = 136) => 90
conv3_for_brb_3 = layers.AvgPool2D(pool_size=(4, 4))(conv3_relu)
brb_1_for_brb_3 = layers.AvgPool2D(pool_size=(2, 2))(brb_1_outputs)
brb_3_concat = layers.concatenate([brb_2_outputs, brb_1_for_brb_3, conv3_for_brb_3])
pre_3_bn = layers.BatchNormalization()(brb_3_concat)
pre_3_relu = layers.ReLU(max_value=6)(pre_3_bn)
brb_3_inputs = layers.Conv2D(filters=90, kernel_size=(1, 1))(pre_3_relu)
# bottleneck residual block 3 - in 56x56x90 - out 28x28x128
brb_31 = layers.Conv2D(filters=90 * 6, kernel_size=(1, 1))(brb_3_inputs)
brb_3_relu1 = layers.ReLU(max_value=6)(brb_31)
brb_3_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_3_relu1)
brb_3_relu2 = layers.ReLU(max_value=6)(brb_3_dc)
brb_3_outputs = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='linear')(brb_3_relu2)
# inputs for bottleneck residual block 4 - out 14x14x(128 + 64 + 40 + 32 = 264) => 160
conv3_for_brb_4 = layers.AvgPool2D(pool_size=(8, 8))(conv3_relu)
brb_1_for_brb_4 = layers.AvgPool2D(pool_size=(4, 4))(brb_1_outputs)
brb_2_for_brb_4 = layers.AvgPool2D(pool_size=(2, 2))(brb_2_outputs)
brb_4_concat = layers.concatenate([brb_3_outputs, brb_2_for_brb_4, brb_1_for_brb_4, conv3_for_brb_4])
pre_4_bn = layers.BatchNormalization()(brb_4_concat)
pre_4_relu = layers.ReLU(max_value=6)(pre_4_bn)
brb_4_inputs = layers.Conv2D(filters=160, kernel_size=(1, 1))(pre_4_relu)
# bottleneck residual block 4 - in 14x14x160 - out 7x7x196
brb_41 = layers.Conv2D(filters=160 * 6, kernel_size=(1, 1))(brb_4_inputs)
brb_4_relu1 = layers.ReLU(max_value=6)(brb_41)
brb_4_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_4_relu1)
brb_4_relu2 = layers.ReLU(max_value=6)(brb_4_dc)
brb_4_outputs = layers.Conv2D(filters=196, kernel_size=(1, 1), activation='linear')(brb_4_relu2)
# inputs for bottleneck residual block 5 - out 14x14x(196 + 128 + 64 + 40 + 32 = 460) => 240
conv3_for_brb_5 = layers.AvgPool2D(pool_size=(16, 16))(conv3_relu)
brb_1_for_brb_5 = layers.AvgPool2D(pool_size=(8, 8))(brb_1_outputs)
brb_2_for_brb_5 = layers.AvgPool2D(pool_size=(4, 4))(brb_2_outputs)
brb_3_for_brb_5 = layers.AvgPool2D(pool_size=(2, 2))(brb_3_outputs)
brb_4_concat = layers.concatenate([brb_4_outputs, brb_3_for_brb_5, brb_2_for_brb_5, brb_1_for_brb_5, conv3_for_brb_5])
pre_4_bn = layers.BatchNormalization()(brb_4_concat)
pre_4_relu = layers.ReLU(max_value=6)(pre_4_bn)
brb_4_inputs = layers.Conv2D(filters=240, kernel_size=(1, 1))(pre_4_relu)
# bottleneck residual block 4 - in 7x7x240 - out 7x7x360
brb_51 = layers.Conv2D(filters=240 * 6, kernel_size=(1, 1))(brb_4_inputs)
brb_5_relu1 = layers.ReLU(max_value=6)(brb_51)
brb_5_dc = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2))(brb_5_relu1)
brb_5_relu2 = layers.ReLU(max_value=6)(brb_5_dc)
brb_5_outputs = layers.Conv2D(filters=360, kernel_size=(1, 1), activation='linear')(brb_5_relu2)
# stem
stem_avgpool = layers.Flatten()(brb_5_outputs)
stem_dense = layers.Dense(units=4)(stem_avgpool)
outputs = layers.Softmax()(stem_dense)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
tf.keras.utils.plot_model(model, 'temp.png')
