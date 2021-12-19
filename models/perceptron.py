import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential

model_name = 'perceptron-40e-globalMAX'

model = Sequential()
model.add(tf.keras.layers.Rescaling(1./255))
model.add(Dense(128, activation='relu', input_shape=(448, 448, 3)))
model.add(GlobalMaxPooling2D())
model.add(Dense(4, activation='softmax'))

'''
first run was with globalAveragePooling and no rescaling
'''