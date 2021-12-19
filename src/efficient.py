import keras

from keras.applications.mobilenet import MobileNet
import keras.backend as K
from tensorflow.keras.applications import EfficientNetB0
from keras import layers

from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout, GlobalAveragePooling2D


base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    classifier_activation="softmax",
    input_shape=(448,448,3))

model = keras.models.Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(4, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(train_ds,
                     batch_size=2,
                     epochs=400,
                     validation_data= val_ds,
                     validation_batch_size=2)