import tensorflow as tf
import matplotlib.pyplot as plt

from model import model


data_dir = 'sanity-train'
batch_size = 1
img_width = 448
img_height = 448

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    # validation_split=0.2,
    validation_split=0.5,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    # validation_split=0.2,
    validation_split=0.5,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_vds = val_ds.map(lambda x, y: (normalization_layer(x), y))
AUTOTUNE = tf.data.AUTOTUNE
train_ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = normalized_vds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = normalized_vds.cache().prefetch(buffer_size=AUTOTUNE)

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=400
)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()