import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

from numpy.random import seed

from models.cnnSmall import model, model_name

SEED = 42
seed(SEED)
tf.random.set_seed(SEED)

train_data_dir = 'SW-PPEW'
# train_data_dir = 'sanity-train'
# val_data_dir = 'sanity-train'
val_data_dir = 'SW-PPEW'
model_path = 'saturday/'
batch_size = 2
img_width = 448
img_height = 448
# todo validation size!
validation_split = 0.2
epochs = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    label_mode='categorical',
    validation_split=validation_split,
    subset="training",
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_data_dir,
    label_mode='categorical',
    validation_split=validation_split,
    subset="validation",
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# model.load_weights(model_path + model_name + ".best.hdf5")
model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])

filepath = model_path + model_name + ".best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint]
)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + model_name + 'acc.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_path + model_name + 'lss.png')
plt.show()

with open(model_path + model_name + '.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

tf.keras.utils.plot_model(model, model_path + model_name + '.png')
