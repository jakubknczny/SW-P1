import keras.models
import tensorflow as tf
from numpy.random import seed


SEED = 42
seed(SEED)
tf.random.set_seed(SEED)

evaluation_data_dir = '../datasets/60-20-10-10'
model_path = '../saturday/'
model_name = 'cnnSmall-32e.best.hdf5'


img_width = 448
img_height = 448
batch_size = 2

eval_ds = tf.keras.utils.image_dataset_from_directory(
    evaluation_data_dir,
    label_mode='categorical',
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.load_model(model_path + model_name)
model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])


results = model.evaluate(x=eval_ds)
print('Model name: ' + model_name)
print('Dataset: 60-20-10-10')
print('Loss: ' + str(results[0]))
print('Accuracy: ' + str(results[1]))
