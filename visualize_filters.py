import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot
from models.kcnn import model


img_width = 448
img_height = 448

model_name = 'kcnn-01-32e'

model = keras.models.load_model('saturday/' + model_name + '.best.hdf5')
#model.summary()

layer = model.get_layer(index = 2)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

def visualize_filters_matrices(layer):
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    pyplot.figure(figsize=(12, 12))
    for f_index in range(n_of_filters):
        filter = filters[:, :, :, f_index]
        for j in range(3):
            ax = pyplot.subplot(n_of_filters, 4, f_index + j + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(filter[:, :, j], cmap='gray')
    pyplot.savefig("filters/" + model_name + "/matrices.png")

def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image():
    # We start from a gray image with some random noise
    img = Image.open('SW-PPEW/Prunus/screen_500x500_2021-12-06_00-13-35.png')
    img = np.asarray(img)
    img.resize((1, img_height, img_width, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return img


def visualize_filter(filter_index, img):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

n_of_filters = 32
#visualize_filters_matrices(layer)
img = initialize_image()
first_layer_model = keras.models.Model(inputs = model.inputs, outputs = model.layers[2].output)
first_layer_model.summary()
feature_maps = first_layer_model.predict(img)


square = 8
ix = 1
pyplot.figure(figsize=(17, 17))
for _ in range(4):
    for _ in range(square):
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps[0, :, :, ix-1])
        ix += 1

pyplot.show()
# for i in range(n_of_filters):
#     img1 = img.copy()
#     loss, img1 = visualize_filter(i, img1)
#     keras.preprocessing.image.save_img("filters/" + model_name + "/" + str(i) + ".png", img1)
