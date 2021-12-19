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


def initialize_image():
    # We start from a gray image with some random noise
    img = Image.open('60-20-10-10/Pine/i_Pine60Prunus10Walnut10Eucal20/screen_500x500_2021-12-07_13-24-28.png')
    img = img.resize((img_height, img_width))
    img = np.asarray(img)
    img = np.array([img])


    return img


def visualize_filters_matrices(layer, n_of_filters):
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    pyplot.figure(figsize=(17, 17))
    for f_index in range(n_of_filters):
        filter = filters[:, :, :, f_index]
        for j in range(3):
            ax = pyplot.subplot(n_of_filters, 8, f_index + j + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(filter[:, :, j], cmap='gray')
    pyplot.savefig("filters/" + model_name + "/matrices.png")

def visualize_filters_on_img1(model):
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
            pyplot.imshow(feature_maps[0, :, :, ix-1] * 255)
            ix += 1

    pyplot.savefig("filters/" + model_name + "/filter_outputs.png")


def visualize_filters_on_img(model, layers):
    img = initialize_image()
    print(model.predict(img))
    outputs = [model.layers[i].output for i in layers]
    filter_layers_model = keras.models.Model(inputs = model.inputs, outputs = outputs)
    filter_layers_model.summary()
    feature_maps = filter_layers_model.predict(img)


    square = 8
    for i in range(len(layers)):
        ix = 1
        pyplot.figure(figsize=(17, 17))
        for _ in range(4):
            for _ in range(square):
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.imshow(feature_maps[i][0, :, :, ix-1] * 255)
                ix += 1

        pyplot.savefig("filters/" + model_name + "/filter_60201010_" + str(layers[i]) + "_outputs.png")


n_of_filters = 32
layers = [2, 9, 20, 32, 45, 59]
#visualize_filters_matrices(layer, n_of_filters)
visualize_filters_on_img(model, layers)

