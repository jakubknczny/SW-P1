import lime
from lime import lime_image
from PIL import Image
import numpy as np
import skimage
from skimage.segmentation.boundaries import mark_boundaries
import matplotlib.pyplot as plt
from tensorflow import keras

img_height = 448
img_width = 448

explainer = lime_image.LimeImageExplainer()


def initialize_image():
    # We start from a gray image with some random noise
    # img = Image.open(
    #   '../datasets/60-20-10-10/Pine/i_Pine60Prunus10Walnut10Eucal20/screen_500x500_2021-12-07_13-24-28.png')
    img = Image.open(
        '../datasets/60-20-10-10/Prunus/r_Pine10Prunus60Walnut10Eucal20/screen_500x500_2021-12-07_13-36-50.png')
    img = img.resize((img_height, img_width))
    img = np.asarray(img)
    print(img.shape)
    img = np.array([img])

    return img


img = initialize_image()

model_name = 'kcnn-01-32e'

model = keras.models.load_model('saturday/' + model_name + '.best.hdf5')
explanation = explainer.explain_instance(img[0], model.predict, top_labels=4, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(2, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp, mask))
plt.savefig('filters/kcnn-01-32e/explainer_60201010Prunus.png')
