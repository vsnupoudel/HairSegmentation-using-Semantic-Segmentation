import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from Upload_Picture import take_picture
from PIL import Image


# Using the model we trained using the hair dataset in Unet
class Train_or_Predict():

    def __init__(self, image_path='.'):
        self.image_path = image_path

    def get_mask_from_picture(self):
        model_up = load_model('93_Percent_HairSegmentation.h5')
        path = take_picture('my_picture')
        img = load_img(path, target_size=(256, 256))
        arr = tf.cast(img_to_array(img) / 255.0, tf.float32)
        arr = np.expand_dims(arr, 0)

        output = model_up.predict(arr)
        mask = np.squeeze(output, axis=(0, 3))
        # Convert to PIL Image and save
        return Image.fromarray(mask)

    def plot_sample_picture(self):
        pass

    def train_on_pictures(self, image_path):
        pass


if __name__ == "__main__":
    tp = Train_or_Predict()
    image = tp.get_mask_from_picture()
    image.save('mask.tif')
