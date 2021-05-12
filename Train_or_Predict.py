import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from keras.models import load_model
from Upload_Picture import take_picture
from PIL import Image

# Using the model we trained using the hair dataset in Unet
class Train_or_Predict():

    def __init__(self, image_path='.'):
        self.image_path = image_path

    def get_mask_from_picture(self):
        #lod the pretrained model
        model_up = load_model('93_Percent_HairSegmentation.h5')
        # take picture function
        input_image, image_size = take_picture('my_picture')
        # resize input image
        arr = self.resize_input_to_model_size(input_image)
        #predict with model
        output = model_up.predict(arr)
        print('output', output)
        # get output image same size as input
        output_mask = self.resize_model_to_input_size(output, image_size)
        return output_mask

    def resize_input_to_model_size(self, input_image):
        img = load_img(input_image, target_size=(256, 256))
        arr = tf.cast(img_to_array(img) / 255.0, tf.float32)
        arr = np.expand_dims(arr, 0)
        return arr

    def resize_model_to_input_size(self, model_output_image, image_size):
        mask = np.squeeze(model_output_image, axis=(0))
        mask_img = array_to_img(mask).resize(image_size)
        # mask_img = Image.fromarray(mask).resize(image_size)
        return mask_img

    def plot_sample_pictures(self):
        pass

    def train_on_pictures(self, image_path):
        pass


if __name__ == "__main__":
    tp = Train_or_Predict()
    image = tp.get_mask_from_picture()
    image.save('mask.tiff')
    plt.imshow(image,cmap='gray')
    plt.show()

