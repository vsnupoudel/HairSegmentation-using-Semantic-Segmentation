import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from keras.models import load_model
from PIL import Image

# Using the model we trained using the hair dataset in Unet
from Upload_Picture import take_picture
from Upload_Picture import upload_from_local


class Predict:
    """
    Class with functions to return mask from:
    1. uploaded picture objects
    2. numpy array objects,
    3. or from local image path
    """

    def __init__(self, image_path='.', image_name='video_image'):
        """
        :param image_path: string defaults to current working directory.
        :param image_name: string defaults to 'video_image.png'
        """
        self.image_path = image_path
        self.model_up = load_model('93_Percent_HairSegmentation.h5')
        self.image_name = image_name

    def get_mask_from_picture(self):
        """
        Takes pictures from current default camera
        :return: A PIL Image object
        :return: A PIL Image object of the mask predicted
        """
        # take picture function
        input_image, image_size = take_picture('my_picture')
        # resize input image
        arr = self.resize_input_to_model_size(input_image)
        #predict with model
        output = self.model_up.predict(arr)
        # get output image same size as input
        output_mask = self.resize_model_to_input_size(output, image_size)
        pil_image = Image.open(input_image)
        return pil_image, output_mask

    def get_mask_from_local_image(self, input_image):
        """
        Takes a input image path as parameter
        :param input_image: string, path to the input image
        :return: A PIL Image object
        :return: A PIL Image object of the mask predicted
        """
        read_image = Image.open(input_image)
        image_size = read_image.size
        # resize input image
        arr = self.resize_input_to_model_size(input_image)
        #predict with model
        output = self.model_up.predict(arr)
        # get output image same size as input
        output_mask = self.resize_model_to_input_size(output, image_size)
        return output_mask, read_image

    def get_mask_from_image_upload(self):
        """
        Takes pictures from local directory. Gives option for user to upload from drive.
        :return: A PIL Image object
        :return: A PIL Image object of the mask predicted
        """
        image_uploaded = upload_from_local()
        read_image = Image.open(image_uploaded)
        image_size = read_image.size
        # resize input image
        arr = self.resize_input_to_model_size(image_uploaded)
        #predict with model
        output = self.model_up.predict(arr)
        # get output image same size as input
        output_mask = self.resize_model_to_input_size(output, image_size)
        return output_mask , read_image

    def get_mask_from_array(self, array_input):
        """
        Takes a numpy array as input parameter.
        :param array_input: a numpy array of image
        :return: A PIL Image object of the mask predicted
        """
        image = Image.fromarray(array_input)
        image_size = (array_input.shape[1], array_input.shape[0])
        # resize input image
        resized = self.resize_input_array_to_model_size(image)
        #predict with model
        output = self.predict_fn(resized)
        # get output image same size as input
        output_mask = self.resize_model_to_input_size(output, image_size)
        return output_mask

    def resize_input_to_model_size(self, input_image):
        """
        :param input_image: PIL Image object
        :return: numpy array of resized image
        """
        img = load_img(input_image, target_size=(256,256))
        arr = tf.cast(img_to_array(img) / 255.0, tf.float32)
        arr = np.expand_dims(arr, 0)
        return arr

    def resize_input_array_to_model_size(self, input_image):
        """
        :param input_image: a numpy array of image
        :return: numpy array of resized image
        """
        img = input_image.resize(size=(256,256))
        arr = tf.cast(img_to_array(img) / 255.0, tf.float32)
        arr = np.expand_dims(arr, 0)
        return arr

    def resize_model_to_input_size(self, model_output_image, image_size):
        """
        :param model_output_image: Numpy array of image
        :param image_size: tuple of 2, desired output size
        :return: PIL Image object of the final output mask
        """
        mask = np.squeeze(model_output_image, axis=(0))
        mask_img = array_to_img(mask).resize(image_size)
        return mask_img

    def predict_fn(self, array_input):
        """
        :param array_input: numpy array of input image
        :return: numpy array after predicting with the model
        """
        return self.model_up.predict(array_input)

    def plot_sample_pictures(self):
        pass

    def train_on_pictures(self, image_path):
        pass


if __name__ == "__main__":
    tp = Predict()
    mask = tp.get_mask_from_image_upload()
    plt.imshow(mask)
    plt.show()


