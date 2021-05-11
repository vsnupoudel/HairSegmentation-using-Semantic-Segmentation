import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from Upload_Picture import take_picture
from PIL import Image

# Using the model we trained using the hair dataset in Unet

def use_trained_model():
    model_up = load_model('93_Percent_HairSegmentation.h5')

    path = take_picture('my_picture')
    # r'C:/Users/admin/Downloads/young-asian-woman-beautiful-long-260nw-1675898824.jpg'
    img = load_img(path, target_size= (256,256))
    arr = tf.cast( img_to_array(img)/255.0 , tf.float32)
    arr = np.expand_dims(arr, 0)

    output = model_up.predict(arr)
    mask = np.squeeze(output, axis = (0,3) )
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # Convert to PIL Image and save
    Image.fromarray(mask).save('test.tif')


if __name__ == "__main__":
    use_trained_model()


