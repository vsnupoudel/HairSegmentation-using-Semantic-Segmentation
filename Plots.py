from PIL import Image
import matplotlib.pyplot as plt
from Train_or_Predict import Train_or_P
from Upload_Picture import take_picture_frames_in_video
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

class ContinuousPlots():

    def __init__(self):
        self.figure , self.axes = plt.subplots(1,1)

    def continuos_plots(self):
        fig, ax = self.figure, self.axes
        tp = Train_or_P()
        img_gen = take_picture_frames_in_video(tp.image_name)
        for array, cam in img_gen:
            mask = tp.get_mask_from_array(array)
            mask = np.array(mask)
            # Otsu's thresholding
            th , maskt = cv2.threshold(mask, 0.0, 1.0, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
            maskt = np.expand_dims(maskt, axis=2)
            ax.imshow( np.multiply( array , maskt ) )
            # ax.imshow(maskt, cmap='gray')
            plt.pause(1 / 4)
            ax.cla()

        mask = tp.get_mask_from_image_upload(tp.image_name + '.png')
        mask.save("mask.tiff")
        return mask


if __name__ == "__main__":
    cp = ContinuousPlots()
    mask = cp.continuos_plots()
    plt.imshow(mask, cmap='gray')
    plt.show()











