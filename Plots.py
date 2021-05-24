from PIL import Image
import matplotlib.pyplot as plt
from Predict import Predict
from Upload_Picture import take_picture_frames_in_video, upload_from_local
import numpy as np
import cv2


class ContinuousPlots:
    """
    Functions in this class take output from the Predict class's functions and plots their output
    and save the output to a .tiff file
    """

    def __init__(self):
        """
        Initialise the plot and Predict object.
        """
        self.figure, self.axes = plt.subplots(1, 1)
        self.tp = Predict()

    def upload_and_get_mask(self):
        """
        Saves the mask to a file, and also returns it as output.
        Binary /Otsu thresholding is done before creating the final mask.
        :return: PIL Image object of the mask
        """
        mask , array = self.tp.get_mask_from_image_upload()
        mask = np.array(mask) ; array = np.array(array)
        # Otsu's thresholding
        th, maskt = cv2.threshold(mask, 0.0, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        maskt = np.expand_dims(maskt, axis=2)
        # print(maskt.shape, array.shape)
        double = Image.fromarray(np.multiply(array, maskt))
        double.save("mask.tiff")
        return double

    def continuos_plots(self):
        """
        Plots the images/frames and its predicted mask in a certain interval as applied in the
        code. Currently the interval is hardcoded to 4 per second.
        :return: PIL Image object of the mask of the final image/frame taken of the video
        """
        fig, ax = self.figure, self.axes
        img_gen = take_picture_frames_in_video(self.tp.image_name)
        for array, cam in img_gen:
            mask = self.tp.get_mask_from_array(array)
            mask = np.array(mask)
            # Otsu's thresholding
            th, maskt = cv2.threshold(mask, 0, 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # ax.imshow(np.multiply(array, np.expand_dims(maskt, axis=2) ), cmap='Reds')
            ax.imshow(array)
            ax.imshow(maskt, alpha=0.5, cmap='OrRd')
            plt.pause(1 / 4 )
            ax.cla()

        mask , array = self.tp.get_mask_from_local_image(self.tp.image_name + '.png')
        mask = np.array(mask)
        th, maskt = cv2.threshold(mask, 0.0, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        maskt = np.expand_dims(maskt, axis=2)
        ax.imshow(array)
        ax.imshow(maskt, alpha=0.7, cmap='OrRd')
        fig.savefig("mask.tiff")
        plt.close(fig)


        double = Image.open("mask.tiff")
        return double

    def take_a_picture_and_get_mask(self):
        """
        This function is for the case where a single image is taken from default camera.
        :return: PIL Image object of the mask of the picture taken
        """
        input_image, maskimg = self.tp.get_mask_from_picture()
        mask = np.array(maskimg) ; array = np.array(input_image)
        # Otsu's thresholding
        th, maskt = cv2.threshold(mask, 0.0, 1.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        maskt = np.expand_dims(maskt, axis=2)
        # print(maskt.shape, array.shape)
        double = Image.fromarray(np.multiply(array, maskt))
        double.save("mask.tiff")
        return double


if __name__ == "__main__":
    pass




