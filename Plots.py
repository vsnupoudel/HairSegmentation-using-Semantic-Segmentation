from PIL import Image
import matplotlib.pyplot as plt
from Train_or_Predict import Train_or_P
from Upload_Picture import take_picture_frames_in_video
import numpy as np
import cv2

class ContinuousPlots():

    def __init__(self):
        self.figure , self.axes = plt.subplots(1,1)

    def continuos_plots(self):
        fig, ax = self.figure, self.axes
        tp = Train_or_P()
        img_gen = take_picture_frames_in_video(tp.image_name)
        for array, cam in img_gen:
            mask = tp.get_mask_from_array(array)
            ax.imshow(mask, cmap='gray')
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











