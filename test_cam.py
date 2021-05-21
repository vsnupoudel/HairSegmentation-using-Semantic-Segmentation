import cv2
from Plots import ContinuousPlots
from tkinter import *
import matplotlib.pyplot as plt

def selected(i):
    if (i == "Video"):
        cp = ContinuousPlots()
        double = cp.continuos_plots()
        plt.imshow(double)
        plt.show()
    elif (i == "Upload Picture"):
        # Plot mask from upload
        cp = ContinuousPlots()
        double = cp.upload_and_get_mask()
    else:
        cams_test = 500
        for i in range(0, cams_test):
            cap = cv2.VideoCapture(i)
            test, frame = cap.read()
            if test:
                print(" &&& =============================================================== &&& \n")
                print("i : " + str(i) + " /// result: " + str(test))
                print(" &&& =============================================================== &&& \n")

root = Tk()
root.geometry("500x500")
root.title('Choose One :-  1.Video  2.Upload  3.Camera Test')
options = ['Video','Upload Picture', 'Camera Test']
clicked = StringVar(root)

drop = OptionMenu(root, clicked, *options, command= selected)
drop.pack(pady=20)

root.mainloop()


