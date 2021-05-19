import cv2
from Plots import ContinuousPlots
from tkinter import *
import matplotlib.pyplot as plt

root = Tk()
i = IntVar()  # Basically Links Any Radiobutton With The Variable=i.
r1 = Radiobutton(root, text="Video", value=1, variable=i)
r2 = Radiobutton(root, text="Upload", value=2, variable=i)
r3 = Radiobutton(root, text="cam test", value=3, variable=i)
#
"""
If both values where equal, when one of the buttons
are pressed all buttons would be pressed.
If a button is pressed its value is true, or 1.
If you want to acess the data from the
radiobuttons, use a if statment like
"""
if (i.get() == 1):
    cp = ContinuousPlots()
    double = cp.continuos_plots()
    plt.imshow(double)
    plt.show()
elif (i.get()==2):
    # Plot mask from upload
    cp = ContinuousPlots()
    double = cp.upload_and_get_mask()
else:
    cams_test = 500
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, frame = cap.read()
        if test:
            print("i : " + str(i) + " /// result: " + str(test))

r1.pack()
r2.pack()
r3.pack()
root.mainloop()


