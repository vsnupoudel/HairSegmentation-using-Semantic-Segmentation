import cv2
from Plots import ContinuousPlots
# from tkinter import *
from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt

def call_video():
    cp = ContinuousPlots()
    double = cp.continuos_plots()
    plt.imshow(double)
    plt.show()

def upload_from_device():
    # Plot mask from upload
    dp = ContinuousPlots()
    double = dp.upload_and_get_mask()
    plt.imshow(double)
    plt.show()

def take_a_picture():
    ep = ContinuousPlots()
    double = ep.take_a_picture_and_get_mask()
    plt.imshow(double)
    plt.show()

def  print_camera_ids():
    cams_test = 500
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, frame = cap.read()
        if test:
            print(" &&& =============================================================== &&& \n")
            print("i : " + str(i) + " /// result: " + str(test))
            print(" &&& =============================================================== &&& \n")


parent = tk.Tk()
my_boolean_var = tk.BooleanVar()
parent.geometry("500x300")
parent.title('Choose One :   1.Video    2.Upload from device    3. Take Picture   4.Camera Test' )

label = ttk.Label(parent, text= 'Choose One :\n\n1. Video\n\n2. Upload from device  '
                  '\n\n3. Take Picture\n\n4. Camera Test').place(x = 0,y = 0)

video_checkbutton = ttk.Button(parent,
    text= "Take a Video",  command=call_video).pack(padx=10, pady=10)

upload_checkbutton = ttk.Button(parent,
    text= "Upload from device",  command= upload_from_device).pack(padx=10, pady=10)

picture_checkbutton = ttk.Button(parent,
    text= "Take picture from Camera",  command=take_a_picture).pack(padx=10, pady=10)

cam_test_checkbutton = ttk.Button(parent,
    text= "Print camera Ids",  command=print_camera_ids).pack(padx=10, pady=10)


parent.mainloop()



