import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import *
# import matplotlib.pyplot as plt
import time
# import numpy as np


def upload_from_local():
    filename = filedialog.askopenfilename(
        initialdir=r'C:\Users\admin\Pictures', title='Select a Image File'
    )
    return filename


def take_picture(filename):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k % 256 == 13:
            # SPACE pressed
            img_name = "{}_{}.png".format(filename, img_counter)
            cv2.imwrite(img_name, frame)
            im = Image.open("{}_{}.png".format(filename, img_counter))
            print("{} written!".format(img_name))
            img_counter += 1
            break
    cam.release()
    cv2.destroyAllWindows()
    return img_name, im.size


def take_video(filename):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Take an image hitting Enter, Hit Esc to go out")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow("Take an image hitting Enter, Hit Esc to go out", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        if k % 256 == 13:
            # SPACE pressed
            img_name = "{}_{}.png".format(filename, img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            yield img_name, cam

    cam.release()
    cv2.destroyAllWindows()


def take_picture_frames_in_video(filename):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Take an image hitting Enter, Hit Esc to go out")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow("Take an image hitting Enter, Hit Esc to go out", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            img_name = "{}.png".format(filename)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            break

        time.sleep(1 / 10)
        img_name = "{}_{}.png".format(filename, img_counter)
        print("{} array created!".format(img_name))
        img_counter += 1
        yield frame, cam

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    upload_from_local()
