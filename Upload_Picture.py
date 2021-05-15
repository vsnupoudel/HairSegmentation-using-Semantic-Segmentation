import cv2
from PIL import Image
import matplotlib.pyplot as plt
from Plots import ContinuousPlots
from Train_or_Predict import Train_or_Predict
import time
import numpy as np

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
        if k%256 == 13:
            # SPACE pressed
            img_name = "{}_{}.png".format(filename, img_counter)
            cv2.imwrite(img_name, frame)
            im = Image.open("{}_{}.png".format(filename, img_counter))
            print("{} written!".format(img_name))
            img_counter += 1
            break
    cam.release()
    cv2.destroyAllWindows()
    return img_name , im.size


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
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        if k%256 == 13:
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
            break
        time.sleep(1/10)
        img_name = "{}_{}.png".format(filename, img_counter)
        # print(frame)
        # cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        yield frame, cam

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_gen = take_picture_frames_in_video('video_image')
    obj = ContinuousPlots()
    fig, ax = obj.figure, obj.axes
    tp = Train_or_Predict()
    for array ,cam in img_gen:
        mask = tp.get_mask_from_array(array)
        print(mask)
    #     # im= Image.open(image)
    #     image = tp.get_mask_from_image_upload(image)
    #     image.save('mask.tiff')
    #     ax.imshow(image, cmap='gray')
    #     plt.pause(1)
    #     ax.cla()


# from PIL import Image
# import numpy as np
#
# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()



