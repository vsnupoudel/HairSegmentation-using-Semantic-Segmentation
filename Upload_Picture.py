import cv2
from PIL import Image

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
        # if k%256 == 27:
        #     # ESC pressed
        #     print("Escape hit, closing...")
        #     break
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

if __name__ == "__main__":
    take_picture('test')



