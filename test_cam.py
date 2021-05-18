import cv2
cams_test = 500
for i in range(0, cams_test):
    cap = cv2.VideoCapture(i)
    test, frame = cap.read()
    if test:
        print("i : "+str(i)+" /// result: "+str(test))


    # cp = ContinuousPlots()
    # double = cp.continuos_plots()
    # plt.imshow(double)
    # # ax.imshow(maskt, cmap='gray')
    # plt.show()
    # Plot mask from upload
    cp =  ContinuousPlots()
    double = cp.upload_and_get_mask()
