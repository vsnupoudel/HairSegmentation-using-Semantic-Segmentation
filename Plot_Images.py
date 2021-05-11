import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import datetime, os


class SamplePictures(base_dir='/root/.keras/datasets/Figaro1k/'):

    def __init__(self):
        self.pic_index = 0

    def plot_nine_pics(self):
        self.pic_index += 9
        # Display input image #7
        fig, ax = plt.subplots(3, 3, figsize=(9, 9))
        fig1, ax1 = plt.subplots(3, 3, figsize=(9, 9))
        for i in range(3):
            for j in range(3):
                img = mpimg.imread(input_img_paths[self.pic_index])
                ax[i, j].imshow(img)
                img = mpimg.imread(target_img_paths[self.pic_index])
                ax1[i, j].imshow(img)
                self.pic_index += 1
        plt.show()

    def create_dirs(self):

        # Directory with the training pictures
        training_dir = os.path.join(self.base_dir, 'Original/Training')
        testing_dir = os.path.join(self.base_dir, 'Original/Testing')

        # Directory with our target pictures pictures
        training_mask_dir = os.path.join(self.base_dir, 'GT/Training')
        testing_mask_dir = os.path.join(self.base_dir, 'GT/Testing')

        input_img_paths = sorted(
            [os.path.join(training_dir, fname)
                for fname in os.listdir(training_dir)
                if fname.endswith(".jpg")
            ])
        target_img_paths = sorted(
            [
                os.path.join(training_mask_dir, fname)
                for fname in os.listdir(training_mask_dir)
                if fname.endswith("gt.pbm")
            ])

        test_img_paths = sorted([
            os.path.join(testing_dir, fname)
            for fname in os.listdir(testing_dir)
            if fname.endswith(".jpg")
        ])

        test_mask_paths = sorted([
            os.path.join(testing_mask_dir, fname)
            for fname in os.listdir(testing_mask_dir)
            if fname.endswith("gt.pbm")
        ])

        print("Number of samples:", len(input_img_paths))

        for input_path, target_path in zip(input_img_paths[:5], target_img_paths[:5]):
            print(input_path, "|", target_path)

        return input_img_paths, target_img_paths, test_img_paths, test_mask_paths
