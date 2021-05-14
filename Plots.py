from PIL import Image
import matplotlib.pyplot as plt

class ContinuousPlots():

    def __init__(self):
        self.figure , self.axes = plt.subplots(1,1)

    def clear_figure(self):
        pass


if __name__ == "__main__":
    obj = ContinuousPlots()
    fig, ax = obj.figure, obj.axes
    ax.plot(1,2)
    plt.pause(2)
    fig.show()
