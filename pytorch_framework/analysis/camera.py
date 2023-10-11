from orca.orca import die

from analysis.predictor import Predictor
import matplotlib.pyplot as plt
from analysis.my_profile import my_profile
import matplotlib
from helpers.viewer import Viewer
import cv2

matplotlib.use('TKAgg')

class main_loop:

    def __init__(self):
        self.predictor = Predictor('/home/alex/workspace/experiments/best_model/model_metric.pth.tar')

        # define a video capture object
        self.vid = cv2.VideoCapture(0)
        self.fig, self.ax = plt.subplots(1)
        self.viewer = Viewer()

        self.i_max = 10

    @my_profile
    def loop(self):
        i = 0
        while (True):
            # Capture the video frame
            # by frame
            ret, frame = self.vid.read()


            frame = self.predictor.opencv_img(frame)

            frame = self.viewer.convert_from_cv2_to_image(frame)
            # Display the image
            self.ax.imshow(frame)
            plt.pause(0.001)

            # frame  = cv2.resize(frame, (height, width), interpolation = cv2.INTER_AREA)
            # cv2.imshow('frame', frame2)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if i == self.i_max:
                break
            i = i + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def main(self):
        self.loop()
        # After the loop release the cap object
        self.vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main = main_loop()
    main.main()