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

            cv2.imshow("frame", frame)


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