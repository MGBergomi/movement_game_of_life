import sys
sys.path.append('./game-of-life')
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv
from game_of_life import GameOfLife
import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

class Preprocess:
    def __init__(self, path = None, cols = 10, rows = 10, display = "plt"):
        if display == "plt":
            plt.ion()
        self.path = path
        self.cols = cols
        self.rows = rows
        self.stream()
        self.video_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.video_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = self.video_h // self.rows
        self.width = self.video_w // self.cols
        self.crop_h = self.height * self.rows
        self.crop_w = self.width * self.cols
        self.game = GameOfLife(H = rows, W = cols, init = None)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, new_path):
        if new_path is None:
            # print("Ready for webcam streaming")
            new_path = 0
        else:
            assert os.path.isfile(new_path), ("Please provide a path to a file" +
                                          "or None to stream from your webcam")
        self._path = new_path

    def stream(self, func = None):
        self.cap = cv.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError("Something seems to be wrong with the video you" +
                             "selected")
    def process_frame(self, frame):
        frame = self.to_gray(frame)
        frame = cv.resize(frame, (self.crop_w, self.crop_h),
                          interpolation = cv.INTER_CUBIC)
        return self.discretize_frame(frame, self.height, self.width)

    def process_video(self, update_game = 20):
        prev_frame = None
        counter = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = self.process_frame(frame)
                if prev_frame is None:
                    prev_frame = frame
                else:
                    diff = self.l2_diff(frame, prev_frame, self.rows , self.cols)
                    xs, ys = self.clip_movement(diff)
                    # print(xs)
                    # print(ys)
                    prev_frame = frame
                if counter % update_game:
                    print("update!")
                    grid = self.game.reset(init = [xs, ys])
                else:
                    grid = self.game.play()
                print(grid, "---<<>>----")
                counter += 1
                prev_frame = frame
                self.display(grid)

    @staticmethod
    def display(frame):
        plt.imshow(frame)
        plt.pause(0.000001)

    @staticmethod
    def to_gray(frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    @staticmethod
    def discretize_frame(frame, height, width):
        frame = np.squeeze(frame)
        return np.reshape(frame, [-1, height, width])

    @staticmethod
    def l2_diff(discretized_frame, discretized_prev_frame, rows, cols):
        """

        Params
        ------
        discretized_frame : np.ndarray
            [num_clipped_rectangles, heigth_rec, width_rec]
        """
        return np.sum(np.square(discretized_frame - discretized_prev_frame),
                      axis = (1,2)).reshape([rows, cols])

    @staticmethod
    def clip_movement(diff, threshold = 250000, n = 20):
        res = np.where(diff > threshold)
        p = np.random.permutation(len(res[0]))
        ys = res[0][p]
        xs = res[1][p]
        if len(xs) > n:
            xs = xs[:n]
            ys = ys[:n]
        return xs, ys

if __name__ == "__main__":
    p = Preprocess(path = None, cols = 20, rows = 10)
    p.process_video()
