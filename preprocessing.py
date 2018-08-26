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
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

cmap = sns.light_palette("Navy", as_cmap=True)

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

    def process_video(self, update_game = 1, max_evolution_cycles = 5):
        prev_frame = None
        counter = 0
        evolutions_counter = 0
        # tf, ta = plt.subplots(1,2)

        while self.cap.isOpened():
            ret, orig_frame = self.cap.read()
            if ret:
                frame = self.process_frame(orig_frame)
                if prev_frame is not None:
                    diff = self.l2_diff(frame, prev_frame, self.rows , self.cols)
                    threshold = 70000
                    xs, ys = self.clip_movement(diff, threshold = threshold)
                    if counter % update_game == 0:
                        grid = self.game.update(init = [xs, ys])
                    if evolutions_counter >= max_evolution_cycles:
                        evolutions_counter = 0
                        if diff.max() < threshold/10:
                            grid = self.game.reset(init = [[],[]])

                    grid = self.game.play()
                    evolutions_counter += 1
                    self.display(grid, orig_frame = orig_frame)
                counter += 1

                prev_frame = frame


    @staticmethod
    def display(frame, orig_frame = None):
        target_size = (600,800)
        frame = imresize(frame, target_size)
        frame = gaussian_filter(frame, sigma=5)
        if orig_frame is not None:
            orig_frame = cv.cvtColor(orig_frame, cv.COLOR_BGR2GRAY)
            orig_frame = imresize(orig_frame, target_size)
            frame = cv.addWeighted(frame, 1, orig_frame, 1, 0)
        cv.imshow('test',frame)
        cv.waitKey(1) & 0xFF == ord('q')

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
    def clip_movement(diff, threshold = 100, n = 20):
        # print(diff.max())
        res = np.where(diff > threshold)
        # print(len(res[0]))
        p = np.random.permutation(len(res[0]))
        ys = res[0][p]
        xs = res[1][p]
        if len(xs) > n:
            xs = xs[:n]
            ys = ys[:n]
        return xs, ys

if __name__ == "__main__":
    path = './test.avi'
    # path = None
    p = Preprocess(path = path, cols = 60, rows = 20)
    p.process_video()
