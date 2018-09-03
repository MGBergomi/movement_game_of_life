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
from utils import blockwise_view
import imageio
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

cmap = sns.light_palette("Navy", as_cmap=True)

class Preprocess:
    def __init__(self, path = None, cols = 10, rows = 10, writer = False):
        # fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.writer =  writer

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
        return frame, self.discretize_frame(frame, self.height, self.width)

    def process_video(self, update_game = 1, max_evolution_cycles = 5,
                      threshold_scale = 20, min_th = 10000):
        prev_frame = None
        counter = 0
        evolutions_counter = 0
        diffs = np.ones(10) * 1e10
        # tf, ta = plt.subplots(1,2)

        while self.cap.isOpened() and counter < 3000:
            ret, orig_frame = self.cap.read()
            if ret:
                orig_frame, frame = self.process_frame(orig_frame)

                if prev_frame is not None:
                    diff = self.l2_diff(frame, prev_frame, self.rows , self.cols)
                    diffs[counter%10] = np.median(diff)
                    threshold = np.max([threshold_scale * np.max(diffs), min_th])
                    xs, ys = self.clip_movement(diff, threshold = threshold)
                    if counter % update_game == 0:
                        grid = self.game.update(init = [xs, ys])
                    if evolutions_counter >= max_evolution_cycles:
                        evolutions_counter = 0
                        if diff.max() < threshold/10:
                            grid = self.game.reset(init = [[],[]])
                    grid = self.game.play()
                    evolutions_counter += 1
                    self.display(grid, orig_frame = orig_frame, save = self.writer)
                counter += 1
                print(counter)
                prev_frame = frame

        self.writer.close()

    @staticmethod
    def display(frame, orig_frame = None, save = None):
        target_size = (576,1024)
        frame = imresize(frame, target_size)
        frame = gaussian_filter(frame, sigma=5)
        if orig_frame is not None:
            # print(frame.shape)
            orig_frame = imresize(orig_frame, target_size)
            # print(orig_frame.shape)
            frame = cv.addWeighted(frame, 1, orig_frame, 1, 0)

        if save is None:
            cv.imshow('test',frame)
            cv.waitKey(10) & 0xFF == ord('q')
        else:
            save.append_data(frame)


    @staticmethod
    def to_gray(frame):
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    @staticmethod
    def discretize_frame(frame, height, width):
        # frame = np.squeeze(frame)
        return blockwise_view(frame, (height, width))

    @staticmethod
    def l2_diff(discretized_frame, discretized_prev_frame, rows, cols):
        """

        Params
        ------
        discretized_frame : np.ndarray
            [num_clipped_rectangles, heigth_rec, width_rec]
        """
        return np.sum(np.square(discretized_frame - discretized_prev_frame),
                      axis = (2,3)).reshape([rows, cols])


    @staticmethod
    def clip_movement(diff, threshold = 100, n = 50):
        # print(diff.max())
        # print(diff)
        res = np.where(diff > threshold)
        p = np.random.permutation(len(res[0]))
        ys = res[0][p]
        xs = res[1][p]
        if len(xs) > n:
            xs = xs[:n]
            ys = ys[:n]
        return xs, ys


if __name__ == "__main__":
    path = './test_2.mp4'
    writer =imageio.get_writer('test__.mp4', fps=25)
    #path = None
    p = Preprocess(path = path, cols = 90, rows = 40, writer = writer)
    f = p.process_video(threshold_scale = 17.5, min_th = 20000)
