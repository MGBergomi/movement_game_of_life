#!/usr/bin/env python

#  An implementation of Conway's Game of Life in Python.

#  Copyright (C) 2013 Christian Jacobs.

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
import sys
if sys.platform == "darwin":
   import matplotlib
   matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
import seaborn as sns
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

class GameOfLife:

   def __init__(self, H=10, W = 10, init = None):
      """ Set up Conway's Game of Life. """
      # Here we create two grids to hold the old and new configurations.
      # This assumes an H*H grid of points.
      # Each point is either alive or dead, represented by integer values of 1
      # and 0, respectively.
      self.H = H
      self.W = W
      if init is None:
         # self.old_grid = np.random.rand(H,W)
         # self.old_grid[self.old_grid >= 1] = 1
         # self.old_grid[self.old_grid < 1] = 0
         self.old_grid = np.zeros((H,W), dtype='i')
      else:
         self.reset(init)
      self.new_grid = np.zeros((H,W), dtype='i')

   def reset(self, init):
      self.old_grid = np.zeros((self.H,self.W), dtype='i')
      self.old_grid[init[1], init[0]] = 1
      self.new_grid = np.zeros((self.H,self.W), dtype='i')
      return self.old_grid

   def update(self, init):
      self.old_grid[init[1], init[0]] = 1
      return self.old_grid

   def live_neighbours(self, i, j, torus = False):
      """ Count the number of live neighbours around point (i, j). """
      s = 0 # The total number of live neighbours.
      # Loop over all the neighbours.
      for x in [i-1, i, i+1]:
         for y in [j-1, j, j+1]:
            if(x == i and y == j):
               continue # Skip the current point itself - we only want to count the neighbours!
            if(x != self.H and y != self.W):
               s += self.old_grid[x][y]
            # The remaining branches handle the case where the neighbour is off the end of the grid.
            # In this case, we loop back round such that the grid becomes a "toroidal array".
            if torus:
               if(x == self.H and y != self.W):
                  s += self.old_grid[0][y]
               elif(x != self.H and y == self.W):
                  s += self.old_grid[x][0]
               else:
                  s += self.old_grid[0][0]
      return s

   def play(self):
      """ Play Conway's Game of Life. """
      # Loop over each cell of the grid and apply Conway's rules.
      for i in range(self.H):
         for j in range(self.W):
            live = self.live_neighbours(i, j)
            if(self.old_grid[i][j] == 1 and live < 2):
               self.new_grid[i][j] = 0 # Dead from starvation.
            elif(self.old_grid[i][j] == 1 and (live == 2 or live == 3)):
               self.new_grid[i][j] = 1 # Continue living.
            elif(self.old_grid[i][j] == 1 and live > 3):
               self.new_grid[i][j] = 0 # Dead from overcrowding.
            elif(self.old_grid[i][j] == 0 and live == 3):
               self.new_grid[i][j] = 1 # Alive from reproduction.
      self.old_grid = self.new_grid.copy()
      return self.old_grid

if(__name__ == "__main__"):
   plt.ion()
   game = GameOfLife(H = 10, W = 30, init = None)
