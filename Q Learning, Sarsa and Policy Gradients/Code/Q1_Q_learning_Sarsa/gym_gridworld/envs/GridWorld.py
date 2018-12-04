# Taken some reference & idea from below github soucre: Some GridWorld environments for OpenAI Gym
# https://github.com/opocaj92/GridWorldEnvs 

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from PIL import Image
import os
import sys
from PIL import Image as Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0: [1.0, 1.0, 1.0], 4: [0.5, 0.5, 0.5],
          -1: [0.0, 0.0, 1.0], -2: [0.0, 1.0, 0.0],
          -3: [1.0, 0.0, 0.0], 3: [1.0, 0.0, 1.0],
          -13: [0.5, 0.0, 0.5], 1: [0.0, 0.0, 0.0], 10: [1, 0.6, 0.4]}


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, terminal_reward=10.0,
                 move_reward=0, puddle1_reward=-1.0,
                 puddle2_reward=-2.0, puddle3_reward=-3.0, westerly_wind=True):
        self.n = None
        self.m = None
        self.bombs = []
        self.puddle1 = []
        self.puddle2 = []
        self.puddle3 = []
        self.walls = []
        self.goals = []
        self.start = []
        self.mapFile = None
        self.saveFile = False

        self.move_reward = move_reward
        self.puddle1_reward = puddle1_reward
        self.puddle2_reward = puddle2_reward
        self.puddle3_reward = puddle3_reward
        self.westerly_wind = westerly_wind
        self.terminal_reward = terminal_reward
        self.steps = 0
        self.dis_return = 0
        self.figurecount = 0
        self.figtitle = None
        self.done = False
        self.optimal_policy = None
        self.draw_arrows = False
        self.first_time = True
        # self._my_init

    # Calling to load Map
    def _my_init(self, mapname):
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.file_name = os.path.join(this_file_path, mapname)
        with open(self.file_name, "r") as f:
            for i, row in enumerate(f):
                row = row.rstrip('\r\n')
                row = row.split(' ')
                if self.n is not None and len(row) != self.n:
                    raise ValueError(
                        "Map's rows are not of the same dimension...")
                self.n = len(row)
                for j, col in enumerate(row):
                    if col == "4":
                        self.start.append([i, j])  # self.n * i + j
                    elif col == "3":
                        self.goals.append([i, j])
                    elif col == "-1":
                        self.puddle1.append([i, j])
                    elif col == "-2":
                        self.puddle2.append([i, j])
                    elif col == "-3":
                        self.puddle3.append([i, j])
                    elif col == "1":
                        self.walls.append([i, j])
            self.m = i + 1
        if len(self.goals) == 0:
            raise ValueError("At least one goal needs to be specified...")
        self.n_states = self.n * self.m
        self.n_actions = 4
        state_index = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        self.state = self.start[state_index]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)
        self.observation = self._gridmap_to_observation()

    def _step(self, action):
        assert self.action_space.contains(action)
        if self.state in self.goals:  # Checking if reaached goal or not
            return self.state, 0.0, self.done, None
        else:
            # Taking correct action with 0.9 probability
            if np.random.binomial(1, 0.9, 1)[0] == 1:
                new_state = self._take_action(action)
            else:  # Taking other 3 action with 0.1/3 probability
                temp_action = [0, 1, 2, 3]
                temp_action.remove(action)
                action_taken = np.random.choice(3, 1, p=[1/3, 1/3, 1/3])[0]
                new_state = self._take_action(temp_action[action_taken])
                # print("\ntaken wrong action")

            reward = self._get_reward(new_state)
            self.state = new_state

            # Westerly wind moving addtional to east with 0.5 probability
            if self.westerly_wind and np.random.binomial(1, 0.5, 1)[0] == 1:
                new_state = self._take_action(1)
                reward += self._get_reward(new_state)
                self.state = new_state

            return self.state[0]*14 + self.state[1], reward, self.done, None

    def _reset(self):
        if self.first_time:
            self._my_init(self.mapFile)
        self.done = False
        state_index = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        self.state = self.start[state_index]
        # self._render(mode='human')
        return self.state[0]*14 + self.state[1]

    def _render(self, mode='human', close=False):
        if close:
            return

        self.observation = self._gridmap_to_observation()
        img = self.observation
        fig = plt.figure(1, figsize=(10, 8), dpi=60,
                         facecolor='w', edgecolor='k')
        fig.canvas.set_window_title(self.figtitle)
        plt.clf()
        plt.xticks(np.arange(0, 15, 1))
        plt.yticks(np.arange(0, 15, 1))
        plt.grid(True)
        plt.title("Start:Grey, Goal:Pink, Robot:Orange\n -1:Blue, -2:Green, -3:Red, Wall:Black\nWesterly wind: 0.5 Prob. Random State: 0.1 Prob.", fontsize=10)
        plt.xlabel("Steps: %d, Reward: %f" %
                   (self.steps + 1, self.dis_return), fontsize=10)

        plt.imshow(img, origin="upper", extent=[0, 14, 0, 14])
        fig.canvas.draw()

        # Drawing arrows
        if self.draw_arrows:
            for k, v in self.optimal_policy.items():
                y = 13-int(k/14)
                x = k % 14
                self._draw_arrows(x, y, v)
            plt.savefig(self.figtitle+"_Arrows.png")

        plt.pause(0.0001)  # 0.01

        # For creating video from tmp png files
        if self.saveFile:
            fname = '_tmp%05d.png' % self.figurecount
            plt.savefig(fname)
            plt.clf()

        return

    def _draw_arrows(self, x, y, direction):

        if direction == UP:
            x += 0.5
            dx = 0
            dy = 0.4
        if direction == DOWN:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if direction == RIGHT:
            y += 0.5
            dx = 0.4
            dy = 0
        if direction == LEFT:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0
        plt.arrow(x,  # x1
                  y,  # y1
                  dx,  # x2 - x1
                  dy,  # y2 - y1
                  facecolor='k',
                  edgecolor='k',
                  width=0.005,
                  head_width=0.4)

    def _take_action(self, action):
        row = self.state[0] 
        col = self.state[1] 
        if action == DOWN and [row + 1, col] not in self.walls:
            row = min(row + 1, self.m - 1)
        elif action == UP and [row - 1, col] not in self.walls:
            row = max(0, row - 1)
        elif action == RIGHT and [row, col + 1] not in self.walls:
            col = min(col + 1, self.n - 1)
        elif action == LEFT and [row, col - 1] not in self.walls:
            col = max(0, col - 1)
        new_state = [row, col]
        return new_state

    def _get_reward(self, new_state):
        if new_state in self.goals:
            self.done = True
            return self.terminal_reward
        elif new_state in self.puddle1:
            return self.puddle1_reward
        elif new_state in self.puddle2:
            return self.puddle2_reward
        elif new_state in self.puddle3:
            return self.puddle3_reward
        return self.move_reward

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _gridmap_to_observation(self):
        grid_map = self._read_grid_map()
        obs_shape = [14, 14, 3]
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    if [i, j] == self.state:
                        this_value = COLORS[10][k]
                    else:
                        this_value = COLORS[grid_map[i, j]][k]
                    observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)
                                * gs1, k] = this_value
        return observation

    def _read_grid_map(self):
        grid_map = open(self.file_name, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

