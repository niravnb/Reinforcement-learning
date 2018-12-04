# Taken some reference & idea from below github soucre: Castellini Jacopo. Some GridWorld environments for OpenAI Gym
# https://github.com/opocaj92/GridWorldEnvs
# Four room gird world enviromnment setup from Aharutyu git
# https://github.com/aharutyu/gym-gridworlds/blob/master/gridworlds/envs/four_rooms.py


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
import dill


# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NUM_ROOMS = 4

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
# COLORS = {0: [1, 1, 1], 4: [0.6, 0, 1],
#           -1: [0.4, 0.3, 1.0], -2: [0.0, 1.0, 0.0],
#           -3: [1.0, 0.0, 0.0], 3: [0.0, 1.0, 0.0],
#           -13: [0.5, 0.0, 0.5], 1: [0.6, 0.3, 0.0], 10: [0, 0, 0]}


COLORS = {0: [1, 1, 1], 4: [0.6, 0, 1],
          -1: [0.4, 0.3, 1.0], -2: [0.0, 1.0, 0.0],
          -3: [1.0, 0.0, 0.0], 3: [0.0, 1.0, 0.0],
          -13: [0.5, 0.0, 0.5], 1: [0.6, 0.3, 0.0], 10: [0.6, 0, 1]}


# COLORS = {0: [1, 1, 1], 4: [1, 1, 1],
#           -1: [0.4, 0.3, 1.0], -2: [0.0, 1.0, 0.0],
#           -3: [1.0, 0.0, 0.0], 3: [1.0, 1.0, 1.0],
#           -13: [0.5, 0.0, 0.5], 1: [0.6, 0.3, 0.0], 10: [1, 1, 1]}


class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, terminal_reward=1.0,
                 move_reward=0):
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
        self.terminal_reward = terminal_reward
        self.steps = 0
        self.dis_return = 0
        self.figurecount = 0
        self.figtitle = None
        self.done = False
        self.optimal_policy = None
        self.draw_arrows = False
        self.first_time = True
        self.state = None
        self.state_coord = None
        self.my_state = None
        self.Q = None
        self.V = None
        self.epsilon = None
        self.options_length = 1
        self.alpha = None
        self.gamma = None
        self.draw_circles = None
        self.hallway_reward = 0
        self.options_Q = None
        self.options_Q_hat = None
        self.intra_options = None
        self.options_poilcy = None
        # self._my_init

        self.room_sizes = [[5, 5], [6, 5], [4, 5], [5, 5]]
        self.pre_hallways = [
            {tuple([2, 4]): [RIGHT, 0], tuple([4, 1]): [DOWN, 3]},
            {tuple([2, 0]): [LEFT, 0], tuple([5, 2]): [DOWN, 1]},
            {tuple([0, 2]): [UP, 1], tuple([2, 0]): [LEFT, 2]},
            {tuple([3, 4]): [RIGHT, 2], tuple([0, 1]): [UP, 3]},
        ]
        self.hallway_coords = [[2, 5], [6, 2], [2, -1], [-1, 1]]
        self.hallways = [  # self.hallways[i][j] = [next_room, next_coord] when taking action j from hallway i#
                        [[0, self.hallway_coords[0]], [1, [2, 0]], [
                            0, self.hallway_coords[0]], [0, [2, 4]]],
                        [[1, [5, 2]], [1, self.hallway_coords[1]], [
                            2, [0, 2]], [1, self.hallway_coords[1]]],
                        [[2, self.hallway_coords[2]], [2, [2, 0]], [
                            2, self.hallway_coords[2]], [3, [3, 4]]],
                        [[0, [4, 1]], [3, self.hallway_coords[3]], [
                            3, [0, 1]], [3, self.hallway_coords[3]]]
        ]
        self.options = []

        self.offsets = [0] * (NUM_ROOMS + 1)
        for i in range(NUM_ROOMS):
            self.offsets[i + 1] = self.offsets[i] + \
                self.room_sizes[i][0] * self.room_sizes[i][1] + 1
        self.n_states = self.offsets[4] + 1
        self.absorbing_state = self.n_states

        # self.goal = [2, [1, 2]]
        self.goal = [1, [6, 2]]
        self.terminal_state = self.encode(self.goal)

        self.noise = 1/3
        self.step_reward = 0.0
        self.terminal_reward = 1.0
        self.bump_reward = 0.0

        # start state random location in start room
        start_room = 0
        sz = self.room_sizes[start_room]
        # self.offsets[start_room] + np.random.randint(sz[0]*sz[1] - 1)
        self.start_state = self.coord2ind([0, 0], sizes=self.room_sizes[0])
        self.state_coord = [1, 1]
        self.my_state = self.my_decode(self.start_state)
        # self._reset()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(self.n_states)  # with absorbing state
        self.n_actions = 6

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

        if mapname == "map1.txt" or mapname == "map3.txt":
            self.goal = [1, [6, 2]]
        else:
            self.goal = [2, [1, 2]]
        self.terminal_state = self.encode(self.goal)

        self.observation = self._gridmap_to_observation()


    def _step(self, action):
        assert self.action_space.contains(action)
        observation = self.state
        a = action

        if self.state == self.terminal_state:
            self.state = self.absorbing_state
            self.done = True
            return self.state, self._get_reward(), self.done, None

        if action < 4: # if primitive action
            self.options_length = 1 # set option lenght to 1
            options_termiantion_indexs = [[25, 103], [56, 25], [56, 77], [77, 103]]

            in_hallway = self.in_hallway_index()
            [room, coord] = self.decode(self.state, in_hallway=in_hallway)
            room2 = room
            coord2 = coord

            if in_hallway:
                options_termiantion_indexs = [[56, 103], [77, 25], [56, 103], [25, 77]]

            start_room = room2

            # Taking other 3 action with 1/3 probability
            if np.random.binomial(1, 1/3, 1)[0] == 1:
                temp_action = [0, 1, 2, 3]
                temp_action.remove(action)
                chosen = np.random.choice(3, 1, p=[1/3, 1/3, 1/3])[0]
                action = temp_action[chosen]
                

            if in_hallway:  # hallway action
                [room2, coord2] = self.hallways[room][action]

            elif tuple(coord) in self.pre_hallways[room].keys():
                hallway_info = self.pre_hallways[room][tuple(coord)]
                if action == hallway_info[0]:
                    room2 = hallway_info[1]
                    coord2 = self.hallway_coords[room2]
                else:  # normal action
                    [row, col] = coord
                    [rows, cols] = self.room_sizes[room]
                    if action == UP:
                        row = max(row - 1, 0)
                    elif action == DOWN:
                        row = min(row + 1, rows - 1)
                    elif action == RIGHT:
                        col = min(col + 1, cols - 1)
                    elif action == LEFT:
                        col = max(col - 1, 0)
                    coord2 = [row, col]

            else:  # normal action
                [row, col] = coord
                [rows, cols] = self.room_sizes[room]
                if action == UP:
                    row = max(row - 1, 0)
                elif action == DOWN:
                    row = min(row + 1, rows - 1)
                elif action == RIGHT:
                    col = min(col + 1, cols - 1)
                elif action == LEFT:
                    col = max(col - 1, 0)
                coord2 = [row, col]

            new_state = self.encode([room2, coord2])
            self.state = new_state
            self.state_coord = coord2
            self.my_state = self.my_decode(self.state)

            reward = self._get_reward(new_state=new_state)


            if self.intra_options: # if running intra option then update Q and Q_hat values accordingly
                # a = action
                next_observation = new_state
                O1 = dill.load(open('P_O1.pkl', 'rb')) # loading option optimal policy
                # loading option optimal policy
                O2 = dill.load(open('P_O2.pkl', 'rb'))

                update_options = []
                if O1[observation] == a:
                    update_options.append(4)
                if O2[observation] == a:
                    update_options.append(5)

                # Intra option Q learning update
                self.options_Q[observation][a] += self.alpha*(reward + self.gamma
                                                              * self.options_Q_hat[next_observation][a] - self.options_Q[observation][a])
                max_o = np.max(self.options_Q[next_observation])
                if (self.terminal_state == next_observation) or (self.absorbing_state == next_observation):
                    beta_s = 1
                else:
                    beta_s = 0
                self.options_Q_hat[next_observation][a] = (
                    1 - beta_s)*self.options_Q[next_observation][a] + beta_s*max_o

                for o in update_options:
                    self.options_Q[observation][o] += self.alpha*(reward + self.gamma
                                                                  * self.options_Q_hat[next_observation][o] - self.options_Q[observation][o])
                    max_o = np.max(self.options_Q[next_observation])
                    if options_termiantion_indexs[start_room][o-4] == next_observation:
                        beta_s = 1
                    else:
                        beta_s = 0
                    self.options_Q_hat[next_observation][o] = (
                        1 - beta_s)*self.options_Q[next_observation][o] + beta_s*max_o

            return new_state, reward, self.done, None
        else: # if option then calling execute option
            return self._execute_options(action)

    def _execute_options(self, action):
        assert self.action_space.contains(action)

        options_termiantion_indexs = [[25, 103], [56, 25], [56, 77], [77, 103]]

        in_hallway = self.in_hallway_index()
        [room, coord] = self.decode(self.state, in_hallway=in_hallway)
        start_room = room

        if in_hallway:
            options_termiantion_indexs = [
                [56, 103], [77, 25], [56, 103], [25, 77]]

        option_terminal_state_index = options_termiantion_indexs[start_room][action-4]

        total_reward = 0
        total_steps = 0
        new_room = start_room

        while (self.state != option_terminal_state_index) and (self.state != self.terminal_state) and (start_room == new_room):
            observation = self.state
            a = self.options_poilcy[observation] # finding action to take according to option policy
            next_observation, reward, self.done, _ = self._step(a) # taking action

            if total_steps > 0: # updating option total reward
                total_reward += reward*(0.9**total_steps)
            else:
                total_reward += reward
            total_steps += 1

            if self.done: 
                break

            in_hallway = self.in_hallway_index()
            [room, coord] = self.decode(
                next_observation, in_hallway=in_hallway)
            new_room = room

        self.options_length = total_steps
        return self.state, total_reward, self.done, None

    def _get_reward(self, new_state=None):
        if self.done:
            return self.terminal_reward

        reward = self.step_reward

        if self.in_hallway_index():
            reward = self.hallway_reward

        return reward

    def at_border(self):
        [row, col] = self.ind2coord(self.state)
        return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

    def _reset(self):
        if self.first_time:
            self._my_init(self.mapFile)
            self.state = self.coord2ind([0, 0], sizes=self.room_sizes[0])
        self.done = False
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return

        self.observation = self._gridmap_to_observation()
        img = self.observation
        fig = plt.figure(1, figsize=(10, 8), dpi=60,
                         facecolor='w', edgecolor='k')
        fig.canvas.set_window_title(self.figtitle)
        plt.clf()
        plt.xticks(np.arange(0, 14, 1))
        plt.yticks(np.arange(0, 14, 1))
        plt.grid(True)
        plt.title("Four room grid world\nStart:Blue, Goal:Green", fontsize=20)
        # plt.xlabel("Steps: %d, Reward: %f" %(self.steps + 1, self.dis_return), fontsize=10)

        plt.imshow(img, origin="upper", extent=[0, 13, 0, 13])
        fig.canvas.draw()

        if self.draw_arrows: # For drawing arrows of optimal policy
            fig = plt.gcf()
            ax = fig.gca()
            for i in range(len(self.optimal_policy)):
                if (i != self.terminal_state) and (i < 104):
                    centre = self.my_decode(i)
                    y = 12-centre[0]
                    x = centre[1]
                    if self.optimal_policy[i] > 3:
                        #  style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
                        if self.optimal_policy[i] == 4:
                            ax.text(x+0.3, y+0.3, 'O1', fontweight='bold')
                        else:
                            ax.text(x+0.3, y+0.3, 'O2', fontweight='bold')
                    elif self.optimal_policy[i] == -1:
                        ax.text(x+0.3, y+0.3, 'NE', fontweight='bold')
                    else:
                        self._draw_arrows(x, y, self.optimal_policy[i])
            plt.title("Four room Grid World learned Optimal Policy\nO1: Hallway Option 1 (clockwise) O2: Hallway Option 2 \n NE: State Not Explored", fontsize=15)
            plt.savefig(self.figtitle+"_Arrows.png")

        if self.draw_circles: # for drawing circles for V values
            fig = plt.gcf()
            ax = fig.gca()
            centre = []
            for i in range(len(self.V)-1):
                centre.append(self.my_decode(i))

            for i in range(len(self.V)-1):
                value = self.V[i]
                centre = self.my_decode(i)
                circle1 = plt.Circle(
                    (centre[1] + 0.5, 12 - centre[0] + 0.5), value, color='k')
                ax.add_artist(circle1)
            plt.title("Four room Grid World learned State Values\n Start: Blue, Goal: Green", fontsize=15)
            plt.savefig(self.figtitle+"_Circles.png")

        plt.pause(0.00001)  # 0.01

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


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _gridmap_to_observation(self):
        grid_map = self._read_grid_map()
        obs_shape = [13, 13, 3]
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    if [i, j] == self.my_state:
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

    def ind2coord(self, index, sizes=None):
        if sizes is None:
            sizes = [self.n]*2
        [rows, cols] = sizes

        assert(index >= 0)

        row = index // cols
        col = index % cols

        return [row, col]

    def coord2ind(self, coord, sizes=None):
        if sizes is None:
            sizes = [self.n]*2

        [rows, cols] = sizes
        [row, col] = coord

        assert(row < rows)
        assert(col < cols)

        return row * cols + col

    def in_hallway_index(self, index=None):
        if index is None:
            index = self.state
        return index in [offset - 1 for offset in self.offsets]

    def in_hallway_coord(self, coord):
        return coord in self.hallway_coords

    def encode(self, location, in_hallway=None):
        [room, coord] = location
        if in_hallway is None:
            in_hallway = self.in_hallway_coord(coord)

        if in_hallway:
            return self.offsets[room + 1] - 1
        # maybe have hallways as input
        ind_in_room = self.coord2ind(coord, sizes=self.room_sizes[room])
        return ind_in_room + self.offsets[room]

    def decode(self, index, in_hallway=None):
        if in_hallway is None:
            in_hallway = self.in_hallway_index(index=index)

        room = [r for r, offset in enumerate(
            self.offsets[1:5]) if index < offset][0]
        if in_hallway:
            coord_in_room = self.hallway_coords[room]
        else:
            coord_in_room = self.ind2coord(
                index - self.offsets[room], sizes=self.room_sizes[room])
        return room, coord_in_room  # hallway

    def my_decode(self, index, in_hallway=None):
      room, coord_in_room = self.decode(index, in_hallway)
      my_room_offset = [[1, 1], [1, 7], [8, 7], [7, 1]]
      return [sum(pair) for pair in zip(coord_in_room, my_room_offset[room])]

    # Creates epsilon greedy policy
    def policy_fn(self, observation):
        A = np.ones(4, dtype=float) * self.epsilon / 4

        Q_val = self.Q[observation]
        Q_primitive_val = Q_val[0:4]
        # Taking random action if all are same
        if np.allclose(Q_primitive_val, Q_primitive_val[0]):
            best_action = np.random.choice(
                4, 1, p=[1/4, 1/4, 1/4, 1/4])[0]
        else:
            best_action = np.argmax(Q_primitive_val)

        A[best_action] += (1.0 - self.epsilon)
        return A

