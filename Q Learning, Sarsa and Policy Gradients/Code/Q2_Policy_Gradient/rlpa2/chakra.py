from gym import Env
from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np
from math import sqrt


class chakra(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))

        self._seed()
        self.viewer = None
        self.state = None
        self.done = False
        self.goal = np.array([0,0])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        #Fill your code here
        # Clipping Action to [-0.025,0.025]
        action = np.clip(action, -0.025, 0.025)

        # Checking if reaached goal or not
        if self.state[0] == self.goal[0] and self.state[1] == self.goal[1]:
            return np.array(self.state), 0.0, self.done, None
        else:
            new_state = self._take_action(action)
            reward = self._get_reward(new_state)
            self.state = new_state
        # Return the next state and the reward, along with 2 additional quantities : False, {}
        return np.array(self.state), reward, self.done, None

    def _take_action(self, action):
        new_state = self.state + action
        # Checking if new state is outside environment then reseting environment
        if new_state[0] > 1 or new_state[0] < -1 or new_state[1] > 1 or new_state[1] < -1:
          return self._reset()
        else:
            return np.array(new_state)

    def _get_reward(self, new_state):
        if new_state[0] == self.goal[0] and new_state[1] == self.goal[1]:
            self.done = True
            return 0
        else:
            dist = sqrt(new_state[0]**2 + new_state[1]**2)
            return -dist


    def _reset(self):
        while True:
            self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
            # Sample states that are far away
            if np.linalg.norm(self.state) > 0.9:
                break
        return np.array(self.state)

    # method for rendering

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            agent.add_attr(trans)
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)

        # self.trans.set_translation(0, 0)
        self.trans.set_translation(
            (self.state[0] + 1) / 2 * screen_width,
            (self.state[1] + 1) / 2 * screen_height,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


register(
    'chakra-v0',
    entry_point='rlpa2.chakra:chakra',
    timestep_limit=40,
)
