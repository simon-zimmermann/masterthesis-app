import numpy as np
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces


class GridWorld2(Environment):
    def __init__(self):
        self.state = None
        self.height = 8
        self.width = 8
        self.start = (0, 0)
        self.goal = (7, 7)

        # MDP properties
        observation_space = spaces.Discrete(self.height * self.width)
        action_space = spaces.Discrete(4)
        horizon = 100  # maximum length of an episode
        gamma = .9  # the closer to 1, the more the agent cares about the future rewards
        dt = 0.5  # timestep for visualization
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        # Visualization
        self._viewer = Viewer(self.width, self.height, 500,
                              self.height * 500 // self.width)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            state = self.convert_to_int(self.start, self.width)

        self.state = state

        return self.state

    def step(self, action):
        state_grid = self.convert_to_grid(self.state, self.width)
        action = action[0]
        if action == 0:
            if state_grid[0] > 0:
                state_grid[0] -= 1
        elif action == 1:
            if state_grid[0] + 1 < self.height:
                state_grid[0] += 1
        elif action == 2:
            if state_grid[1] > 0:
                state_grid[1] -= 1
        elif action == 3:
            if state_grid[1] + 1 < self.width:
                state_grid[1] += 1

        # obstacles

        if np.array_equal(state_grid, self.goal):
            reward = 10
            absorbing = True
        else:
            reward = 0
            absorbing = False

        self.state = self.convert_to_int(state_grid, self.width)

        return self.state, reward, absorbing, {}

    def render(self, record=False):
        for row in range(1, self.height):
            for col in range(1, self.width):
                self._viewer.line(np.array([col, 0]),
                                  np.array([col, self.height]))
                self._viewer.line(np.array([0, row]),
                                  np.array([self.width, row]))

        goal_center = np.array([.5 + self.goal[1],
                                self.height - (.5 + self.goal[0])])
        self._viewer.square(goal_center, 0, 1, (0, 255, 0))

        start_grid = self.convert_to_grid(self.start, self.width)
        start_center = np.array([.5 + start_grid[1],
                                 self.height - (.5 + start_grid[0])])
        self._viewer.square(start_center, 0, 1, (255, 0, 0))

        state_grid = self.convert_to_grid(self.state, self.width)
        state_center = np.array([.5 + state_grid[1],
                                 self.height - (.5 + state_grid[0])])
        self._viewer.circle(state_center, .4, (0, 0, 255))

        frame = self._viewer.get_frame() if record else None

        self._viewer.display(.1)

        return frame

    def stop(self):
        self._viewer.close()

    @staticmethod
    def convert_to_grid(state, width):
        return np.array([state[0] // width, state[0] % width])

    @staticmethod
    def convert_to_int(state, width):
        return np.array([state[0] * width + state[1]])
