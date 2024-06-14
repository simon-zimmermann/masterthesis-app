# from queue import Queue
import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces

from drillEnv.workplan import Workplan
from drillEnv.action import DrillAction


class DrillEnv(Environment):
    CHANGE_DURATION = 2  # how many steps it takes to change the drill bit

    def __init__(self):
        self.state = None  # the state to be passed to the agent
        self.workplan: Workplan
        self.remainingLife = 100
        self.waitsteps = 0  # how many steps we have to wait until we can perform the next action. used for changing the drill bit

        # MDP properties
        observation_space = spaces.Discrete(100 * Workplan.calculate_discrete_states())
        action_space = spaces.Discrete(len(DrillAction))
        horizon = 1000  # maximum length of an episode
        gamma = .2  # the closer to 1, the more the agent cares about the future rewards
        # dt = 0.5  # timestep for visualization
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self.workplan = Workplan()
            self.workplan.random_init()
            self.remainingLife = 100
            self.waitsteps = 0
            state = self.observe()

        self.state = state
        return self.state

    def step(self, action):
        action_obj: DrillAction = DrillAction(action[0])

        # check if drillbit is currently being changed
        if self.waitsteps > 0:
            self.waitsteps -= 1
            reward = 0  # do not give a reward for waiting
            self.print_state("FORCED_IDLE", reward)
            return self.state, reward, False, {}  # do not change the state, do not end the episode

        # check if the drill is broken
        if self.remainingLife <= 0:
            reward = -100
            self.print_state("DRILL_BROKE", reward)
            return self.state, reward, True, {}  # do not change the state, end the episode

        # Perform the action
        match action_obj:
            case DrillAction.CHANGE_BIT:
                self.remainingLife = 100
                self.waitsteps = self.CHANGE_DURATION
                reward = -50  # penalty for changing the drill bit
            case DrillAction.REQUEUE_PART:
                self.workplan.requeue_part()
                reward = -1  # small penalty for requeueing a part
            case DrillAction.DO_WORK:
                workitem = self.workplan.get_next_part()
                self.remainingLife -= workitem.intensity
                reward = 5 * workitem.intensity  # give a reward for doing work. proportional to the intensity of the work
            case DrillAction.IDLE:
                self.workplan.get_next_part()  # but do nothing with it
                reward = 0  # do not give a reward for idling

        self.print_state(action_obj.name, reward)
        self.state = self.observe()
        return self.state, reward, False, {}  # Absorbing states are handled above

    def stop(self):
        pass

    def render(self, record: bool):
        pass

    def observe(self) -> np.ndarray:
        return np.array([self.remainingLife] + self.workplan.observe())  # TODO this is not the correct observation

    def print_state(self, action: str, reward: float):
        # print("completed action: %12s, reward: %4d, life: %3d, workplan: %s" %
        #      (action, reward, self.remainingLife, [item.intensity for item in self.workplan.plan]))
        pass
