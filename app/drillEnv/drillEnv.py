# from queue import Queue
import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces

from drillEnv.workplan import Workplan
from drillEnv.action import DrillAction


class DrillEnv(Environment):
    CHANGE_DURATION = 2  # how many steps it takes to change the drill bit
    DRILL_MAX_LIFE = 99  # the maximum life of the drill. 99 to mathe the maths easier

    def __init__(self):
        self.state = None  # the state to be passed to the agent
        self.action: DrillAction = None  # the last action performed by the agent
        self.workplan: Workplan = None
        self.remainingLife = 100
        self.waitsteps = 0  # how many steps we have to wait until we can perform the next action. used for changing the drill bit

        # MDP properties
        observation_space = spaces.Discrete((DrillEnv.DRILL_MAX_LIFE + 1) * Workplan.num_discrete_states())
        print("Observation space size: ", observation_space.values[-1] + 1)
        action_space = spaces.Discrete(len(DrillAction))
        print("Action space size: ", action_space.values[-1] + 1)
        horizon = 1000  # maximum length of an episode
        gamma = .2  # the closer to 1, the more the agent cares about the future rewards
        # dt = 0.5  # timestep for visualization
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)
        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self.workplan = Workplan()
            self.workplan.random_init()
            self.remainingLife = DrillEnv.DRILL_MAX_LIFE
            self.waitsteps = 0
            state = self.observe()

        self.state = state
        return self.state

    def step(self, action):
        self.action: DrillAction = DrillAction(action[0])

        # check if drillbit is currently being changed
        if self.waitsteps > 0:
            self.waitsteps -= 1
            reward = 0  # do not give a reward for waiting
            self.action = DrillAction.IDLE  # force idle action
            return self.state, reward, False, {}  # do not change the state, do not end the episode

        # check if the drill is broken
        if self.remainingLife <= 0:
            reward = -100
            return self.state, reward, True, {}  # do not change the state, end the episode

        # Perform the action
        match self.action:
            case DrillAction.CHANGE_BIT:
                reward = -5 * self.remainingLife  # penalty for changing the drill bit
                self.remainingLife = DrillEnv.DRILL_MAX_LIFE
                self.waitsteps = self.CHANGE_DURATION
            case DrillAction.REQUEUE_PART:
                self.workplan.requeue_part()
                reward = -1  # small penalty for requeueing a part
            case DrillAction.DO_WORK:
                workitem = self.workplan.get_next_part()
                self.remainingLife -= workitem.intensity
                reward = 10 * workitem.intensity  # give a reward for doing work. proportional to the intensity of the work
            case DrillAction.IDLE:
                ignored_item = self.workplan.get_next_part()  # but do nothing with it
                reward = -10 * ignored_item.intensity  # punish if ideling on a part, that could have been worked on

        # print("state after action: %12s, reward: %4d, life: %3d, workplan: %s" %
        #       (DrillAction(action).name, reward, self.remainingLife, self.workplan))
        self.state = self.observe()
        return self.state, reward, False, {}  # Absorbing states are handled above

    def stop(self):
        pass

    def render(self, record: bool):
        pass

    def observe(self) -> np.ndarray:
        return np.array([self.remainingLife + self.workplan.observe() * (DrillEnv.DRILL_MAX_LIFE + 1)])

    def unobserve(state: np.ndarray):
        return {
            'remainingLife': state[0] % (DrillEnv.DRILL_MAX_LIFE + 1),
            'workplan': Workplan.unobserve(state[0] // (DrillEnv.DRILL_MAX_LIFE + 1))
        }

    def test_observation():
        env = DrillEnv()
        env.reset()
        env_info_str = "life: %d, workplan: %s" % (env.remainingLife, env.workplan)
        state = env.observe()
        unobserved_info = DrillEnv.unobserve(state)
        unobserved_info_str = "life: %d, workplan: %s" % (env.remainingLife, env.workplan)
        assert env_info_str == unobserved_info_str, "DrillEnv could not unobserve its own observation"
        print("DrillEnv can unobserve its own observation")
