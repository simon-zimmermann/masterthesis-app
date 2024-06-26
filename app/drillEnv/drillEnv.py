import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from typing import TYPE_CHECKING

import drillEnv.config as config
from drillEnv.workplan import Workplan
from drillEnv.action import DrillAction
if TYPE_CHECKING:  # stupid circular imports
    from drillEnv.logger import DrillEnvDataLogger


class DrillEnv(Environment):
    def __init__(self, gamma: int, logger: 'DrillEnvDataLogger'):
        self.state = None  # the state to be passed to the agent
        self.action: DrillAction = None  # the last action performed by the agent
        self.workplan: Workplan = None
        self.remainingLife = config.DRILL_MAX_LIFE
        self.waitsteps = 0  # how many steps we have to wait until we can perform the next action. used for changing the drill bit
        self.logger = logger

        # MDP properties
        observation_space = spaces.Discrete((config.DRILL_MAX_LIFE + 1) * Workplan.num_discrete_states())
        action_space = spaces.Discrete(len(DrillAction))
        mdp_info = MDPInfo(observation_space, action_space, gamma, config.EVENT_HORIZON)

        logger.info("Initializing DrillEnv.")
        logger.info("Observation space size: %d" % (observation_space.values[-1] + 1))
        logger.info("Action space size: %d" % (action_space.values[-1] + 1))
        logger.info("Gamma: %f" % gamma)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self.workplan = Workplan()
            self.workplan.random_init()
            self.remainingLife = config.DRILL_MAX_LIFE
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
            #self.action = DrillAction.IDLE  # force idle action
            return self.state, reward, False, {}  # do not change the state, do not end the episode

        # check if the drill is broken
        if self.remainingLife <= 0:
            reward = config.REWARD_BROKEN
            return self.state, reward, True, {}  # do not change the state, end the episode

        # Perform the action
        match self.action:
            case DrillAction.CHANGE_BIT:
                # penalty for changing the drill bit
                reward = config.REWARD_FACTOR_CHANGE * (self.remainingLife - config.DRILL_ACCEPTED_CHANGE_LIFE)
                w1 = self.workplan.plan[0].intensity
                w2 = self.workplan.plan[1].intensity
                reward_penalty = -10 * (w1 + w2)  
                reward += reward_penalty
                self.remainingLife = config.DRILL_MAX_LIFE
                self.waitsteps = config.CHANGE_DURATION
            case DrillAction.REQUEUE_PART:
                self.workplan.requeue_part()
                reward = config.REWARD_REQUEUE  # small penalty for requeueing a part
            case DrillAction.DO_WORK:
                workitem = self.workplan.get_next_part()
                self.remainingLife -= config.DRILL_WORK_FACTOR * workitem.intensity
                # give a reward for doing work. proportional to the intensity of the work
                reward = config.REWARD_FACTOR_WORK * workitem.intensity
            # case DrillAction.IDLE:
            #     ignored_item = self.workplan.get_next_part()  # but do nothing with it
            #     # punish if ideling on a part, that could have been worked on
            #     reward = config.REWARD_FACTOR_IDLE * ignored_item.intensity

        self.state = self.observe()
        return self.state, reward, False, {}  # Absorbing states are handled above

    def stop(self):
        pass

    def render(self, record: bool):
        pass

    def observe(self) -> np.ndarray:
        return np.array([self.remainingLife + self.workplan.observe() * (config.DRILL_MAX_LIFE + 1)])

    def unobserve(state: np.ndarray):
        return {
            'remainingLife': state[0] % (config.DRILL_MAX_LIFE + 1),
            'workplan': Workplan.unobserve(state[0] // (config.DRILL_MAX_LIFE + 1))
        }

    def test_observation(self):
        self.reset()
        env_info_str = "life: %d, workplan: %s" % (self.remainingLife, self.workplan)
        state = self.observe()
        unobserved_info = DrillEnv.unobserve(state)
        unobserved_info_str = "life: %d, workplan: %s" % (unobserved_info["remainingLife"], unobserved_info["workplan"])
        assert env_info_str == unobserved_info_str, "DrillEnv could not unobserve its own observation"
        self.logger.info("Test complete. DrillEnv can unobserve its own observation.")
