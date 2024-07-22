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

        # check if the drill is broken
        if self.remainingLife <= 0:
            reward = config.REWARD_BROKEN
            self.state = self.observe()
            self.state = self.observe()
            return self.state, reward, True, {}  # do not change the state, end the episode

        # Perform the action
        match self.action:
            case DrillAction.CHANGE_BIT:
                # penalty for changing the drill bit
                reward = config.REWARD_FACTOR_CHANGE * (self.remainingLife - config.DRILL_ACCEPTED_CHANGE_LIFE)
                workitem = self.workplan.get_next_part()
                reward_penalty = config.REWARD_PENALTY_CHANGE * workitem.get_drill_wear()
                if reward_penalty == 0:
                    self.workplan.pop_next_part()  # skip part if no work needs to be done while drilling
                reward += reward_penalty
                self.remainingLife = config.DRILL_MAX_LIFE
            case DrillAction.REQUEUE_PART:
                self.workplan.requeue_part()
                reward = config.REWARD_REQUEUE  # small penalty for requeueing a part
            case DrillAction.DO_WORK:
                workitem = self.workplan.pop_next_part()
                self.remainingLife -= config.DRILL_WORK_FACTOR * workitem.get_drill_wear()
                # give a reward for doing work. proportional to the intensity of the work
                reward = config.REWARD_FACTOR_WORK * workitem.get_drill_wear()

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
        self.logger.info("Testing observation")
        self.remainingLife = 10
        env_info_str = "life: %d, workplan: %s" % (self.remainingLife, str(self.workplan))
        state = self.observe()
        unobserved_info = DrillEnv.unobserve(state)
        unobserved_info_str = "life: %d, workplan: %s" % (
            unobserved_info["remainingLife"], str(unobserved_info["workplan"]))
        self.logger.info(env_info_str)
        self.logger.info(unobserved_info_str)
        assert env_info_str == unobserved_info_str, "DrillEnv could not unobserve its own observation"
        self.logger.info("Test complete. DrillEnv can unobserve its own observation.")
