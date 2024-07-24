import numpy as np
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.algorithms.value import QLearning
from drillEnv.logger import DrillEnvDataLogger
from drillEnv.drillEnv import DrillEnv
import drillEnv.config as config


class DrillExperimentQ:
    def __init__(self, load_agent_name: None | str = None):
        np.random.seed()
        self.logger = DrillEnvDataLogger()
        self.logger.strong_line()
        self.logger.info('Experiment Algorithm: ' + QLearning.__name__)
        self.logger.strong_line()
        self.last_dataset = None  # Result dataset from last evaluation run

        # Environment
        gamma = config.TRAINING_GAMMA  # the closer to 1, the more the agent cares about the future rewards
        self.env = DrillEnv(gamma, self.logger)
        self.env.test_observation()  # sanity check

        # Policy
        self.epsilon = LinearParameter(value=1.0, threshold_value=1.0, n=1)
        self.pi = EpsGreedy(epsilon=self.epsilon)

        # Agent
        if (load_agent_name):
            self.logger.weak_line()
            self.logger.info("Loading agent from file: %s.msh" % load_agent_name)
            self.logger.weak_line()
            self.agent = QLearning.load(load_agent_name + ".msh")
        else:
            self.learning_rate = Parameter(value=config.TRAINING_LEARNING_RATE)
            self.agent = QLearning(self.env.info, self.pi, learning_rate=self.learning_rate)

        # MushroomRL Core
        self.core = Core(self.agent, self.env, callbacks_fit=[self.logger.log_step_callback])

    def train(self, n_episodes: int, print_plot=False):
        # initialize logger
        self.logger.init_training(self.epsilon, self.learning_rate)

        # Reset parameter counters, so they will start decreasing from the beginning
        self.epsilon._n_updates[0] = 0
        self.learning_rate._n_updates[0] = 0

        # re-initialize epsilon, so that it will be linearly decreasing across the whole episode length
        expected_sample_count = n_episodes * config.EVENT_HORIZON
        self.epsilon.__init__(value=1.0, threshold_value=0.05, n=(expected_sample_count * 0.5))

        # Log training info
        self.logger.weak_line()
        self.logger.info("Start Training for %d episodes" % n_episodes)
        self.logger.info("Expected sample count: %d" % expected_sample_count)
        self.logger.info("Epsilon: Linear decrease from 1.0 to 0.01")
        self.logger.info("Learning rate: constant at %f" % self.learning_rate.get_value())
        self.logger.weak_line()

        # Now train
        self.core.learn(n_episodes=n_episodes, n_steps_per_fit=1)

        if print_plot:
            self.logger.plot_training_log()

    def eval_performance(self, print_demo=False):
        n_episodes = 10
        self.logger.weak_line()
        self.logger.info("Start Evaluation for %d episodes" % n_episodes)
        self.logger.weak_line()
        dataset = self.core.evaluate(n_episodes=n_episodes, render=False)
        self.last_dataset = dataset

        # Log evaluation results
        self.logger.print_metrics(self.last_dataset, self.env.info.gamma)
        if print_demo:
            self.logger.log_dataset(self.last_dataset, 200)

    def save_agent(self, filename: str):
        self.agent.save(filename + ".msh", full_save=True)
