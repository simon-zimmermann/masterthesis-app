import numpy as np
import matplotlib.pyplot as plt
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter, ExponentialParameter, LinearParameter
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_metrics

from drillEnv.action import DrillAction
from drillEnv.drillEnv import DrillEnv


class DrillExperiment:
    def __init__(self):
        np.random.seed()
        self.logger = Logger(DrillExperiment.__name__, results_dir="logs")
        self.logger.strong_line()
        self.logger.info('Experiment Algorithm: ' + QLearning.__name__)
        self.logger.strong_line()
        self.last_dataset = None  # Result dataset from last evaluation run
        self.training_log_step = {  # While training, relevant information is logged here
            "end_life": [],
            "end_wp": [],
            "action": [],
            "reward": [],
            "epsilon": [],
            "learning_rate": [],
            "absorbing": [],
            "last": []
        }
        self.training_log_episode = {  # While training, relevant information is logged here
            "min_life": [],
            "avg_life": [],
            "cum_reward": [],
            "epsilon": [],
            "learning_rate": [],
        }

        # Environment
        gamma = 0.7
        self.env = DrillEnv(gamma, self.logger)
        self.env.test_observation()  # sanity check

        # Policy
        # self.epsilon = ExponentialParameter(value=0.5, exp=0.03, min_value=0.1)
        self.epsilon = LinearParameter(value=1.0, threshold_value=0.1, n=5000000)
        self.pi = EpsGreedy(epsilon=self.epsilon)

        # Agent
        # self.learning_rate = ExponentialParameter(value=.8, exp=0.05, min_value=0.1)
        self.learning_rate = Parameter(value=0.1)
        self.agent = QLearning(self.env.info, self.pi, learning_rate=self.learning_rate)

        # MushroomRL Core
        self.core = Core(self.agent, self.env, callbacks_fit=[self.agent_fit_callback])

    def train(self, n_episodes: int):
        # First reset parameters
        self.epsilon._n_updates[0] = 0
        self.learning_rate._n_updates[0] = 0
        # Then train
        self.logger.weak_line()
        self.logger.info("Start Training for %d episodes" % n_episodes)
        self.logger.weak_line()
        self.core.learn(n_episodes=n_episodes, n_steps_per_fit=1)

    def eval_perf(self, n_episodes: int):
        self.logger.weak_line()
        self.logger.info("Start Evaluation for %d episodes" % n_episodes)
        self.logger.weak_line()
        dataset = self.core.evaluate(n_episodes=n_episodes, render=False)
        self.last_dataset = dataset

    def explode_eval_dataset_entry(data):
        end_state_obj = DrillEnv.unobserve(data[3])
        ret = {
            "end_life": end_state_obj["remainingLife"],
            "end_wp": end_state_obj["workplan"],
            "action": data[1][0],
            "action_str": DrillAction(data[1][0]).name,
            "reward": data[2]
        }
        return ret

    def print_dataset(self, limit: int = 30):
        self.logger.weak_line()
        self.logger.info("Printing %d steps of the last dataset" % limit)
        for i in range(min(limit, len(self.last_dataset))):
            data = self.last_dataset[i]
            exploded = DrillExperiment.explode_eval_dataset_entry(data)
            print("state after action: %12s, reward: %4d, life: %3d, workplan: %s" %
                  (exploded["action_str"],
                   exploded["reward"],
                   exploded["end_life"],
                   exploded["end_wp"]))

    def print_metrics(self):
        min, max, mean, median, length = compute_metrics(self.last_dataset, self.env.info.gamma)
        self.logger.info("Metrics:")
        self.logger.info("\tMinimum score: %.2f" % min)
        self.logger.info("\tMaximum score: %.2f" % max)
        self.logger.info("\tMean score: %.2f" % mean)
        self.logger.info("\tMedian Score: %.2f" % median)
        self.logger.info("\tNumber of episodes: %d" % length)

    def plot_training_data(self):
        self.logger.info("Plotting training data")
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(self.training_log_episode["min_life"], label="Min Life")
        ax[0].plot(self.training_log_episode["avg_life"], label="Avg Life")
        ax[0].legend()
        ax[1].plot(self.training_log_episode["cum_reward"], label="Cumulative Reward")
        ax[1].legend()
        ax[2].plot(self.training_log_episode["epsilon"], label="Epsilon")
        ax[2].plot(self.training_log_episode["learning_rate"], label="Learning Rate")
        ax[2].legend()
        plt.show()

    def agent_fit_callback(self, data):
        data = data[0]
        end_state_obj = DrillEnv.unobserve(data[3])
        self.training_log_step["end_life"].append(end_state_obj["remainingLife"])
        #self.training_log_step["end_wp"].append(end_state_obj["workplan"])
        self.training_log_step["action"].append(data[1][0])
        self.training_log_step["reward"].append(data[2])
        self.training_log_step["epsilon"].append(self.epsilon.get_value())
        self.training_log_step["learning_rate"].append(self.learning_rate.get_value())
        self.training_log_step["absorbing"].append(data[4])
        self.training_log_step["last"].append(data[5])

        if(data[4] or data[5]): # if a episode has finished, calculate stuff
            self.training_log_episode["min_life"].append(min(self.training_log_step["end_life"]))
            self.training_log_episode["avg_life"].append(np.average(self.training_log_step["end_life"]))
            self.training_log_episode["cum_reward"].append(sum(self.training_log_step["reward"]))
            self.training_log_episode["epsilon"].append(self.epsilon.get_value())
            self.training_log_episode["learning_rate"].append(self.learning_rate.get_value())
            self.training_log_step = {
            "end_life": [],
            "end_wp": [],
            "action": [],
            "reward": [],
            "epsilon": [],
            "learning_rate": [],
            "absorbing": [],
            "last": []
        }
        pass
