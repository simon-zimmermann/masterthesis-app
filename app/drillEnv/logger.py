from matplotlib import pyplot as plt
from mushroom_rl.core import Logger
import numpy as np
from mushroom_rl.utils.dataset import compute_metrics
from mushroom_rl.utils.parameters import Parameter
import os

from drillEnv.action import DrillAction
from drillEnv.drillEnv import DrillEnv
import drillEnv.config as config


class DrillEnvDataLogger(Logger):

    def __init__(self):
        self.reset_data()
        self.epsilon = Parameter(value=0)
        self.learning_rate = Parameter(value=0)

        super().__init__(DrillEnvDataLogger.__name__, results_dir="logs", log_console=True, use_timestamp=True)

    def reset_data(self):
        self.training_log_step = {  # While training, relevant information is logged here
            "end_life": [],
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

    def init_training(self, epsilon: Parameter, learning_rate: Parameter):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.reset_data()

    def log_dataset(self, dataset, limit: int):
        self.weak_line()
        self.info("Printing %d steps of the last dataset" % limit)
        #drillchange_remaining_steps = 0
        for i in range(min(limit, len(dataset))):
            data = dataset[i]
            # exploded = DrillExperimentQ.explode_eval_dataset_entry(data)
            end_state_obj = DrillEnv.unobserve(data[3])
            # check for idle due to drill bit change
            action = DrillAction(data[1][0])
            #if(drillchange_remaining_steps > 0):
            #    action = DrillAction.CHANGE_BIT
            #    drillchange_remaining_steps -= 1
            #elif(action == DrillAction.CHANGE_BIT):
            #    drillchange_remaining_steps = config.CHANGE_DURATION
            print("state after action: %12s, reward: %4d, life: %3d, workplan: %s" %
                  (action.name,
                   data[2],
                   end_state_obj["remainingLife"],
                   end_state_obj["workplan"]))

    def log_step_callback(self, data):
        data = data[0]
        end_state_obj = DrillEnv.unobserve(data[3])
        self.training_log_step["end_life"].append(end_state_obj["remainingLife"])
        # self.training_log_step["end_wp"].append(end_state_obj["workplan"])
        self.training_log_step["action"].append(data[1][0])
        self.training_log_step["reward"].append(data[2])
        self.training_log_step["epsilon"].append(self.epsilon.get_value())
        self.training_log_step["learning_rate"].append(self.learning_rate.get_value())
        self.training_log_step["absorbing"].append(data[4])
        self.training_log_step["last"].append(data[5])

        if (data[4] or data[5]):  # if a episode has finished, calculate stuff
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

    def print_metrics(self, dataset, gamma: float):
        min, max, mean, median, length = compute_metrics(dataset, gamma)
        self.info("Metrics:")
        self.info("\tMinimum score: %.2f" % min)
        self.info("\tMaximum score: %.2f" % max)
        self.info("\tMean score: %.2f" % mean)
        self.info("\tMedian Score: %.2f" % median)
        self.info("\tNumber of episodes: %d" % length)

    def plot_training_log(self):
        self.info("Plotting training data")
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(self.training_log_episode["min_life"], label="Min Life")
        ax[0].plot(self.training_log_episode["avg_life"], label="Avg Life")
        ax[0].legend()
        ax[1].plot(self.training_log_episode["cum_reward"], label="Cumulative Reward")
        ax[1].legend()
        ax[2].plot(self.training_log_episode["epsilon"], label="Epsilon")
        ax[2].plot(self.training_log_episode["learning_rate"], label="Learning Rate")
        ax[2].legend()
        plt.savefig(os.path.join(self._results_dir, "training_plot.png"))
        plt.show()
