import random
from time import sleep
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

from mushroom_rl.environments import GridWorld
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_J
from tqdm import trange

from agent.agent import Action, Agent
from env.drill import Drill
from env.workplan import Workplan


def print_env_status(drill: Drill, workplan: Workplan, last_action: Action, reward: float):
    str = "life: %d,\taction: %s,\treward: %.02f,\tworkplan: %s" % (
        drill.remainingLife, last_action, reward, workplan.pretty_print())
    print(str)


def main():
    MAX_CHANGES = 10
    print("Running simulation for a maximum of %d drill changes" % MAX_CHANGES)
    random.seed()  # Make sure random numbers are actually random
    workplan = Workplan()
    drill = Drill()
    agent = Agent()
    cum_reward = 0

    i = 0  # Simualtion iteration counter
    while i < MAX_CHANGES:
        # Determine the next action
        action = agent.act(drill, workplan)
        # Calculate reward
        reward = 0
        if drill.remainingLife <= 0:
            reward = -100
            drill.change_drillbit()  # Emergency change
        elif action == Action.DRILL or action == Action.SKIP:
            reward = 1
        elif action == Action.CHANGE:
            holes_curr = workplan.next().num_holes
            reward = -0.1 * drill.remainingLife - holes_curr
        cum_reward += reward
        # Print status
        print_env_status(drill, workplan, action, reward)
        # Perform the action
        match action:
            case Action.CHANGE:
                drill.change_drillbit()
                i += 1
            case Action.SKIP:
                workplan.pop()
                pass
            case Action.DRILL:
                drill.do_drill_action(workplan.pop())

    print("Simulation ended after %d drill changes" % MAX_CHANGES)
    print("Cumulative reward: %.02f" % cum_reward)


def experiment():
    np.random.seed()

    mdp = GridWorld(width=4, height=4, goal=(2, 2), start=(0, 0))
    epsilon = Parameter(value=1.)
    policy = EpsGreedy(epsilon=epsilon)
    learning_rate = Parameter(value=.1)
    agent = QLearning(mdp.info, policy, learning_rate)
    core = Core(agent, mdp)

    logger = Logger('tutorial', results_dir='/tmp/logs', log_console=True)
    # core.learn(n_steps=10000, n_steps_per_fit=1)
    logger.info('Experiment started')
    logger.strong_line()

    dataset = core.evaluate(n_steps=100)
    J = np.mean(compute_J(dataset, mdp.info.gamma))  # Discounted returns
    R = np.mean(compute_J(dataset))  # Undiscounted returns
    logger.epoch_info(0, J=J, R=R)

    for i in trange(10):
        # Here some learning
        core.learn(n_steps=10000, n_steps_per_fit=1)
        sleep(0.5)
        dataset = core.evaluate(n_steps=100)
        sleep(0.5)
        J = np.mean(compute_J(dataset, mdp.info.gamma))  # Discounted returns
        R = np.mean(compute_J(dataset))  # Undiscounted returns

        # Here logging epoch results to the console
        logger.epoch_info(i + 1, J=J, R=R)

        # Logging the data in J.npy and E.npy
        logger.log_numpy(J=J, R=R)

        # Logging the best agent according to the best J
        logger.log_best_agent(agent, J)

    core.evaluate(n_steps=100, render=True)

    # Logging the last agent
    logger.log_agent(agent)

    # Log the last dataset
    logger.log_dataset(dataset)

    logger.info('Experiment terminated')

    # shape = agent.Q.shape
    # print("Q shape: ", shape)
    # q = np.zeros(shape)
    # for i in range(shape[0]):
    #    for j in range(shape[1]):
    #        state = np.array([i])
    #        action = np.array([j])
    #        q[i, j] = agent.Q.predict(state, action)
    # print(q)


if __name__ == "__main__":
    experiment()
#    main()
