import numpy as np
from mushroom_rl.environments import GridWorld
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter, ExponentialParameter
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.dataset import compute_metrics


from GridWorld2 import GridWorld2


def experiment():
    np.random.seed()

    logger = Logger(QLearning.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + QLearning.__name__)

    # MDP
    # mdp = generate_simple_chain(state_n=5, goal_states=[2], prob=.8, rew=1,
    #                            gamma=.9)
    env = GridWorld2()

    # Policy
    epsilon = ExponentialParameter(value=1.0, exp=0.99, min_value=0.1)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    learning_rate = Parameter(value=.2)
    agent = QLearning(env.info, pi, learning_rate=learning_rate)

    # Core
    core = Core(agent, env)

    # Initial policy Evaluation
    # dataset = core.evaluate(n_steps=100)
    # J = np.mean(dataset.discounted_return)
    # J = compute_metrics(dataset, env.info.gamma)
    # logger.info(f'metrics start: {J}')

    # core.evaluate(n_steps=100, render=True)
    # Train
    core.learn(n_steps=None, n_episodes=20000, n_steps_per_fit=1)

    # Final Policy Evaluation
    dataset = core.evaluate(n_episodes=5)
    # J = np.mean(dataset.discounted_return)
    J = compute_metrics(dataset, env.info.gamma)
    logger.info(f'metrics final: {J}')

    core.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    experiment()
#    main()
