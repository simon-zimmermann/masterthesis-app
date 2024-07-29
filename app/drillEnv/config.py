# Simulation
DRILL_MAX_LIFE = 60  # the maximum life of the drill
EVENT_HORIZON = 1000  # maximum length of an episode
INTENSITY_MAX = 3 # maximum intensity of a workpiece; minimum is 1
NUM_HOLES_MAX = 2 # maximum number of holes in a workpiece; minimum is 0
WORKPLAN_LENGTH = 5 # length of the workplan, influences the size of the state space massively
DRILL_WORK_FACTOR = 3  # how much life is lost per work done (intensity*num_holes)
# Rewards
REWARD_BROKEN = -10000  # penalty for breaking the drill
REWARD_REQUEUE = -100  # penalty for requeueing a part
REWARD_FACTOR_CHANGE = -50  # factor to determine reward based on remaining drill life
REWARD_PENALTY_CHANGE = -50  # factor to determine penalty based on workpiece at the time of change
REWARD_FACTOR_WORK = 100  # reward factor for doing work
DRILL_ACCEPTED_CHANGE_LIFE = 20  # how much life is acceptable to be lost when changing the drill bit
# Hyperparameters
TRAINING_GAMMA = 0.7  # the closer to 1, the more the agent cares about the future rewards
TRAINING_LEARNING_RATE = 0.3  # the learning rate of the Q-learning algorithm