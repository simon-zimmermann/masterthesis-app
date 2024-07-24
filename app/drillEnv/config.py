
# CHANGE_DURATION = 3  # how many steps it takes to change the drill bit
DRILL_MAX_LIFE = 60  # the maximum life of the drill
EVENT_HORIZON = 1000  # maximum length of an episode

REWARD_BROKEN = -10000  # reward for breaking the drill
REWARD_REQUEUE = -100  # reward for requeueing a part
REWARD_FACTOR_CHANGE = -50  # reward factor for changing the drill bit
REWARD_PENALTY_CHANGE = -50  # will be multiplied with the waiting workpiece
REWARD_FACTOR_WORK = 100  # reward factor for doing work
DRILL_ACCEPTED_CHANGE_LIFE = 20  # how much life is acceptable to be lost when changing the drill bit
DRILL_WORK_FACTOR = 3  # how much life is lost per work item intensity

TRAINING_GAMMA = 0.3  # the closer to 1, the more the agent cares about the future rewards
TRAINING_LEARNING_RATE = 0.5  # the learning rate of the Q-learning algorithm

INTENSITY_MAX = 3
NUM_HOLES_MAX = 2
WORKPLAN_LENGTH = 5