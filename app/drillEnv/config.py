
#CHANGE_DURATION = 3  # how many steps it takes to change the drill bit
DRILL_MAX_LIFE = 99  # the maximum life of the drill. 99 to mathe the maths easier
EVENT_HORIZON = 1000  # maximum length of an episode

REWARD_BROKEN = -100  # reward for breaking the drill
REWARD_REQUEUE = -10  # reward for requeueing a part
REWARD_REQUEUE = -10  # reward for requeueing a part
REWARD_FACTOR_CHANGE = -20  # reward factor for changing the drill bit
REWARD_FACTOR_WORK = 20  # reward factor for doing work
# REWARD_FACTOR_IDLE = -5  # reward factor for ignoring a part
DRILL_ACCEPTED_CHANGE_LIFE = 20  # how much life is acceptable to be lost when changing the drill bit
DRILL_WORK_FACTOR = 3  # how much life is lost per work item intensity

TRAINING_GAMMA = 0.5  # the closer to 1, the more the agent cares about the future rewards
TRAINING_LEARNING_RATE = 0.2  # the learning rate of the Q-learning algorithm

INTENSITY_MAX = 2
NUM_HOLES_MAX = 2
WORKPLAN_LENGTH = 6