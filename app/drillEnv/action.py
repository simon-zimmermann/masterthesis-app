from enum import Enum


class DrillAction(Enum):
    DO_WORK = 0  # actually process the part
    REQUEUE_PART = 1  # put the part back in the queue, simulating it being rerouted in a loop
    CHANGE_BIT = 2  # change the drill bit, resetting the life of the drill
    IDLE = 3  # do nothing, the part queue still moves
