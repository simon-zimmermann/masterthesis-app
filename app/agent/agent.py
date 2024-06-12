from enum import Enum
from env.drill import Drill
from env.workplan import Workplan


class Action(Enum):
    DRILL = 1
    SKIP = 2
    CHANGE = 3


class Agent:
    def __init__(self):
        self.name = "Manual Agent"

    def act(self, drill: Drill, worplan: Workplan):
        if drill.remainingLife <= 4:
            return Action.CHANGE
        if worplan.next().num_holes == 0:
            return Action.SKIP
        return Action.DRILL
