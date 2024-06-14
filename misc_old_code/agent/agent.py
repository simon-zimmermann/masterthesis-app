from enum import Enum
from env.drill import Drill
from env.workplan import Workplan




class Agent:
    def __init__(self):
        self.name = "Manual Agent"

    def act(self, drill: Drill, worplan: Workplan):
        if drill.remainingLife <= 4:
            return Action.CHANGE
        if worplan.next().num_holes == 0:
            return Action.SKIP
        return Action.DRILL
