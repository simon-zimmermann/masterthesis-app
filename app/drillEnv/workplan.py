from collections import deque

from drillEnv.workitem import Workitem


class Workplan:
    OBSERVED_LENGTH = 10

    def __init__(self):
        self.plan = deque[Workitem]()

    def random_init(self):
        self.plan.clear()
        # init with random workitems to fill the observed length
        while len(self.plan) < self.OBSERVED_LENGTH:
            self.plan.append(Workitem.generate_random())

    def observe(self):
        return [item.intensity for item in self.plan]
    
    def requeue_part(self):
        self.plan.rotate(1)

    def get_next_part(self):
        self.plan.append(Workitem.generate_random())
        return self.plan.popleft()

    def calculate_discrete_states():
        return Workplan.OBSERVED_LENGTH * Workitem.INTENSITY_STEPS
