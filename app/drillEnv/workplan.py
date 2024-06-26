from collections import deque

from drillEnv.workitem import Workitem


class Workplan:
    OBSERVED_LENGTH = 6

    def __init__(self):
        self.plan = deque[Workitem]()

    def random_init(self):
        self.plan.clear()
        # init with random workitems to fill the observed length
        while len(self.plan) < self.OBSERVED_LENGTH:
            self.plan.append(Workitem.generate_random())

    def observe(self) -> int:
        self_as_int = 0
        tmp_list = list(self.plan)
        for i in range(self.OBSERVED_LENGTH):
            self_as_int += tmp_list[i].intensity * Workitem.INTENSITY_STEPS ** i
        return self_as_int

    def unobserve(observation: int) -> 'Workplan':
        workplan = Workplan()
        for i in range(Workplan.OBSERVED_LENGTH):
            workplan.plan.append(Workitem(observation % Workitem.INTENSITY_STEPS))
            observation //= Workitem.INTENSITY_STEPS
        return workplan

    def each(self):
        for item in self.plan:
            yield item

    def requeue_part(self):
        self.plan.rotate(-1)

    def get_next_part(self):
        self.plan.append(Workitem.generate_random())
        return self.plan.popleft()

    def __str__(self):
        return str([item.intensity for item in self.plan])

    def num_discrete_states():
        return Workitem.INTENSITY_STEPS ** Workplan.OBSERVED_LENGTH
