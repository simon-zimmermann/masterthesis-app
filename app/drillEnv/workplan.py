from collections import deque

from drillEnv.workitem import Workitem
import drillEnv.config as config


class Workplan:
    def __init__(self):
        self.plan = deque[Workitem]()

    def random_init(self):
        self.plan.clear()
        # init with random workitems to fill the observed length
        while len(self.plan) < config.WORKPLAN_LENGTH:
            self.plan.append(Workitem.generate_random())

    def observe(self) -> int:
        self_as_int = 0
        tmp_list = list(self.plan)
        for i in range(config.WORKPLAN_LENGTH):
            self_as_int += tmp_list[i].observe() * (Workitem.num_discrete_states() ** i)
        # print(f"observe into {self_as_int}")
        return self_as_int

    def unobserve(observation: int) -> 'Workplan':
        # print(f"unobserve from {observation}")
        workplan = Workplan()
        for i in range(config.WORKPLAN_LENGTH):
            workplan.plan.append(Workitem.unobserve(observation % Workitem.num_discrete_states()))
            observation //= Workitem.num_discrete_states()
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
        array = ", ".join([str(item) for item in self.plan])
        return f"[{array}]"

    def num_discrete_states():
        return Workitem.num_discrete_states() ** config.WORKPLAN_LENGTH
