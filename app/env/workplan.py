from queue import Queue
import random


class Workitem:
    def __init__(self, num_holes):
        self.num_holes = num_holes
        pass


class Workplan:
    WORKPLAN_QUEUE_SIZE = 10

    def __init__(self):
        self.queue = Queue(Workplan.WORKPLAN_QUEUE_SIZE)
        for _ in range(Workplan.WORKPLAN_QUEUE_SIZE):
            self.add_new_workitem()

    def next(self) -> Workitem:
        return self.queue.queue[0]

    def pop(self):
        item = self.queue.get()
        self.add_new_workitem()
        return item

    def add_new_workitem(self):
        self.queue.put(Workitem(random.randint(0, 4)))

    def pretty_print(self):
        return str([item.num_holes for item in self.queue.queue])
