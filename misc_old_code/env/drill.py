from env.workplan import Workitem


class Drill:

    def __init__(self):
        self.remainingLife = 100
        pass

    def do_drill_action(self, workitem: Workitem):
        self.remainingLife -= workitem.num_holes

    def change_drillbit(self):
        self.remainingLife = 100
        pass
