
import random


class Workitem:
    INTENSITY_MIN = 0
    INTENSITY_MAX = 3
    INTENSITY_STEPS = INTENSITY_MAX - INTENSITY_MIN + 1

    def __init__(self, intensity: int):
        if intensity < self.INTENSITY_MIN or intensity > self.INTENSITY_MAX:
            raise ValueError("Intensity must be between %d and %d" % (self.INTENSITY_MIN, self.INTENSITY_MAX))
        self.intensity = intensity
        pass

    def generate_random():
        return Workitem(random.randint(Workitem.INTENSITY_MIN, Workitem.INTENSITY_MAX))
