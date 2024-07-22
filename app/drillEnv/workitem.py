import random

import drillEnv.config as config


class Workitem:

    def __init__(self, intensity: int, num_holes: int):
        if intensity < 1 or intensity > config.INTENSITY_MAX:
            raise ValueError("Intensity must be between %d and %d" % (1, config.INTENSITY_MAX))
        if num_holes < 1 or num_holes > config.NUM_HOLES_MAX:
            raise ValueError("Num holes must be between %d and %d" % (0, config.NUM_HOLES_MAX))
        self.intensity = intensity
        self.num_holes = num_holes
        pass

    def observe(self) -> int:
        ret = (self.intensity - 1) * config.NUM_HOLES_MAX + (self.num_holes - 1)
        # print(f"observe from {str(self)} into {ret}")
        return ret

    def unobserve(observation: int) -> 'Workitem':
        intensity = observation // config.NUM_HOLES_MAX + 1
        num_holes = observation % config.NUM_HOLES_MAX + 1
        ret = Workitem(intensity, num_holes)
        # print(f"unobserve from {observation} into {str(ret)}")
        return ret

    def num_discrete_states():
        return config.INTENSITY_MAX * config.NUM_HOLES_MAX

    def __str__(self):
        return f"i{self.intensity};h{self.num_holes}"

    def generate_random():
        rand_intensity = random.randint(1, config.INTENSITY_MAX)
        rand_holes = random.randint(1, config.NUM_HOLES_MAX)

        return Workitem(rand_intensity, rand_holes)
