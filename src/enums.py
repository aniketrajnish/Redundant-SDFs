from enum import Enum

class SampleMethod(Enum):
    RANDOM = 1
    PROXIMITY = 2

class ReconstructionMethod(Enum):
    MARCHING_CUBES = 1
    REACH_FOR_THE_SPHERES = 2
    REACH_FOR_THE_ARCS = 3
    ALL = 4