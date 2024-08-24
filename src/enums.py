from enum import Enum

class SampleMethod(Enum):
    RANDOM = 1
    PROXIMITY = 2

class SDFMethod(Enum):
    GRID = 1
    SAMPLE = 2

class SDFReconstructionMethod(Enum):
    MARCHING_CUBES = 1
    REACH_FOR_THE_SPHERES = 2
    REACH_FOR_THE_ARCS = 3
    ALL = 4

class VDFReconstructionMethod(Enum):
    GRADIENT = 1
    BARYCENTRIC = 2
    CENTROID_NORMAL = 3
    ALL = 4