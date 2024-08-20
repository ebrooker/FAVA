
from enum import Enum, IntEnum, auto

class AXIS(IntEnum):
    I = 0
    J = 1
    K = 2

class EDGE(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

class GEOMETRY(Enum):
    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    POLAR = "polar"

class CARTESIAN(IntEnum):
    X = 0
    Y = 1
    Z = 2

class CYLINDRICAL(IntEnum):
    RADIUS = 0
    THETA = 1
    Z = 2

class SPHERICAL(IntEnum):
    RADIUS = 0
    THETA = 1
    PHI = 2

class POLAR(IntEnum):
    RADIUS = 0
    THETA = 1