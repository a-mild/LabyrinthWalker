import numpy as np

NORTH = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [0, 0, 0]]
)

NORTHEAST = np.array(
    [[1, 1, 1],
     [0, 0, 1],
     [0, 0, 1]]
)

EAST = np.array(
    [[0, 1, 1],
     [0, 0, 1],
     [0, 1, 1]]
)

SOUTHEAST = np.array(
    [[0, 0, 1],
     [0, 0, 1],
     [1, 1, 1]]
)

SOUTH = np.array(
    [[0, 0, 0],
     [1, 0, 1],
     [1, 1, 1]]
)

SOUTHWEST = np.array(
    [[1, 0, 0],
     [1, 0, 0],
     [1, 1, 1]]
)

WEST = np.array(
    [[1, 1, 0],
     [1, 0, 0],
     [1, 1, 0]]
)

NORTHWEST = np.array(
    [[1, 1, 1],
     [1, 0, 0],
     [1, 0, 0]]
)

KERNELS = {
    (0, 0): np.ones((3, 3)),
    (0, -1): NORTH,
    (1, -1): NORTHEAST,
    (1, 0): EAST,
    (1, 1): SOUTHEAST,
    (0, 1): SOUTH,
    (-1, 1): SOUTHWEST,
    (-1, 0): WEST,
    (-1, -1): NORTHWEST
}
