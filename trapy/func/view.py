from typing import Union

import numpy as np


def unit_vector(vector: np.ndarray):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def angle_between(v1: Union[list, np.ndarray], v2: Union[list, np.ndarray]) -> np.ndarray:
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def test_angle_between():
    v1 = [0, 0, 1]
    v2 = [0, 0, 1]
    assert angle_between(v1, v2) == 0

    v1 = [0, 0, 1]
    v2 = [0, 0, -1]
    assert angle_between(v1, v2) == np.pi

    v1 = [0, 0, 1]
    v2 = [0, 1, 0]
    assert angle_between(v1, v2) == np.pi / 2

    v1 = [0, 0, 1]
    v2 = [1, 0, 0]
    assert angle_between(v1, v2) == np.pi / 2


def phi(xyz: np.ndarray, norm: np.ndarray, area: np.ndarray, indexes: np.ndarray):
    xyz0 = xyz[indexes.astype(dtype=int)[:, 0]]
    xyz1 = xyz[indexes.astype(dtype=int)[:, 1]]

    n0_ = norm[indexes.astype(dtype=int)[:, 0]]
    n1_ = norm[indexes.astype(dtype=int)[:, 1]]

    a_min_b = xyz0 - xyz1
    d_square = np.einsum("ij,ij->i", a_min_b, a_min_b)

    v01_ = xyz1 - xyz0
    v10_ = xyz0 - xyz1

    phi_ = np.zeros((len(indexes),), dtype=np.float64)

    for i, v in enumerate(indexes):
        i0, i1 = int(v[0]), int(v[1])

        # angles between the rays and normals
        a0 = angle_between(v01_[i0], n0_[i0])
        a1 = angle_between(v10_[i0], n1_[i0])
        # a0_ = a0 / np.pi * 180
        # a1_ = a0 / np.pi * 180

        # distance
        # area
        a = area[i0]

        # view factor
        aaa = np.cos(a0) * np.cos(a1)
        bbb = np.pi * d_square[i0]
        phi_[i] = (aaa / bbb * a)

    return phi_


def test_phi_perpendicular():
    import itertools
    import numpy as np

    # PROPERTIE

    arr = np.linspace(0, 10, 201)
    arr = (arr[1:] + arr[:-1]) / 2
    arr = np.asarray(list(itertools.product(arr, arr)))
    xyz = np.zeros((np.shape(arr)[0] + 1, 3))
    xyz[0:np.shape(arr)[0], 0] = arr[:, 0]
    xyz[0:np.shape(arr)[0], 1] = arr[:, 1]
    xyz[-1, 2] = 10

    temp = np.full((len(arr) + 1,), 1000)
    temp[-1] = 300

    area = np.full_like(temp, 100 / len(arr), dtype=np.float64)
    area[-1] = 0

    norm = [[0, 0, 1] for i in arr] + [[0, 1, 0]]
    norm = np.asarray(norm)

    # Ray destination indexes
    # [[emitter, receiver], [emitter, receiver], ...]
    indexes = np.zeros((len(arr), 2))
    indexes[:, 0] = np.arange(0, len(arr), 1)
    indexes[:, 1] = len(arr)

    p = phi(
        xyz=xyz,
        norm=norm,
        area=area,
        indexes=indexes
    )

    assert np.allclose(0.0557341, np.sum(p))


def test_phi_parallel():
    import itertools
    import numpy as np

    # PROPERTIE

    arr = np.linspace(0, 10, 101)
    arr = (arr[1:] + arr[:-1]) / 2
    arr = np.asarray(list(itertools.product(arr, arr)))
    xyz = np.zeros((np.shape(arr)[0] + 1, 3))
    xyz[0:np.shape(arr)[0], 0] = arr[:, 0]
    xyz[0:np.shape(arr)[0], 1] = arr[:, 1]
    xyz[-1, 2] = 10

    temp = np.full((len(arr) + 1,), 1000)
    temp[-1] = 300

    area = np.full_like(temp, 100 / len(arr), dtype=np.float64)
    area[-1] = 0

    norm = [[0, 0, 1] for i in arr] + [[0, 0, -1]]
    norm = np.asarray(norm)

    # Ray destination indexes
    # [[emitter, receiver], [emitter, receiver], ...]
    indexes = np.zeros((len(arr), 2))
    indexes[:, 0] = np.arange(0, len(arr), 1)
    indexes[:, 1] = len(arr)

    p = phi(
        xyz=xyz,
        norm=norm,
        area=area,
        indexes=indexes
    )

    assert np.allclose(0.1385316, np.sum(p))


if __name__ == '__main__':
    test_angle_between()
    test_phi_parallel()
    test_phi_perpendicular()
