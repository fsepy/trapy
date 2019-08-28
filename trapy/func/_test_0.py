def my_func():
    import numpy as np
    from trapy.func.transformations import angle_between

    # PROPERTIE

    xyz = np.asarray([
        [5, 5, 0],
        [0, 0, 10]
    ])

    temp = np.asarray([
        1273.15,
        293.15
    ])

    area = np.asarray([
        100,
        1
    ])

    dir = np.asarray([
        [0, 0, 1],
        [0, 0, -1]
    ])

    # RELATIONSHIP

    # Combinations
    # [[emitter, receiver], [emitter, receiver], ...]
    comb = np.asarray([
        [0, 1],
        [1, 0]
    ])

    # Angle
    phi = list()

    for i in comb:
        i0 = i[0]
        i1 = i[1]

        # coordinates
        p0 = xyz[i0, :]
        p1 = xyz[i1, :]

        # normal
        n0 = dir[i0, :]
        n1 = dir[i1, :]

        # ray between two coordinates, both directions
        v01 = p1 - p0
        v10 = p0 - p1

        # angles between the rays and normals
        a0 = angle_between(v01, n0)
        a1 = angle_between(v10, n1)
        a0_ = a0 / np.pi * 180
        a1_ = a0 / np.pi * 180

        # distance
        d = np.square(v01)
        d = np.sum(d)
        d = np.sqrt(d)

        # area
        a = area[i0]

        # view factor
        aaa = np.cos(a0) * np.cos(a1)
        bbb = np.pi * np.square(d)
        phi.append(aaa / bbb * a)

    print(phi)


if __name__ == '__main__':
    import numpy as np
    from typing import Union


    def unit_vector(vector: np.ndarray):
        a = vector.astype(np.float64)
        b = np.sqrt(np.einsum("ij,ij->i", a, a))
        for j in range(np.shape(vector)[1]):
            a[:, j] = np.divide(a[:, j], b)
        return a


    def angle_between(v1: Union[list, np.ndarray], v2: Union[list, np.ndarray]) -> np.ndarray:
        """ Returns the angle in radians between vectors 'v1' and 'v2' """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


    v1 = [0, 1, 1]
    v2 = [1, 1, 0]

    v1 = np.asarray([v1, v1])
    v2 = np.asarray([v2, v2])

    res = angle_between(v1, v2)

    print(res)
