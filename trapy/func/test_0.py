import numpy as np

xyz = np.asarray([
    [5, 5, 0],
    [0, 0, 10]
])

temp = np.asarray([
    1273.15,
    293.15
])

area = np.asarray([
    1,
    1
])

dir = np.asarray([
    [0, 0, 1],
    [0, 0, -1]
])

comb = np.asarray([
    [0, 1],
    [1, 0]
])


def my_func():
    from trapy.func.transformations import angle_between_vectors

    # CONSTRUCT CONTAINERS

    theta = list()
    theta_cos = list()

    for i in comb:
        i0 = i[0]
        i1 = i[1]

        p0 = xyz[i0, :]
        p1 = xyz[i1, :]

        dir0 = dir[i0, :]

        v01 = p1 - p0

        # CALCULATE ANGLE
        a = angle_between_vectors(v01, dir0)
        a_ = a / np.pi * 180

        a_cos = np.cos(a)
        theta_cos.append(a_cos)

        # DISTANCE
        d = np.square(v01)
        d = np.sum(d)
        d = np.sqrt(d)

        theta.append(a)

    aaa = theta_cos[0] * theta_cos[1]
    bbb = np.pi * np.square(d)
    ccc = area[0]

    phi = aaa / bbb * ccc

    print(phi)


if __name__ == '__main__':
    my_func()
