# coding: utf-8

import numpy


def phi_parallel_corner_br187(W_m, H_m, S_m, multiplier=1):
    """

    :param W_m: width of emitter panel
    :param H_m: height of emitter panel
    :param S_m: separation distance from EMITTER TO RECEIVER
    :return phi: configuration factor
    """

    # Calculate view factor, phi
    X = W_m / S_m
    Y = H_m / S_m
    a = 1 / 2 / numpy.pi
    b = X / numpy.sqrt(1 + X**2)
    c = numpy.arctan(Y / numpy.sqrt(1 + X**2))
    d = Y / numpy.sqrt(1 + Y ** 2)
    e = numpy.arctan(X / numpy.sqrt(1 + Y ** 2))
    phi = a * (b * c + d * e)

    return phi*multiplier


def phi_perpendicular_corner(W_m, H_m, S_m, multiplier=1):
    """

    :param W_m:
    :param H_m:
    :param S_m:
    :param multiplier:
    :return:
    """
    X = W_m / S_m
    Y = H_m / S_m

    a = 1 / 2 / numpy.pi
    b = numpy.arctan(X)
    c = 1 / numpy.sqrt(Y ** 2 + 1)
    d = numpy.arctan(X / numpy.sqrt(Y ** 2 + 1))

    phi = a * (b - c * d)

    return phi*multiplier


def four_planes(W_m: float, H_m: float, w_m: float, h_m: float) -> tuple:
    """

    :param W_m:
    :param H_m:
    :param w_m:
    :param h_m:
    :return:
    """

    # COORDINATES
    o = (0, 0)
    e1 = (0, 0)
    e2 = (W_m, H_m)
    r1 = (w_m, h_m)

    # GLOBAL MIN, MEDIAN AND MAX
    min = (numpy.min([W_m, w_m, 0]), numpy.min([H_m, h_m, 0]))
    mid = (numpy.median([W_m, w_m, 0]), numpy.median([H_m, h_m, 0]))
    max = (numpy.max([W_m, w_m, 0]), numpy.max([H_m, h_m, 0]))

    # FOUR PLANES
    A = 0, 0, 0
    B = 0, 0, 0
    C = 0, 0, 0
    D = 0, 0, 0

    # RECEIVER AT CORNER
    if e1 == e2 or e1 == r1 or e1 == (e2[0], r1[1]) or e1 == (r1[0], e2[1]):
        A = (max[0] - min[0], max[1] - min[1], 1)
        B = (0, 0, 0)
        C = (0, 0, 0)
        D = (0, 0, 0)

        # A = phi_parallel_corner_br187(*A, S_m)
        #
        # phi = A

    # RECEIVER ON EDGE
    elif ((r1[0] == e1[0] or r1[0] == e2[0]) and e1[1] < r1[1] < e2[1]) or ((r1[1] == e1[1] or r1[1] == e2[1]) and e1[0] < r1[0] < e2[0]):
        # vertical edge
        if (r1[0] == e1[0] or r1[0] == e2[0]) and e1[1] < r1[1] < e2[1]:
            A = (max[0] - min[0], max[1] - mid[1], 1)
            B = (max[0] - min[0], mid[1] - min[1], 1)
            C = (0, 0, 0)
            D = (0, 0, 0)

        # horizontal edge
        elif (r1[1] == e1[1] or r1[1] == e2[1]) and e1[0] < r1[0] < e2[0]:
            A = (max[0] - mid[0], max[1] - min[1], 1)
            B = (mid[0] - min[0], max[1] - min[1], 1)
            C = (0, 0, 0)
            D = (0, 0, 0)
        else:
            print('error')

    # RECEIVER WITHIN EMITTER
    elif o[0] < w_m < W_m and o[1] < h_m < H_m:
        A = (mid[0] - min[0], mid[1] - min[1], 1)
        B = (max[0] - mid[0], max[1] - mid[1], 1)
        C = (mid[0] - min[0], max[1] - mid[1], 1)
        D = (max[0] - mid[0], mid[1] - min[1], 1)

    # RECEIVER OUTSIDE EMITTER
    else:
        # within y-axis range max[1] and min[1], far right
        if min[1] < r1[1] < max[1] and r1[0] == max[0]:
            A = max[0] - min[0], max[1] - mid[1], 1
            B = max[0] - min[0], mid[1] - min[1], 1
            C = max[0] - mid[0], max[1] - mid[1], -1  # negative
            D = max[0] - mid[0], mid[1] - min[1], -1  # negative
        # within y-axis range max[1] and min[1], far left
        elif min[1] < r1[1] < max[1] and r1[0] == min[0]:
            A = max[0] - min[0], max[1] - mid[1], 1
            B = max[0] - min[0], mid[1] - min[1], 1
            C = mid[0] - min[0], max[1] - mid[1], -1  # negative
            D = mid[0] - min[0], mid[1] - min[1], -1  # negative
        # within x-axis range max[0] and min[0], far top
        elif min[0] < r1[0] < max[0] and r1[1] == max[1]:
            A = max[0] - mid[0], max[1] - min[1], 1
            B = mid[0] - min[0], max[1] - min[1], 1
            C = max[0] - mid[0], max[1] - mid[1], -1
            D = mid[0] - min[0], max[1] - mid[1], -1
        # within x-axis range max[0] and min[0], far bottom
        elif min[0] < r1[0] < max[0] and r1[1] == min[1]:
            A = max[0] - mid[0], max[1] - min[1], 1
            B = mid[0] - min[0], max[1] - min[1], 1
            C = max[0] - mid[0], mid[1] - min[1], -1
            D = mid[0] - min[0], mid[1] - min[1], -1
        # receiver out, within 1st quadrant
        elif r1[0] == max[0] and r1[1] == max[1]:
            A = max[0] - min[0], max[1] - min[1], 1
            B = max[0] - mid[0], max[1] - mid[1], 1
            C = max[0] - mid[0], max[1] - min[1], -1
            D = max[0] - min[0], max[1] - mid[1], -1
        # receiver out, within 2nd quadrant
        elif r1[0] == max[0] and r1[1] == min[1]:
            A = max[0] - min[0], max[1] - min[1], 1
            B = max[0] - mid[0], mid[1] - min[1], 1
            C = max[0] - min[0], mid[1] - min[1], -1
            D = max[0] - mid[0], max[1] - min[1], -1
        # receiver out, within 3rd quadrant
        elif r1[0] == min[0] and r1[1] == min[1]:
            A = max[0] - min[0], max[1] - min[1], 1
            B = mid[0] - min[0], mid[1] - min[1], 1
            C = mid[0] - min[0], max[1] - min[1], -1
            D = max[0] - min[0], mid[1] - min[1], -1
        # receiver out, within 4th quadrant
        elif r1[0] == min[0] and r1[1] == max[1]:
            A = max[0] - min[0], max[1] - min[1], 1
            B = mid[0] - min[0], max[1] - mid[1], 1
            C = mid[0] - min[0], max[1] - min[1], -1
            D = max[0] - min[0], max[1] - mid[1], -1
        # unkown
        else:
            return numpy.nan, numpy.nan, numpy.nan

    return A, B, C, D


def phi_parallel_any_br187(W_m, H_m, w_m, h_m, S_m):
    phi = [phi_parallel_corner_br187(*P[0:-1], S_m, P[-1]) for P in four_planes(W_m, H_m, w_m, h_m)]
    return numpy.sum(phi)


def phi_perpendicular_any_br187(W_m, H_m, w_m, h_m, S_m):
    four_P = four_planes(W_m, H_m, w_m, h_m)
    phi = [phi_perpendicular_corner(*P[0:-1], S_m, P[-1]) for P in four_P]
    return numpy.sum(phi)


def test_phi_parallel_any_br187():

    # All testing values are taken from independent sources

    # check receiver at emitter corner
    assert abs(phi_parallel_any_br187(*(10, 10, 0, 0, 10)) - 0.1385316060) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 0, 10, 10)) - 0.1385316060) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 10, 10, 10)) - 0.1385316060) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 10, 0, 10)) - 0.1385316060) < 1e-8

    # check receiver on emitter edge
    assert abs(phi_parallel_any_br187(*(10, 10, 2, 0, 10)) - 0.1638694545) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 2, 10, 10)) - 0.1638694545) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 0, 2, 10)) - 0.1638694545) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 10, 2, 10)) - 0.1638694545) < 1e-8

    # check receiver within emitter, center
    assert abs(phi_parallel_any_br187(*(10, 10, 5, 5, 10)) - 0.2394564705) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 2, 2, 10)) - 0.1954523349) < 1e-8

    # check receiver fall outside, side ways
    assert abs(phi_parallel_any_br187(*(10, 10, 5, 15, 10)) - 0.0843536644) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 5, -5, 10)) - 0.0843536644) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 15, 5, 10)) - 0.0843536644) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, -5, 5, 10)) - 0.0843536644) < 1e-8

    # check receiver fall outside, 1st quadrant
    assert abs(phi_parallel_any_br187(*(10, 10, 20, 15, 10)) - 0.0195607021) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, 20, -5, 10)) - 0.0195607021) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, -10, -5, 10)) - 0.0195607021) < 1e-8
    assert abs(phi_parallel_any_br187(*(10, 10, -10, 15, 10)) - 0.0195607021) < 1e-8


def fire_height_drysdale(Q_kW, D_m):

    if not 7 < Q_kW ** (2/5) / D_m < 700:
        # raise Exception('Not in range 7 < Q_kW**(2/5)/D_m < 700: {}'.format(Q_kW ** (2/5) / D_m))
        l = numpy.nan
    else:
        l = 0.23 * Q_kW ** (2/5) - 1.02 * D_m

    return l


def test_phi_perpendicular_br187():

    # All testing values are taken from independent sources

    # check receiver at emitter corner
    assert abs(phi_perpendicular_any_br187(10, 10, 0, 0, 10) - 0.05573419700) < 1e-8

    # check receiver on emitter edge
    assert abs(phi_perpendicular_any_br187(10, 10, 2, 0, 10) - 0.06505816388) < 1e-8
    assert abs(phi_perpendicular_any_br187(10, 10, 2, 10, 10) - 0.06505816388) < 1e-8
    assert abs(phi_perpendicular_any_br187(10, 10, 0, 2, 10) - 0.04656468770) < 1e-8
    assert abs(phi_perpendicular_any_br187(10, 10, 10, 2, 10) - 0.04656468770) < 1e-8

    # check receiver fall outside, side ways
    assert abs(phi_perpendicular_any_br187(10, 10, 5, -10, 10) - 0.04517433814) < 1e-8
    assert abs(phi_perpendicular_any_br187(10, 10, 5, 20, 10) - 0.04517433814) < 1e-8


if __name__ == '__main__':
    test_phi_perpendicular_br187()
    test_phi_parallel_any_br187()

    a = fire_height_drysdale(500, 1)
    print(a)
