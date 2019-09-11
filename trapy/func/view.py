from typing import Union
from matplotlib.path import Path
import numpy as np
from matplotlib.path import Path


def polygon_area_2d(x, y):
    # shoelace method:
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def test_poly_area_2d():
    """To test polygon_area_2d"""
    x = [0, 10, 10, 4]
    y = [0, 0, 5, 5]
    assert polygon_area_2d(x, y) == 40

    x = [0, 10, 10, 6, 10, 4, 0]
    y = [0, 0, 5, 5, 10, 10, 6]
    assert polygon_area_2d(x, y) == 82


def polygon_area_3d(x, y, z):
    """
    TODO: WIP
    :param x:
    :param y:
    :return:
    """

    return polygon_area_2d(x, y)


def points_in_polygon_2d(points, polygon):
    path = Path(polygon)
    return path.contains_points(points)


def scatter_in_polygon_2d(polygon: np.ndarray, n_points):
    x1, x2 = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    y1, y2 = np.min(polygon[:, 1]), np.max(polygon[:, 1])

    domain_area = (x2 - x1) * (y2 - y1)

    poly_area = polygon_area_2d(polygon[:, 0], polygon[:, 1])

    n_points *= (domain_area / poly_area)
    n_points = int(n_points) + 1

    a = domain_area / n_points

    l = a ** 0.5

    # xx = np.arange(x1, x2+l, l)
    # yy = np.arange(y1, y2+l, l)
    xx = np.linspace(x1, x2, int((x2 - x1) / l) + 1, endpoint=True, dtype=np.float64)
    yy = np.linspace(y1, y2, int((y2 - y1) / l) + 1, endpoint=True, dtype=np.float64)

    xx, yy = np.meshgrid(xx, yy)

    xx = xx.flatten().reshape(xx.size, 1)
    # xx = xx.reshape(len(xx), 1)

    yy = yy.flatten().reshape(yy.size, 1)
    # yy = yy.reshape(len(xx), 1)

    xy = np.concatenate([xx, yy], axis=1)

    return xy


def test_scatter_in_polygon_2d():
    # points = np.random.uniform(low=-2, high=12, size=10000).reshape(5000, 2)

    x = [0, 10, 10, 6, 10, 4, 0]
    y = [0, 0, 5, 5, 10, 10, 6]

    xy = list(zip(x, y))

    xy = [list(i) for i in xy]
    polygon = np.array(xy)

    xy = scatter_in_polygon_2d(polygon, 1000)

    points_in_polygon_ = points_in_polygon_2d(xy, polygon)

    print(np.sum(points_in_polygon_))

    import matplotlib.pyplot as plt
    plt.scatter(xy[:, 0][points_in_polygon_], xy[:, 1][points_in_polygon_])
    plt.show()


def scatter_in_polygon_3d(polygon: np.ndarray, n_points: float) -> np.ndarray:
    """
    TODO: WIP
    :param polygon:
    :param n_points:
    :return:
    """

    z = np.mean(polygon[:, 2])

    xy = scatter_in_polygon_2d(polygon[:, 0:2], n_points)

    xyz = np.zeros(np.array(np.shape(xy)) + np.array([0, 1]), dtype=np.float64)
    xyz[:, 0:2] = xy
    xyz[:, 2] = z

    return xyz


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

        # distance
        # area
        a = area[i0]

        # view factor
        aaa = np.cos(a0) * np.cos(a1)
        bbb = np.pi * d_square[i0]
        phi_[i] = (aaa / bbb * a)

    return phi_


def resultant_heat_flux(xyz: np.ndarray, norm: np.ndarray, area: np.ndarray, temperature: np.ndarray, indexes: np.ndarray):
    xyz0 = xyz[indexes.astype(dtype=int)[:, 0]]
    xyz1 = xyz[indexes.astype(dtype=int)[:, 1]]

    n0_ = norm[indexes.astype(dtype=int)[:, 0]]
    n1_ = norm[indexes.astype(dtype=int)[:, 1]]

    a_min_b = xyz0 - xyz1
    d_square = np.einsum("ij,ij->i", a_min_b, a_min_b)

    v01_ = xyz1 - xyz0
    v10_ = xyz0 - xyz1

    heat_flux_dosage = np.zeros((len(indexes),), dtype=np.float64)

    for i, v in enumerate(indexes):
        i0, i1 = int(v[0]), int(v[1])

        # angles between the rays and normals
        a0 = angle_between(v01_[i0], n0_[i0])
        a1 = angle_between(v10_[i0], n1_[i0])

        # area
        a = area[i0]

        # view factor
        aaa = np.cos(a0) * np.cos(a1)
        bbb = np.pi * d_square[i0]
        phi = (aaa / bbb * a)

        # temperature difference
        t0 = temperature[i0]
        t1 = temperature[i1]
        dt4 = t0**4 - t1**4

        heat_flux_dosage[i] = 5.67e-8 * 1.0 * dt4 * phi

    return heat_flux_dosage


def thermal_radiation_dose(xyz: np.ndarray, heat_flux: np.ndarray, ):
    pass


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


def single_receiver(ep: list = None, receiver: list = None):
    """

    :param ep: list of emitter polygon/panel
    :param receiver: coordinate of the receiver point shape (3,)
    :return:
    """

    # required parameters
    ep = list()
    norm = list()
    temperature = list()
    ep.append(
        [
            [-10, -10, 10],
            [10, -10, 10],
            [10, 10, 10],
            [-10, 10, 10]
        ]
    )
    ep = np.asarray(ep)
    norm.append([0, 0, -1])
    temperature.append([1000.])

    ep_xyz = None  # coordinates of sample points on emitter polygons
    ep_xyz_norm = None
    ep_xyz_temperature = None
    ep_xyz_area = None
    for i, p in enumerate(ep):
        xyz_ = scatter_in_polygon_3d(polygon=p, n_points=100)
        norm_ = np.zeros_like(xyz_)
        temperature_ = np.zeros(shape=(len(xyz_),))

        if ep_xyz:
            pass
            # ep_xyz = np.concatenate([ep_xyz, xyz_])
        else:
            ep_xyz = xyz_
            ep_xyz_norm = np.zeros_like(xyz_, dtype=np.float64)
            ep_xyz_norm[:, :] = norm[i]
            ep_xyz_temperature = np.zeros(shape=(len(xyz_),), dtype=np.float64)
            ep_xyz_temperature[:] = temperature[i]
            ep_xyz_area = np.zeros((np.shape(xyz_)[0],), dtype=np.float64)
            ep_xyz_area[:] = polygon_area_3d(p[:, 0], p[:, 1], p[:, 2]) / ep_xyz_area.size

    # add the single hot spot (i.e. receiver)
    ep_xyz = np.concatenate([np.array([[0, 0, 0]]), ep_xyz])
    ep_xyz_norm = np.concatenate([np.array([[0, 0, 1]]), ep_xyz_norm])
    ep_xyz_temperature = np.concatenate([[293.15], ep_xyz_temperature])
    ep_xyz_area = np.concatenate([[1], ep_xyz_area])

    indexes = np.zeros((len(ep_xyz)-1, 2), dtype=int)
    indexes[:, 0] = np.arange(1, len(ep_xyz), 1)
    indexes[:, 1] = 0

    resultant_heat_flux(
        xyz=ep_xyz,
        norm=ep_xyz_norm,
        temperature=ep_xyz_temperature,
        area=ep_xyz_area,
        indexes=indexes
    )
    # xyz = np.concatenate([xyz_0, xyz_1], axis=0)
    # from trapy.func.vis import plot_3d_plotly
    # fig = plot_3d_plotly(xyz[:, 0], xyz[:, 1], xyz[:, 2], v=xyz[:, 2])


if __name__ == '__main__':
    # test_angle_between()
    # test_phi_parallel()
    # test_phi_perpendicular()
    # test_poly_area_2d()
    #
    # test_scatter_in_polygon_2d()

    single_receiver()
