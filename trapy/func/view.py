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


def polygon_area_3d(polygon: np.ndarray) -> np.ndarray:
    """

    :param polygon:
    :param n_points:
    :return:
    """

    area = polygon_area_2d(polygon[:, 0], polygon[:, 1])

    return area


def scatter_in_polygon_2d(polygon: np.ndarray, n_points: int) -> np.ndarray:
    """Cast n_points number of points on a surface, within a defined polygon, in 2 dimensional space.

    :param polygon:     [[x0, y0], [x1, y1], [x2, y2], ... [xn, yn]], each item polygon[i, :] is a point in a 2d space
                        and all points forms a polygon.
    :param n_points:    number of dots to be casted onto the surface.
    :return xy:         a list of [x, y] coordinates represents points casted on the surface.
    """

    assert isinstance(polygon, np.ndarray) and np.shape(polygon)[1] == 2
    assert isinstance(n_points, int) and n_points > 0

    # work out x and y boundary
    x1, x2 = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    y1, y2 = np.min(polygon[:, 1]), np.max(polygon[:, 1])

    # the total area within the polygon
    poly_area = polygon_area_2d(polygon[:, 0], polygon[:, 1])

    # the area within the square that contains the polygon
    domain_area = (x2 - x1) * (y2 - y1)

    # reworked total points within the square
    n_points *= (domain_area / poly_area)
    n_points = int(n_points) + 1

    # area of each points on the square surface
    a = domain_area / n_points

    l = a ** 0.5

    xx = np.linspace(x1, x2, int((x2 - x1) / l) + 1, endpoint=True, dtype=np.float64)
    yy = np.linspace(y1, y2, int((y2 - y1) / l) + 1, endpoint=True, dtype=np.float64)

    xx -= xx[1] - xx[0]
    yy -= yy[1] - yy[0]

    xx, yy = np.meshgrid(xx, yy)
    xx = xx.flatten()
    xx = xx.reshape((xx.size, 1))
    yy = yy.flatten()
    yy = yy.reshape((yy.size, 1))
    # xx, yy = xx.flatten().reshape(xx.size, 1), yy.flatten().reshape(yy.size, 1)

    # get the points co-ordinates within the polygon
    path = Path(polygon)
    xy = np.concatenate([xx, yy], axis=1)
    xy_filter_inside_the_polygon = path.contains_points(xy)
    xy = xy[xy_filter_inside_the_polygon]

    return xy


def test_scatter_in_polygon_2d():
    # points = np.random.uniform(low=-2, high=12, size=10000).reshape(5000, 2)

    x = [0, 10, 10, 6, 10, 4, 0]
    y = [0, 0, 5, 5, 10, 10, 6]

    xy = list(zip(x, y))

    xy = [list(i) for i in xy]
    polygon = np.array(xy)

    xy = scatter_in_polygon_2d(polygon, 1000)

    print(xy)


def scatter_in_polygon_3d(polygon: np.ndarray, n_points: int) -> np.ndarray:
    """WIP. Cast n_points number of points on a surface, within a defined polygon, in 3 dimensional space.
    As WIP, the third dimension will be discarded but returned as orginal inputs.

    :param polygon:
    :param n_points:
    :return:
    """

    z_averaged = np.average(polygon[:, 2])

    xy = scatter_in_polygon_2d(polygon[:, 0:2], n_points)

    xyz = np.c_[xy, np.full(np.shape(xy)[0], z_averaged)]

    return xyz


def test_scatter_in_polygon_3d():
    # points = np.random.uniform(low=-2, high=12, size=10000).reshape(5000, 2)

    x = [0, 10, 10, 6, 10, 4, 0]
    y = [0, 0, 5, 5, 10, 10, 6]
    z = [2, 2, 2, 2, 2, 2, 2]

    xyz = list(zip(x, y, z))

    xyz = [list(i) for i in xyz]
    polygon = np.array(xyz)

    xy = scatter_in_polygon_3d(polygon, 1000)

    print(xy)


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
        a0 = angle_between(v01_[i], n0_[i])
        a1 = angle_between(v10_[i], n1_[i])

        # distance
        # area
        a = area[i0]

        # view factor
        aaa = np.cos(a0) * np.cos(a1)
        bbb = np.pi * d_square[i]
        phi_[i] = (aaa / bbb * a)

    return phi_


def resultant_heat_flux(xyz: np.ndarray, norm: np.ndarray, area: np.ndarray, temperature: np.ndarray, indexes: np.ndarray):

    xyz0 = xyz[indexes.astype(dtype=int)[:, 0]]
    xyz1 = xyz[indexes.astype(dtype=int)[:, 1]]

    n0_ = norm[indexes.astype(dtype=int)[:, 0]]
    n1_ = norm[indexes.astype(dtype=int)[:, 1]]

    a_min_b = xyz0 - xyz1
    d_square = np.einsum("ij,ij->i", a_min_b, a_min_b)

    v01_ = xyz1 - xyz0  # vector array from vertex 0 to 1
    v10_ = xyz0 - xyz1  # vector array from vertex 1 to 0

    heat_flux_dosage = 0.
    phi_ = 0.

    for i, v in enumerate(indexes):
        i0, i1 = int(v[0]), int(v[1])

        # angles between the rays and normals
        a0 = angle_between(v01_[i], n0_[i])
        a1 = angle_between(v10_[i], n1_[i])

        # area
        a = area[i0]

        # view factor
        aaa = np.cos(a0) * np.cos(a1)
        bbb = np.pi * d_square[i]
        phi = (aaa / bbb) * a

        # temperature difference
        t0 = temperature[i0]
        t1 = temperature[i1]
        dt4 = t0**4 - t1**4

        heat_flux_dosage += 5.67e-8 * 1.0 * dt4 * phi
        phi_ += phi

    return heat_flux_dosage, phi_


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

    p_sum = np.sum(p)

    from trapy.func.vis import plot_3d_plotly
    my_fig = plot_3d_plotly(xyz[:, 0], xyz[:, 1], xyz[:, 2], p)
    my_fig.show()

    assert np.allclose(0.1385316, p_sum)


def test_single_receiver():
    """
    Emitter panel:
        - dimension         5 x 5 m
        - location          5 m above z=0
        - orientation       (0, 0, -1)
        - temperature       1153 K (equivalent to 100 kW/m2)
    Receiver spot:
        - location          5 meters away from the emitter and center to the emitter
        - orientation       (0, 0, 1)
        - temperature       293.15 K (20 deg.C)
    """

    # Define Emitter Panel
    # ====================

    # dimension
    ep = np.asarray([
        [0, 0, 5],
        [0, 5, 5],
        [5, 5, 5],
        [5, 0, 5],
    ])

    # orientation
    ep_norm = np.asarray([0, 0, -1])

    # surface temperature
    ep_temperature = [1153.]  # equivalent to 100 kW/m2

    # Define Receiver
    # ===============

    # location
    rp = np.asarray([2.5, 2.5, 0])

    # orientation
    rp_norm = np.asarray([0, 0, 1])

    # temperature
    rp_temperature = 293.15

    single_receiver(
        ep_vertices=ep,
        ep_norm=ep_norm,
        ep_temperature=ep_temperature,
        n_points=1000,  # number of hot spots
        rp_vertices=rp,
        rp_norm=rp_norm,
        rp_temperature=rp_temperature
    )


def single_receiver(
        ep_vertices: np.ndarray,
        ep_norm: np.ndarray,
        ep_temperature: float,
        n_points: int,
        rp_vertices: np.ndarray,
        rp_norm: np.ndarray,
        rp_temperature: float
):
    """Calculates resultant heat flux at a receiver from an emitter.

    :param ep_vertices: vertices defining an emitter polygon, in [[x1, y1, z1], [x2, y2, z2], ...]
    :param ep_norm: a vector definiting the facing direction of the emitter polygon, in [x, y, z]
    :param ep_temperature: [K] emitter surface temperature.
    :param n_points:
    :param rp_vertices:
    :param rp_norm:
    :param rp_temperature:
    :return:
    """

    global_vertices = np.asarray(ep_vertices)

    ep_xyz = None  # coordinates of sampled points on emitter polygons
    ep_xyz_norm = None
    ep_xyz_temperature = None
    ep_xyz_area = None

    # Process Data for Emitter Panel
    # ==============================

    # Get coordinates of individual hot spots
    ep_xyz_spots = scatter_in_polygon_3d(polygon=ep_vertices, n_points=n_points)

    # Get norm of individual hot spots
    norm_ = np.zeros_like(ep_xyz_spots)
    ep_xyz_norm = np.zeros_like(ep_xyz_spots, dtype=np.float64)
    ep_xyz_norm[:, :] = ep_norm

    # Get temperature of individual hot spots
    ep_xyz_temperature = np.full(shape=(len(ep_xyz_spots),), fill_value=ep_temperature, dtype=np.float64)

    ep_xyz_area = np.full(shape=np.shape(ep_xyz_spots)[0], fill_value=polygon_area_3d(ep_vertices) / np.shape(ep_xyz_spots)[0], dtype=np.float64)

    n_points = ep_xyz_area.size

    # add the single cold spot (i.e. receiver)
    ep_xyz = np.concatenate([np.reshape(rp_vertices, (1, rp_vertices.size)), ep_xyz_spots])
    ep_xyz_norm = np.concatenate([np.reshape(rp_norm, (1, rp_norm.size)), ep_xyz_norm])
    ep_xyz_temperature = np.concatenate([[rp_temperature], ep_xyz_temperature])
    ep_xyz_area = np.concatenate([[1], ep_xyz_area])

    indexes = np.zeros((n_points, 2), dtype=int)
    indexes[:, 0] = np.arange(1, n_points+1, 1)
    indexes[:, 1] = 0

    res, phi_ = resultant_heat_flux(
        xyz=ep_xyz,
        norm=ep_xyz_norm,
        temperature=ep_xyz_temperature,
        area=ep_xyz_area,
        indexes=indexes
    )

    print('heat flux', np.sum(res)/1000)
    print('phi', np.sum(phi_))

    # from trapy.func.vis import plot_3d_plotly
    # my_fig = plot_3d_plotly(ep_xyz[:, 0], ep_xyz[:, 1], ep_xyz[:, 2], phi_)
    # my_fig.show()
    return res, phi_


def heat_flux_to_temperature(heat_flux: float, exposed_temperature:float = 293.15):
    """Function returns surface temperature of an emitter for a given heat flux.

    :param heat_flux: [W/m2] heat flux of emitter.
    :param exposed_temperature: [K] ambient/receiver temperature, 20 deg.C by default.
    :return temperature: [K] calculated emitter temperature based on black body radiation model.
    """
    epsilon = 1.0  # radiation view factor
    sigma = 5.67e-8  # [W/m2/K4] stefan-boltzmann constant
    # E_dash_dash_dot = epsilon * sigma * (T_1 ** 4 - T_0 ** 4)  # [W/m2]
    return ((heat_flux / sigma / epsilon) + exposed_temperature ** 4) ** 0.25


def temperature_to_heat_flux(temperature: float, ambient_temperature: float = 293.15):
    """Function returns hot surface heat flux for a given temperature.

    :param temperature: [K] emitter temperature.
    :param ambient_temperature: [K] ambient/receiver temperature, 20 deg.C by default.
    :return heat_flux: [K] calculated emitter temperature based on black body radiation model.
    """

    epsilon = 1.0  # radiation view factor
    sigma = 5.67e-8  # [W/m2/K4] stefan-boltzmann constant

    heat_flux = epsilon * sigma * (temperature ** 4 - ambient_temperature ** 4)

    return heat_flux


if __name__ == '__main__':

    # test_angle_between()
    # test_phi_parallel()
    # test_phi_perpendicular()
    # test_poly_area_2d()

    # test_scatter_in_polygon_2d()

    # print(heat_flux_to_temperature(84000, 293.15))
    # print(heat_flux_to_temperature(100000))

    test_single_receiver()

    pass
