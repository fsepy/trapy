import math
from typing import Union

import numpy as np


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class Rectangle:
    def __init__(self):
        self._name = None
        self._local_vertex_1 = None
        self._local_vertex_2 = None
        self._temperature = None
        self._emissivity = 1.0
        self._type = None
        self._is_reverse = False

        self._local2global_xyz = None
        self._local2global_rotation_angle = None
        self._local2global_rotation_axis = None
        self._global_vertex_1 = None
        self._global_vertex_2 = None
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name
    
    @property
    def local_vertex_1(self) -> np.ndarray:
        return self._local_vertex_1

    @local_vertex_1.setter
    def local_vertex_1(self, vertex: Union[tuple, list, np.ndarray]):
        self._local_vertex_1 = vertex

    @property
    def local_vertex_2(self) -> np.ndarray:
        return self._local_vertex_2

    @local_vertex_2.setter
    def local_vertex_2(self, vertex: Union[tuple, list, np.ndarray]):
        self._local_vertex_2 = vertex

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float):
        self._temperature = temperature

    @property
    def local2global_rotation_angle(self):
        return self._local2global_rotation_angle

    @local2global_rotation_angle.setter
    def local2global_rotation_angle(self, angle):
        self._local2global_rotation_angle = angle

    @property
    def local2global_rotation_axis(self):
        return self._local2global_rotation_axis

    @local2global_rotation_axis.setter
    def local2global_rotation_axis(self, axis):
        self._local2global_rotation_axis = axis

    @property
    def local2global_xyz(self):
        return self._local2global_xyz

    @local2global_xyz.setter
    def local2global_xyz(self, xyz: Union[list, tuple, np.ndarray]):
        self._local2global_xyz = xyz

    @property
    def global_vertex_1(self):
        return self._global_vertex_1
    @global_vertex_1.setter
    def global_vertex_1(self, vertex):
        self._global_vertex_1 = vertex

    @property
    def global_vertex_2(self):
        return self._global_vertex_2
    @global_vertex_2.setter
    def global_vertex_2(self, vertex):
        self._global_vertex_2 = vertex

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, x: str):
        self._type = x

    @property
    def is_reverse(self):
        return self._is_reverse

    @is_reverse.setter
    def is_reverse(self, x: bool):
        self._is_reverse = x

    def local2global_vertices(
            self,
            v1=None,
            v2=None,
            xyz: Union[list, tuple, np.ndarray] = None,
            axis: Union[list, tuple, np.ndarray] = None,
            angle: float = None
    ):
        # assign parameters
        if v1:
            self.local_vertex_1 = v1
        if v2:
            self.local_vertex_2 = v2
        if xyz:
            self.local2global_xyz = xyz
        if axis:
            self.local2global_rotation_axis = axis
        if angle:
            self.local2global_rotation_angle = angle

        if not (self.local_vertex_1 or self.local_vertex_2):
            raise ValueError('Missing local vertex information.')

        # local2global rotation
        rot_mat = rotation_matrix(self.local2global_rotation_axis, self.local2global_rotation_angle)
        vertex_1 = np.dot(rot_mat, self.local_vertex_1)
        vertex_2 = np.dot(rot_mat, self.local_vertex_2)

        # local2global shift
        vertex_1 += self.local2global_xyz
        vertex_2 += self.local2global_xyz

        # assign global vertices to object
        self.global_vertex_1 = vertex_1
        self.global_vertex_2 = vertex_2
        return vertex_1, vertex_2

    @staticmethod
    def get_global_vertices(v1, v2):
        def max_a_min_b(a, b):
            # if a < b:
            #     a += b
            #     b = a - b
            #     a -= b
            return a, b

        xmax, xmin = max_a_min_b(v1[0], v2[0])
        ymax, ymin = max_a_min_b(v1[1], v2[1])
        zmax, zmin = max_a_min_b(v1[2], v2[2])

        vv1 = [xmin, ymin, zmin]
        vv2 = [xmax, ymax, zmin]
        vv3 = [xmax, ymax, zmax]
        vv4 = [xmin, ymin, zmax]

        return vv1, vv2, vv3, vv4

    def get_tra_command(self):

        type = self.type

        geometry = self.get_global_vertices(self.global_vertex_1, self.global_vertex_2)
        geometry = [':'.join(['{:.3f}'.format(c) for c in v]) for v in geometry]
        geometry = '*'.join(geometry)

        name = self.name

        temperature = self.temperature

        reverse = self.is_reverse

        emissivity = self._emissivity

        return f'<Type={type}, Geometry={geometry}, Name={name}, Temperature={temperature}, Reverse={reverse}, Emissivity={emissivity}>'


def w3_emitter():

    """
    Type=Emitter
    Geometry=5:0:0 * 10.2:0:0 * 10.2:-1.47:5.39 * 5:-1.47:5.39
    Name=Glazing
    Temperature=1105
    Reverse=FALSE
    Emissivity=1
    <Type=Emitter,Geometry=5:0:0 * 10.2:0:0 * 10.2:-1.47:5.39 * 5:-1.47:5.39,Name=Glazing,Temperature=1105,Reverse=FALSE,Emissivity=1>
    """

    angle = (180 + 90 + 9.5) / 180 * np.pi

    # w3 - level 0
    edges__ = [-5.4] + list(np.arange(0, 45 + 0.1, 3))
    height__ = np.full((len(edges__)-1,), 3.5)
    height__[0] = 3.7
    temperature__ = np.full_like(height__, fill_value=1105, dtype=float)
    temperature__[[1, 3, 5, 8, 9, 12, 14]] = 1105.
    p_w3_l0 = list()
    for i, height in enumerate(height__):
        p = Rectangle()
        p.name = f'w3_0_{i}'
        p.local_vertex_1 = [edges__[i], 0, 0]
        p.local_vertex_2 = [edges__[i+1], 0, height]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = [0, 0, 0]
        p.temperature = temperature__[i]
        p.local2global_vertices()
        p.type = 'Emitter'
        p.is_reverse = True
        p_w3_l0.append(p)

    for p in p_w3_l0:
        # print(p.name , p.global_vertex_1, p.global_vertex_2)
        print(p.get_tra_command())

    # w3 - level 1
    cx__ = list(np.arange(2.25, 3 * 6, 3)) + list(np.arange(30.75, 16 * 3, 3))
    cx__[-1] = 15 * 3 + 1.5
    cz__ = list(np.full((6,), 4.25 + 2.25)) + list(np.full((6,), 4.25 + 1.75))
    cz__[-1] = 4.25 + 4.2 / 2
    width__ = np.full_like(cx__, 1.5)
    width__[-1] = 3
    height__ = np.full_like(cx__, 2.5)
    height__[6:] = 3.5
    height__[-1] = 4.2
    temperature__ = np.full_like(cx__, 1105)
    p_w3_l1 = list()
    for i, cx in enumerate(cx__):
        cz = cz__[i]
        h = height__[i]
        w = width__[i]
        p = Rectangle()
        p.name = f'w3_1_{i}'
        p.local_vertex_1 = [cx - w / 2, 0, cz - h / 2]
        p.local_vertex_2 = [cx + w / 2, 0, cz + h / 2]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = [0, 0, 0]
        p.local2global_vertices()
        p.type = 'Emitter'
        p.is_reverse = True
        p.temperature = temperature__[i]
        p_w3_l1.append(p)

    for p in p_w3_l1:
        # print(p.name , p.global_vertex_1, p.global_vertex_2)
        print(p.get_tra_command())

    # w3 - level 2
    cx__ = list(np.arange(2.25, 3 * 6, 3)) + list(np.arange(30.75, 16 * 3, 3))
    cz__ = list(np.full((12,), 8.5 + 1.75))
    width__ = np.full_like(cx__, 1.5)
    height__ = np.full_like(cx__, 3.5)
    temperature__ = np.full_like(cx__, 1105)
    p_w3_l2 = list()
    for i, cx in enumerate(cx__):
        cz = cz__[i]
        h = height__[i]
        w = width__[i]
        p = Rectangle()
        p.name = f'w3_2_{i}'
        p.local_vertex_1 = [cx - w / 2, 0, cz - h / 2]
        p.local_vertex_2 = [cx + w / 2, 0, cz + h / 2]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = [0, 0, 0]
        p.local2global_vertices()
        p.type = 'Emitter'
        p.is_reverse = True
        p.temperature = temperature__[i]
        p_w3_l2.append(p)

    for p in p_w3_l2:
        # print(p.name, p.global_vertex_1, p.global_vertex_2)
        print(p.get_tra_command())


def w3_receiver():
    angle = (180 + 90 + 9.5) / 180 * np.pi

    # w3 - receiver
    cx__ = [55 / 2]
    cz__ = [15 / 2]
    width__ = np.full_like(cx__, 55)
    height__ = np.full_like(cx__, 15)
    temperature__ = np.full_like(cx__, 293.15)
    p_w3_lm = list()
    for i, cx in enumerate(cx__):
        cz = cz__[i]
        h = height__[i]
        w = width__[i]
        p = Rectangle()
        p.name = f'w3_m_{i}'
        p.local_vertex_1 = [cx - w / 2, 0, cz - h / 2]
        p.local_vertex_2 = [cx + w / 2, 0, cz + h / 2]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = [0, 0, 0]
        p.local2global_vertices()
        p.type = 'Receiver'
        p.is_reverse = True
        p.temperature = temperature__[i]
        p_w3_lm.append(p)

    for p in p_w3_lm:
        # print(p.name , p.global_vertex_1, p.global_vertex_2)
        print(p.get_tra_command())


def w2_receiver():
    angle = (90 + 36.5) / 180 * np.pi
    # angle = 0. * np.pi

    # w2 - receiver
    cx__ = [54 / 2]
    cz__ = [30 / 2]
    width__ = np.full_like(cx__, 54)
    height__ = np.full_like(cx__, 30)
    temperature__ = np.full_like(cx__, 293.15)
    p_w2_all = list()
    for i, cx in enumerate(cx__):
        cz = cz__[i]
        h = height__[i]
        w = width__[i]
        p = Rectangle()
        p.name = f'w2_2_{i}'
        p.local_vertex_1 = [cx - w / 2, 0, cz - h / 2]
        p.local_vertex_2 = [cx + w / 2, 0, cz + h / 2]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = np.asarray([-0.6, -50.8, 0])
        # p.local2global_xyz = [0, 0, 0]
        p.local2global_vertices()
        p.type = 'Receiver'
        p.is_reverse = True
        p.temperature = temperature__[i]
        p_w2_all.append(p)

    for p in p_w2_all:
        # print(p.name, p.global_vertex_1, p.global_vertex_2)
        print(p.get_tra_command())


def w2_emitter():
    angle = (90 + 36.5) / 180 * np.pi
    # angle = 0. * np.pi

    # w2 - receiver
    cx__ = [13 / 2]
    cz__ = [5 / 2]
    width__ = np.full_like(cx__, 13)
    height__ = np.full_like(cx__, 5)
    temperature__ = np.full_like(cx__, 1313)
    p_w2_all = list()
    for i, cx in enumerate(cx__):
        cz = cz__[i]
        h = height__[i]
        w = width__[i]
        p = Rectangle()
        p.name = f'w2_2_{i}'
        p.local_vertex_1 = [cx - w / 2, 0, cz - h / 2]
        p.local_vertex_2 = [cx + w / 2, 0, cz + h / 2]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = np.asarray([-0.6, -50.8, 0])
        # p.local2global_xyz = [0, 0, 0]
        p.local2global_vertices()
        p.type = 'Emitter'
        p.is_reverse = True
        p.temperature = temperature__[i]
        p_w2_all.append(p)

    for p in p_w2_all:
        # print(p.name, p.global_vertex_1, p.global_vertex_2)
        print(p.get_tra_command())


if __name__ == '__main__':
    # w3_emitter()
    # w2_receiver()
    #
    w3_receiver()
    w2_emitter()
