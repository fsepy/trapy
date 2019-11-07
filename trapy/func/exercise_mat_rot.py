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


def array_windows(
        x: list,
        z: list,
        h: list,
        w: list,
        temperature: list,
        angle: float,
        local2global_xyz: np.ndarray = [0, 0, 0]
) -> list:

    p_ = list()

    for i, cx in enumerate(x):
        z_ = z[i]
        h_ = h[i]
        w_ = w[i]
        p = Rectangle()
        p.name = f'w3_1_{i}'
        p.local_vertex_1 = [cx - w_ / 2, 0, z_ - h_ / 2]
        p.local_vertex_2 = [cx + w_ / 2, 0, z_ + h_ / 2]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = angle
        p.local2global_xyz = local2global_xyz
        p.local2global_vertices()
        p.type = 'Emitter'
        p.is_reverse = True
        p.temperature = temperature[i]
        p_.append(p)

    return p_

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
    # x = [1.5, 1 * 6 + 1.5, 2 * 6 + 1.5, 4 * 6, 6 * 6 - 1.5, 7 * 6 - 1.5]
    # x = [1.5, 1 * 6 + 1.5, 2 * 6 + 1.5, 4 * 6, 6 * 6 - 1.5]
    # z = [1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
    # w = [3, 3, 3, 6, 3, 3]
    # h = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
    # t = np.full_like(x, 1105)
    # p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    # [print(p.get_tra_command()) for p in p_]

    # w3 - level 1 timber facade
    x = [0.75, 3.75, 6.75, 9.75, 12.75, 15.75, 32.25, 35.25, 38.25, 41.25, 44.25]
    z = np.full_like(x, 4.25+3.55/2)
    w = np.full_like(x, 1.5)
    h = np.full_like(x, 3.55)
    t = np.full_like(x, 931)
    p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    [print(p.get_tra_command()) for p in p_]

    # w3 - level 1 window
    x = [2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 30.75, 33.75, 36.75, 39.75, 42.75, 46.5]
    x = [2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 30.75, 33.75, 36.75, 39.75, 42.75,]  # fire rate 1 windows
    z = np.full_like(x, 4.25+3.55/2)
    w = np.full_like(x, 1.5)
    w[-1] = 3
    h = np.full_like(x, 3.55)
    t = np.full_like(x, 1105)
    p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    [print(p.get_tra_command()) for p in p_]

    # w3 - level 1 soffit timber
    x = [9, 5*6+5*3/2]
    z = np.full_like(x, 4.25+3.55+1.45/2)
    w = [3*6, 5*3]
    h = np.full_like(x, 1.45)
    t = np.full_like(x, 931)
    p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    [print(p.get_tra_command()) for p in p_]

    # w3 - level 2 timber facade
    x = [0.75, 3.75, 6.75, 9.75, 12.75, 15.75, 32.25, 35.25, 38.25, 41.25, 44.25, 47.25]
    z = np.full_like(x, 8.5+3.55/2)
    w = np.full_like(x, 1.5)
    h = np.full_like(x, 3.55)
    t = np.full_like(x, 931)
    p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    [print(p.get_tra_command()) for p in p_]

    # w3 - level 2 window
    x = [2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 24, 30.75, 33.75, 36.75, 39.75, 42.75, 45.75]
    x = [2.25, 5.25, 8.25, 11.25, 14.25, 17.25, 24, 30.75, 33.75, 36.75, 39.75, 42.75,]  # fire rate the end three
    z = np.full_like(x, 8.5+3.55/2)
    w = np.full_like(x, 1.5)
    w[6] = 12  # to add the central windows
    h = np.full_like(x, 3.55)
    t = np.full_like(x, 1105)
    p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    [print(p.get_tra_command()) for p in p_]

    # w3 - level 2 soffit timber
    x = [9, 5*6+3*6/2]
    z = np.full_like(x, 8.5+3.55+1.45/2)
    w = [3*6, 3*6]
    h = np.full_like(x, 1.45)
    t = np.full_like(x, 931)
    p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    [print(p.get_tra_command()) for p in p_]

    # w2 - recessed windows
    # x = [24]
    # z = np.full_like(x, 4.25+3.55/2)
    # w = np.full_like(x, 6)
    # h = np.full_like(x, 3.55)
    # t = np.full_like(x, 1105)
    # local2global_xyz = np.array([0, -45, 0])
    # p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle)
    # [print(p.get_tra_command()) for p in p_]

    # w3 - far end bit
    # angle = (180 + 90 + 75) / 180 * np.pi
    # x = [5.75/2,    ]
    # z = [3.5/2,     ]
    # w = [5.75,      ]
    # h = [3.5,       ]
    # t = np.full_like(x, 1105)
    # local2global_xyz = np.array([7.8, -45, 0])
    # p_ = array_windows(x=x, z=z, w=w, h=h, temperature=t, angle=angle, local2global_xyz=local2global_xyz)
    # [print(p.get_tra_command()) for p in p_]


def w3_receiver():
    angle = (180 + 90 + 9.5) / 180 * np.pi

    # w3 - receiver
    cx__ = [13.5 / 2]
    cz__ = [15 / 2]
    width__ = np.full_like(cx__, 55)
    height__ = np.full_like(cx__, 13.5)
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
    cz__ = [13.5 / 2]
    width__ = np.full_like(cx__, 54)
    height__ = np.full_like(cx__, 13.5)
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
    w3_emitter()
    w2_receiver()

    # w3_receiver()
    # w2_emitter()
