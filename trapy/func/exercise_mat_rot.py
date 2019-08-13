import numpy as np
import math
from typing import Union


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
        self._receiver = True

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


if __name__ == '__main__':

    """
    Type=Emitter
    Geometry=5:0:0 * 10.2:0:0 * 10.2:-1.47:5.39 * 5:-1.47:5.39
    Name=Glazing
    Temperature=1105
    Reverse=FALSE
    Emissivity=1
    <Type=Emitter,Geometry=5:0:0 * 10.2:0:0 * 10.2:-1.47:5.39 * 5:-1.47:5.39,Name=Glazing,Temperature=1105,Reverse=FALSE,Emissivity=1>
    """

    panels = list()

    # w3 - level 0
    edges__ = [0] + list(np.arange(5.4, 50.4+0.1, 3))
    height__ = np.full((len(edges__)-1,), 3.5)
    height__[0] = 3.7
    temperature__ = np.full_like(height__, fill_value=293.15, dtype=float)
    temperature__[[1, 3, 5, 8, 9, 12, 14]] = 1011.
    p__ = list()
    for i, height in enumerate(height__):
        p = Rectangle()
        p.name = f'w2_0_{i}'
        p.local_vertex_1 = [edges__[i], 0, 0]
        p.local_vertex_2 = [edges__[i+1], 0, height]
        p.local2global_rotation_axis = [0, 0, 1]
        p.local2global_rotation_angle = 80.5 / 180 * np.pi
        p.local2global_xyz = [0, 0, 0]
        p.local2global_vertices()
        p__.append(p)

    for p in p__:
        print(p.name , p.global_vertex_1, p.global_vertex_2)